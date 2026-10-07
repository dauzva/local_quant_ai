"""Static + runtime sandboxing for LLM-generated Python strategy files.

The generation pipeline (src/strategy_codegen.py) now asks the LLM to return
a COMPLETE, self-contained Python file per strategy (a StratGen*-style class
with domain()/default_params()/validate(p)/signals(df, p)) instead of a JSON
DSL. That is far more expressive than the old fixed-indicator JSON schema,
but it means we are about to `exec()` text an LLM wrote -- something the
project's original design explicitly promised never to do.

This module is the replacement safety boundary:

1. `static_check(source)` -- an AST allowlist. Only `from __future__ import
   annotations`, `from typing import ...`, and `import numpy as np` /
   `import pandas as pd` / `import math` are permitted at module level,
   followed by exactly one class definition. Inside that class, dangerous
   builtins (eval/exec/open/__import__/getattr/...), any import statement,
   dunder attribute access, and `with`/`global`/`nonlocal`/`del` are all
   rejected outright.
2. `compile_strategy(source)` -- exec's the (already statically-checked)
   source with a restricted builtins dict and a restricted `__import__`
   that only allows the same tiny module allowlist, then runs a bounded
   smoke test (validate() + signals() on synthetic OHLCV data, in a worker
   thread with a timeout so a runaway `while True` can't hang the run)
   before handing back a CompiledStrategy.

This is defense-in-depth for a local research tool running free/small LLM
output, not a hard multi-tenant security boundary -- see README.md.
"""
from __future__ import annotations

import ast
import builtins as _builtins_module
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("strategy_sandbox")

ALLOWED_MODULE_IMPORTS = {"numpy", "pandas", "math"}
ALLOWED_TOP_LEVEL_MODULES = ALLOWED_MODULE_IMPORTS | {"__future__", "typing"}

FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "hasattr", "input", "help",
    "breakpoint", "exit", "quit", "memoryview", "__builtins__",
}

REQUIRED_STATIC_METHODS = {"domain", "default_params", "validate", "signals"}
REQUIRED_CLASS_CONSTANTS = {
    "STRATEGY_NAME", "RATIONALE", "MODE", "TAG", "REQUIRED_COLS", "PRICE_COL",
    "STOP_LOSS_PCT", "TAKE_PROFIT_PCT", "RISK_PER_TRADE", "POSITION_SIZING",
}
BASE_COLUMNS = {"open", "high", "low", "close", "volume"}
SMOKE_TEST_TIMEOUT_SECONDS = 8
SMOKE_TEST_BARS = 400


class StrategySandboxError(Exception):
    """Raised with a list of human-readable problems in .problems."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass
class CompiledStrategy:
    strategy_name: str
    rationale: str
    mode: str
    tag: str
    required_cols: list[str]
    price_col: str
    stop_loss_pct: float | None
    take_profit_pct: float | None
    risk_per_trade: float
    position_sizing: str
    domain: dict[str, list]
    params: dict[str, Any]
    cls: type
    source: str
    class_name: str
    signature: str = field(default="")

    def signals(self, df: pd.DataFrame):
        return self.cls.signals(df, self.params)


# --------------------------------------------------------------------------
# 1. Static AST allowlist
# --------------------------------------------------------------------------

def _is_future_annotations_import(node: ast.ImportFrom) -> bool:
    return node.module == "__future__" and any(a.name == "annotations" for a in node.names)


def _module_level_problems(tree: ast.Module) -> tuple[list[str], ast.ClassDef | None]:
    problems: list[str] = []
    class_defs = []

    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and i == 0:
            continue  # module docstring, if any
        if isinstance(node, ast.ImportFrom):
            if _is_future_annotations_import(node):
                continue
            if node.module == "typing":
                continue
            problems.append(f"Disallowed top-level import: 'from {node.module} import ...' (only __future__.annotations and typing are allowed)")
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULE_IMPORTS:
                    problems.append(f"Disallowed top-level import: 'import {alias.name}' (allowed: {sorted(ALLOWED_MODULE_IMPORTS)})")
            continue
        if isinstance(node, ast.ClassDef):
            class_defs.append(node)
            continue
        problems.append(f"Disallowed top-level statement: {type(node).__name__} (only imports and exactly one class definition are allowed)")

    if len(class_defs) != 1:
        problems.append(f"Expected exactly 1 top-level class definition, found {len(class_defs)}")
        return problems, None

    return problems, class_defs[0]


def _class_body_problems(class_node: ast.ClassDef) -> list[str]:
    problems: list[str] = []

    for node in ast.walk(class_node):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            problems.append("Import statements are not allowed inside the class body")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append("'global'/'nonlocal' are not allowed")
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            problems.append("'with' statements are not allowed")
        elif isinstance(node, ast.Delete):
            problems.append("'del' statements are not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"Use of forbidden name '{node.id}'")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            problems.append(f"Dunder attribute access is not allowed: '.{node.attr}'")

    method_names = {
        n.name for n in class_node.body
        if isinstance(n, ast.FunctionDef)
    }
    missing = REQUIRED_STATIC_METHODS - method_names
    if missing:
        problems.append(f"Missing required static method(s): {sorted(missing)}")

    for n in class_node.body:
        if isinstance(n, ast.FunctionDef) and n.name in REQUIRED_STATIC_METHODS:
            has_staticmethod = any(
                isinstance(d, ast.Name) and d.id == "staticmethod" for d in n.decorator_list
            )
            if not has_staticmethod:
                problems.append(f"Method '{n.name}' must be decorated with @staticmethod")

    assigned_constants = set()
    for n in class_node.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    assigned_constants.add(t.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            assigned_constants.add(n.target.id)
    missing_constants = REQUIRED_CLASS_CONSTANTS - assigned_constants
    if missing_constants:
        problems.append(f"Missing required class constant(s): {sorted(missing_constants)}")

    return problems


def static_check(source: str) -> tuple[list[str], ast.ClassDef | None]:
    """Return (problems, class_node). class_node is None if the module-level
    structure itself is invalid (problems will always be non-empty then)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"], None

    problems, class_node = _module_level_problems(tree)
    if class_node is not None:
        problems += _class_body_problems(class_node)
    return problems, class_node


# --------------------------------------------------------------------------
# 2. Restricted exec + smoke test
# --------------------------------------------------------------------------

_SAFE_BUILTIN_NAMES = (
    "abs", "min", "max", "sum", "len", "range", "enumerate", "zip", "sorted",
    "reversed", "int", "float", "str", "bool", "list", "dict", "tuple", "set",
    "frozenset", "isinstance", "round", "map", "filter", "any", "all",
    "print", "type", "iter", "next", "slice",
    "staticmethod", "__build_class__",
    "True", "False", "None",
    "Exception", "ValueError", "TypeError", "AssertionError", "KeyError",
    "IndexError", "ZeroDivisionError", "ArithmeticError", "StopIteration",
    "RuntimeError", "NotImplementedError",
)


def _restricted_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    # numpy/pandas lazily import their own submodules at call time (e.g.
    # `numpy._core._methods` inside np.std) through whatever __import__ is in
    # scope; judge by root package so those internals are not mistaken for the
    # strategy importing something forbidden. The static check already
    # guarantees the strategy's OWN import statements are on the allowlist.
    if level != 0 or name.split(".")[0] not in ALLOWED_TOP_LEVEL_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed in a generated strategy file")
    return _builtins_module.__import__(name, globals, locals, fromlist, level)


def _make_safe_globals() -> dict:
    safe_builtins = {n: getattr(_builtins_module, n) for n in _SAFE_BUILTIN_NAMES if hasattr(_builtins_module, n)}
    safe_builtins["__import__"] = _restricted_import
    # `__name__` is read implicitly by every `class ...:` statement (to set
    # __module__/__qualname__) -- a plain global, not a builtin.
    return {"__builtins__": safe_builtins, "__name__": "generated_strategy"}


def _synthetic_ohlcv(n: int = SMOKE_TEST_BARS) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    steps = rng.normal(loc=0.0002, scale=0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + steps)
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.004, size=n)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.004, size=n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.integers(1000, 50000, size=n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", name="datetime")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def _find_class_object(namespace: dict, class_name: str) -> type:
    obj = namespace.get(class_name)
    if obj is None or not isinstance(obj, type):
        raise StrategySandboxError([f"Class '{class_name}' not found after exec"])
    return obj


def _run_smoke_test(cls: type) -> dict:
    """Runs validate()/signals() on synthetic data in a worker thread with a
    timeout, so a pathological generated strategy (accidental infinite loop,
    O(n^2) blowup) can't hang the whole research loop.

    Uses a bare daemon Thread (NOT ThreadPoolExecutor): a pool's __exit__
    calls shutdown(wait=True), which blocks forever waiting for a thread
    stuck in a genuine `while True`, defeating the timeout entirely. A
    daemon thread is simply abandoned on timeout instead (Python has no
    safe cross-platform hard-kill for a running thread) -- acceptable for a
    local tool where this is a rare, self-inflicted failure mode, not an
    adversarial one; the daemon flag ensures it can't keep the whole process
    alive either."""
    result_box: dict = {}

    def _work():
        try:
            params = cls.default_params()
            if not isinstance(params, dict):
                raise StrategySandboxError(["default_params() must return a dict"])
            if cls.validate(params) is not True:
                raise StrategySandboxError(["validate(default_params()) did not return True"])
            df = _synthetic_ohlcv()
            result = cls.signals(df, params)
            if not (isinstance(result, tuple) and len(result) == 4):
                raise StrategySandboxError(["signals() must return a 4-tuple (long_entry, long_exit, short_entry, short_exit)"])
            for series in result:
                if not isinstance(series, pd.Series):
                    raise StrategySandboxError(["Each element returned by signals() must be a pandas Series"])
                if not series.index.equals(df.index):
                    raise StrategySandboxError(["Each series returned by signals() must share df's index"])
            # Dynamic lookahead test: a causal strategy's signal on bar t cannot
            # change when bars after t are removed. Catches shift(-k),
            # rolling(center=True), full-series normalisation (x / x.max(),
            # whole-series ranks/z-scores), negative parameters, anything.
            cut = int(len(df) * 0.75)
            truncated = cls.signals(df.iloc[:cut].copy(), params)
            names = ("long_entry", "long_exit", "short_entry", "short_exit")
            for name, full_s, part_s in zip(names, result, truncated):
                a = full_s.iloc[:cut].fillna(False).to_numpy(dtype=bool)
                b = part_s.iloc[:cut].fillna(False).to_numpy(dtype=bool)
                if a.shape != b.shape or not np.array_equal(a, b):
                    raise StrategySandboxError([
                        f"LOOKAHEAD: {name} signals on past bars change when future bars are removed "
                        "(uses future data, e.g. shift(-k), center=True, or whole-series max/min/mean/rank)"])
            result_box["ok"] = {"domain": cls.domain(), "params": params}
        except BaseException as exc:  # capture for the parent thread, including StrategySandboxError
            result_box["error"] = exc

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    thread.join(timeout=SMOKE_TEST_TIMEOUT_SECONDS)

    if thread.is_alive():
        raise StrategySandboxError([f"signals()/validate() did not finish within {SMOKE_TEST_TIMEOUT_SECONDS}s (possible infinite loop)"])
    if "error" in result_box:
        exc = result_box["error"]
        if isinstance(exc, StrategySandboxError):
            raise exc
        raise StrategySandboxError([f"Smoke test raised: {exc!r}"])
    return result_box["ok"]


def _structural_signature(class_node: ast.ClassDef) -> str:
    """Hash of the *logic* methods only (domain/validate/signals), so two
    strategies that differ only in name/rationale/tag collapse to the same
    signature for near-duplicate detection (see src/memory.py)."""
    parts = []
    for n in class_node.body:
        if isinstance(n, ast.FunctionDef) and n.name in ("domain", "validate", "signals"):
            parts.append(ast.dump(n, annotate_fields=False, include_attributes=False))
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return digest[:24]


def compile_strategy(source: str) -> CompiledStrategy:
    """Full pipeline: static AST check -> restricted exec -> smoke test.
    Raises StrategySandboxError with all collected problems on any failure.
    """
    problems, class_node = static_check(source)
    if problems or class_node is None:
        raise StrategySandboxError(problems or ["Unknown static-check failure"])

    class_name = class_node.name
    namespace = _make_safe_globals()
    try:
        code_obj = compile(source, filename="<generated_strategy>", mode="exec")
        exec(code_obj, namespace)  # noqa: S102 -- vetted above; restricted builtins/imports
    except StrategySandboxError:
        raise
    except Exception as exc:
        raise StrategySandboxError([f"Error executing strategy module: {exc!r}"]) from exc

    cls = _find_class_object(namespace, class_name)

    try:
        smoke = _run_smoke_test(cls)
    except StrategySandboxError:
        raise
    except Exception as exc:
        raise StrategySandboxError([f"Smoke test raised: {exc!r}"]) from exc

    required_cols = list(getattr(cls, "REQUIRED_COLS", []) or [])
    if not set(required_cols) <= BASE_COLUMNS:
        raise StrategySandboxError([f"REQUIRED_COLS must be a subset of {sorted(BASE_COLUMNS)}, got {required_cols}"])
    price_col = getattr(cls, "PRICE_COL", "close")
    if price_col not in BASE_COLUMNS:
        raise StrategySandboxError([f"PRICE_COL must be one of {sorted(BASE_COLUMNS)}, got {price_col!r}"])

    stop_loss_pct = getattr(cls, "STOP_LOSS_PCT", None)
    if stop_loss_pct is not None and not (0.0 < float(stop_loss_pct) <= 0.5):
        raise StrategySandboxError([f"STOP_LOSS_PCT must be None or in (0, 0.5], got {stop_loss_pct}"])
    take_profit_pct = getattr(cls, "TAKE_PROFIT_PCT", None)
    if take_profit_pct is not None and not (0.0 < float(take_profit_pct) <= 2.0):
        raise StrategySandboxError([f"TAKE_PROFIT_PCT must be None or in (0, 2.0], got {take_profit_pct}"])
    risk_per_trade = float(getattr(cls, "RISK_PER_TRADE", 0.01))
    if not (0.0 < risk_per_trade <= 0.1):
        raise StrategySandboxError([f"RISK_PER_TRADE must be in (0, 0.1], got {risk_per_trade}"])

    strategy_name = str(getattr(cls, "STRATEGY_NAME", class_name))

    return CompiledStrategy(
        strategy_name=strategy_name,
        rationale=str(getattr(cls, "RATIONALE", "")),
        mode=str(getattr(cls, "MODE", "")),
        tag=str(getattr(cls, "TAG", "")),
        required_cols=required_cols,
        price_col=price_col,
        stop_loss_pct=float(stop_loss_pct) if stop_loss_pct is not None else None,
        take_profit_pct=float(take_profit_pct) if take_profit_pct is not None else None,
        risk_per_trade=risk_per_trade,
        position_sizing=str(getattr(cls, "POSITION_SIZING", "risk_based_contracts")),
        domain=dict(smoke["domain"]),
        params=dict(smoke["params"]),
        cls=cls,
        source=source,
        class_name=class_name,
        signature=_structural_signature(class_node),
    )
