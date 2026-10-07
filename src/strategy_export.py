"""Convert a local_quant_ai strategy (validated JSON) into a standalone
Python class matching the StratGen* format used by the GA optimizer
framework: a class with `domain()`, `validate(p)`, and `signals(df, p)`
static methods, `signals` returning
`(long_entry, long_exit, short_entry, short_exit)` boolean Series.

Short logic is translated when the source strategy defines it
(strategy.short_entry / strategy.short_exit both present); short_entry/
short_exit are emitted as always-False Series only when the source
strategy is genuinely long-only (no short_entry/short_exit at all) --
never inferred by mirroring/inverting the long conditions, since that
would fabricate behavior the source strategy never specified.

The generated file is self-contained: only `numpy`/`pandas` imports, no
dependency on this project's own src/ modules, so it can be dropped
directly into another codebase (e.g. a GA optimizer framework).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.schema import BASE_COLUMNS, INDICATOR_PARAM_BOUNDS, validate_strategy_dict


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return re.sub(r"_+", "_", s) or "strategy"


def _pascal(text: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", text.strip())
    return "".join(p.capitalize() for p in parts if p)


def _num_std_tag(num_std) -> str:
    return str(int(num_std)) if float(num_std).is_integer() else str(num_std)


class _NameAllocator:
    """Hands out unique domain-key / local-variable names, appending _2,
    _3, ... on collision (e.g. two EMAs on the same column)."""

    def __init__(self):
        self._used: set[str] = set()

    def alloc(self, base: str) -> str:
        if base not in self._used:
            self._used.add(base)
            return base
        i = 2
        while f"{base}_{i}" in self._used:
            i += 1
        name = f"{base}_{i}"
        self._used.add(name)
        return name


# --- per-indicator code generators -----------------------------------
# Each returns: (list_of_code_lines, {domain_key: [concrete_value]}, output_columns)
# output_columns maps the deterministic column name(s) (as used by entry/
# exit conditions) to themselves -- they're already valid Python identifiers,
# so no separate variable-name translation table is needed.

def _gen_sma(idx, spec, names: _NameAllocator):
    column = spec.get("column", "close")
    period = int(spec["params"]["period"])
    out = f"sma_{column}_{period}"
    key = names.alloc(f"sma_{column}_period")
    lines = [f"{out} = {column}.rolling(int(p['{key}']), min_periods=int(p['{key}'])).mean()"]
    return lines, {key: [period]}, [out], {key: "period"}


def _gen_ema(idx, spec, names: _NameAllocator):
    column = spec.get("column", "close")
    period = int(spec["params"]["period"])
    out = f"ema_{column}_{period}"
    key = names.alloc(f"ema_{column}_period")
    lines = [f"{out} = {column}.ewm(span=int(p['{key}']), adjust=False, min_periods=int(p['{key}'])).mean()"]
    return lines, {key: [period]}, [out], {key: "period"}


def _gen_rsi(idx, spec, names: _NameAllocator):
    column = spec.get("column", "close")
    period = int(spec["params"]["period"])
    out = f"rsi_{column}_{period}"
    key = names.alloc(f"rsi_{column}_period")
    d, g, l, ag, al, rs = (f"_d{idx}", f"_gain{idx}", f"_loss{idx}", f"_avg_gain{idx}", f"_avg_loss{idx}", f"_rs{idx}")
    lines = [
        f"{d} = {column}.diff()",
        f"{g} = {d}.clip(lower=0.0)",
        f"{l} = -{d}.clip(upper=0.0)",
        f"{ag} = {g}.ewm(alpha=1/int(p['{key}']), adjust=False, min_periods=int(p['{key}'])).mean()",
        f"{al} = {l}.ewm(alpha=1/int(p['{key}']), adjust=False, min_periods=int(p['{key}'])).mean()",
        f"{rs} = {ag} / {al}.replace(0.0, np.nan)",
        f"{out} = 100 - (100 / (1 + {rs}))",
        f"{out} = {out}.fillna(100.0).where({al} != 0, 100.0)",
    ]
    return lines, {key: [period]}, [out], {key: "period"}


def _gen_roc(idx, spec, names: _NameAllocator):
    column = spec.get("column", "close")
    period = int(spec["params"]["period"])
    out = f"roc_{column}_{period}"
    key = names.alloc(f"roc_{column}_period")
    lines = [f"{out} = {column}.pct_change(periods=int(p['{key}'])) * 100.0"]
    return lines, {key: [period]}, [out], {key: "period"}


def _gen_macd(idx, spec, names: _NameAllocator):
    column = spec.get("column", "close")
    fast, slow, signal = (int(spec["params"]["fast"]), int(spec["params"]["slow"]), int(spec["params"]["signal"]))
    prefix = f"macd_{fast}_{slow}_{signal}"
    kf = names.alloc(f"macd_{column}_fast")
    ks = names.alloc(f"macd_{column}_slow")
    kg = names.alloc(f"macd_{column}_signal")
    ef, es = f"_ema_fast{idx}", f"_ema_slow{idx}"
    lines = [
        f"{ef} = {column}.ewm(span=int(p['{kf}']), adjust=False, min_periods=int(p['{kf}'])).mean()",
        f"{es} = {column}.ewm(span=int(p['{ks}']), adjust=False, min_periods=int(p['{ks}'])).mean()",
        f"{prefix}_line = {ef} - {es}",
        f"{prefix}_signal = {prefix}_line.ewm(span=int(p['{kg}']), adjust=False, min_periods=int(p['{kg}'])).mean()",
        f"{prefix}_hist = {prefix}_line - {prefix}_signal",
        f"{prefix} = {prefix}_line",
    ]
    out = [f"{prefix}_line", f"{prefix}_signal", f"{prefix}_hist", prefix]
    return lines, {kf: [fast], ks: [slow], kg: [signal]}, out, {kf: "fast", ks: "slow", kg: "signal"}


def _gen_atr(idx, spec, names: _NameAllocator):
    period = int(spec["params"]["period"])
    out = f"atr_{period}"
    key = names.alloc("atr_period")
    pc, tr = f"_prev_close{idx}", f"_tr{idx}"
    lines = [
        f"{pc} = close.shift(1)",
        f"{tr} = pd.concat([high - low, (high - {pc}).abs(), (low - {pc}).abs()], axis=1).max(axis=1)",
        f"{out} = {tr}.ewm(alpha=1/int(p['{key}']), adjust=False, min_periods=int(p['{key}'])).mean()",
    ]
    return lines, {key: [period]}, [out], {key: "period"}


def _gen_bollinger_bands(idx, spec, names: _NameAllocator):
    column = spec.get("column", "close")
    period = int(spec["params"]["period"])
    num_std = spec["params"].get("num_std", 2.0)
    prefix = f"bb_{period}_{_num_std_tag(num_std)}"
    kp = names.alloc(f"bb_{column}_period")
    ks = names.alloc(f"bb_{column}_num_std")
    mid, std = f"_bb_mid{idx}", f"_bb_std{idx}"
    lines = [
        f"{mid} = {column}.rolling(int(p['{kp}']), min_periods=int(p['{kp}'])).mean()",
        f"{std} = {column}.rolling(int(p['{kp}']), min_periods=int(p['{kp}'])).std(ddof=0)",
        f"{prefix}_middle = {mid}",
        f"{prefix}_upper = {mid} + p['{ks}'] * {std}",
        f"{prefix}_lower = {mid} - p['{ks}'] * {std}",
    ]
    out = [f"{prefix}_middle", f"{prefix}_upper", f"{prefix}_lower"]
    return lines, {kp: [period], ks: [num_std]}, out, {kp: "period", ks: "num_std"}


def _gen_donchian(idx, spec, names: _NameAllocator):
    period = int(spec["params"]["period"])
    prefix = f"donchian_{period}"
    key = names.alloc("donchian_period")
    lines = [
        # Prior-bars-only (shift(1)) -- matches local_quant_ai's fix so
        # breakout conditions are actually satisfiable (see ASSUMPTIONS.md #16).
        f"{prefix}_upper = high.shift(1).rolling(int(p['{key}']), min_periods=int(p['{key}'])).max()",
        f"{prefix}_lower = low.shift(1).rolling(int(p['{key}']), min_periods=int(p['{key}'])).min()",
        f"{prefix}_middle = ({prefix}_upper + {prefix}_lower) / 2.0",
    ]
    out = [f"{prefix}_upper", f"{prefix}_lower", f"{prefix}_middle"]
    return lines, {key: [period]}, out, {key: "period"}


def _gen_stochastic(idx, spec, names: _NameAllocator):
    period, sk, sd = (int(spec["params"]["period"]), int(spec["params"]["smooth_k"]), int(spec["params"]["smooth_d"]))
    prefix = f"stoch_{period}_{sk}_{sd}"
    kp = names.alloc("stoch_period")
    kk = names.alloc("stoch_smooth_k")
    kd = names.alloc("stoch_smooth_d")
    ll, hh, den, rk = f"_lowest_low{idx}", f"_highest_high{idx}", f"_denom{idx}", f"_raw_k{idx}"
    lines = [
        f"{ll} = low.rolling(int(p['{kp}']), min_periods=int(p['{kp}'])).min()",
        f"{hh} = high.rolling(int(p['{kp}']), min_periods=int(p['{kp}'])).max()",
        f"{den} = ({hh} - {ll}).replace(0.0, np.nan)",
        f"{rk} = 100 * (close - {ll}) / {den}",
        f"{prefix}_k = {rk}.rolling(int(p['{kk}']), min_periods=int(p['{kk}'])).mean()",
        f"{prefix}_d = {prefix}_k.rolling(int(p['{kd}']), min_periods=int(p['{kd}'])).mean()",
    ]
    out = [f"{prefix}_k", f"{prefix}_d"]
    return lines, {kp: [period], kk: [sk], kd: [sd]}, out, {kp: "period", kk: "smooth_k", kd: "smooth_d"}


def _gen_adx(idx, spec, names: _NameAllocator):
    period = int(spec["params"]["period"])
    key = names.alloc("adx_period")
    um, dm, pdm, mdm, pc, tr, ats, dx = (
        f"_up_move{idx}", f"_down_move{idx}", f"_plus_dm{idx}", f"_minus_dm{idx}",
        f"_prev_close{idx}", f"_tr{idx}", f"_atr_smooth{idx}", f"_dx{idx}",
    )
    lines = [
        f"{um} = high.diff()",
        f"{dm} = -low.diff()",
        f"{pdm} = (({um} > {dm}) & ({um} > 0)).astype(float) * {um}.clip(lower=0.0)",
        f"{mdm} = (({dm} > {um}) & ({dm} > 0)).astype(float) * {dm}.clip(lower=0.0)",
        f"{pc} = close.shift(1)",
        f"{tr} = pd.concat([high - low, (high - {pc}).abs(), (low - {pc}).abs()], axis=1).max(axis=1)",
        f"{ats} = {tr}.ewm(alpha=1/int(p['{key}']), adjust=False, min_periods=int(p['{key}'])).mean()",
        f"plus_di_{period} = 100 * {pdm}.ewm(alpha=1/int(p['{key}']), adjust=False, min_periods=int(p['{key}'])).mean() / {ats}",
        f"minus_di_{period} = 100 * {mdm}.ewm(alpha=1/int(p['{key}']), adjust=False, min_periods=int(p['{key}'])).mean() / {ats}",
        f"{dx} = 100 * (plus_di_{period} - minus_di_{period}).abs() / (plus_di_{period} + minus_di_{period}).replace(0.0, np.nan)",
        f"adx_{period} = {dx}.ewm(alpha=1/int(p['{key}']), adjust=False, min_periods=int(p['{key}'])).mean()",
    ]
    out = [f"adx_{period}", f"plus_di_{period}", f"minus_di_{period}"]
    return lines, {key: [period]}, out, {key: "period"}


_GENERATORS = {
    "sma": _gen_sma, "ema": _gen_ema, "rsi": _gen_rsi, "roc": _gen_roc,
    "macd": _gen_macd, "atr": _gen_atr, "bollinger_bands": _gen_bollinger_bands,
    "donchian": _gen_donchian, "stochastic": _gen_stochastic, "adx": _gen_adx,
}

_INDICATOR_REQUIRED_COLS = {
    "sma": set(), "ema": set(), "rsi": set(), "roc": set(), "macd": set(),
    "atr": {"high", "low", "close"},
    "bollinger_bands": set(),
    "donchian": {"high", "low"},
    "stochastic": {"high", "low", "close"},
    "adx": {"high", "low", "close"},
}


def _condition_expr(cond: dict) -> str:
    left = cond["left"]
    op = cond["op"]
    right = cond["right"]
    right_is_numeric = isinstance(right, (int, float))
    right_expr = repr(right) if right_is_numeric else right
    if op in (">", "<", ">=", "<="):
        return f"({left} {op} {right_expr})"
    if op in ("cross_above", "cross_below"):
        cmp_now = ">" if op == "cross_above" else "<"
        cmp_prev = "<=" if op == "cross_above" else ">="
        prev_right = right_expr if right_is_numeric else f"{right_expr}.shift(1)"
        return f"(({left}.shift(1) {cmp_prev} {prev_right}) & ({left} {cmp_now} {right_expr}))"
    raise ValueError(f"Unsupported operator: {op}")


def _group_expr(group: dict) -> str:
    joiner = " & " if group["logic"] == "and" else " | "
    parts = [_condition_expr(c) for c in group["conditions"]]
    return "(" + joiner.join(parts) + ")"


def _widen_int_domain(value: int, lo: int, hi: int) -> list[int]:
    """Build a logical search neighborhood around an integer parameter value:
    a handful of multiplicative steps (halving/doubling-style) plus small
    additive steps, clamped to the same bounds src/schema.py enforces at
    generation time. Always includes the original value."""
    candidates = {value}
    for factor in (0.5, 0.75, 1.25, 1.5, 2.0):
        candidates.add(round(value * factor))
    for delta in (-2, -1, 1, 2):
        candidates.add(value + delta)
    candidates = {max(lo, min(hi, c)) for c in candidates if c >= 1}
    return sorted(candidates)


def _widen_float_domain(value: float, lo: float, hi: float, step: float = 0.5) -> list[float]:
    """Same idea as `_widen_int_domain` but for continuous params (currently
    only bollinger_bands' num_std), stepping by `step` in both directions."""
    candidates = {round(value, 4)}
    for k in (-2, -1, 1, 2):
        candidates.add(round(value + k * step, 4))
    candidates = {round(max(lo, min(hi, c)), 4) for c in candidates}
    return sorted(candidates)


def _widen_domain_value(value, bound_param: str, bounds: tuple[float, float]) -> list:
    lo, hi = bounds
    if bound_param == "num_std":
        return _widen_float_domain(float(value), float(lo), float(hi))
    return _widen_int_domain(int(value), int(lo), int(hi))


def generate_class_source(strategy_raw: dict) -> str:
    """Take a strategy dict already validated by src.schema.validate_strategy_dict
    and return the full Python source of a standalone StratGen*-style class."""
    strategy, problems = validate_strategy_dict(strategy_raw)
    if strategy is None:
        raise ValueError(f"Strategy is not valid, cannot export: {problems}")

    class_name = f"StratGen{_pascal(strategy.strategy_name)}"
    mode = _slug(strategy.strategy_name)
    tag = "g" + "".join(w[0] for w in mode.split("_"))[:10]

    names = _NameAllocator()
    code_lines: list[str] = []
    domain: dict[str, list] = {}
    default_params: dict[str, Any] = {}
    validate_asserts: list[str] = []
    required_cols: set[str] = set()

    for idx, ind in enumerate(strategy.indicators):
        spec = {"name": ind.name, "column": ind.column, "params": ind.params}
        gen = _GENERATORS[ind.name]
        lines, dom, _out_cols, key_to_bound_param = gen(idx, spec, names)
        code_lines.extend(lines)
        domain.update(dom)
        required_cols |= _INDICATOR_REQUIRED_COLS[ind.name]
        if ind.column:
            required_cols.add(ind.column)
        bounds = INDICATOR_PARAM_BOUNDS.get(ind.name, {})
        for key, bound_param in key_to_bound_param.items():
            # Capture the source strategy's actual concrete value before it's
            # overwritten below by the widened search domain -- callers that
            # want to reproduce the strategy as originally generated (rather
            # than probe the search neighborhood) need this, since domain()[0]
            # is just the smallest widened candidate and can be a degenerate
            # edge case (e.g. stoch smooth_d=1 collapses K==D).
            default_params[key] = domain[key][0]
            if bound_param in bounds:
                lo, hi = bounds[bound_param]
                validate_asserts.append(f"assert {lo} <= p['{key}'] <= {hi}, \"{key} out of bounds\"")
                # Widen from the single concrete value to a logical search
                # neighborhood, still clamped to the same bounds (see
                # _widen_int_domain / _widen_float_domain docstrings).
                domain[key] = _widen_domain_value(domain[key][0], bound_param, bounds[bound_param])

    groups_for_cols = [strategy.entry, strategy.exit]
    if strategy.has_short_logic:
        groups_for_cols += [strategy.short_entry, strategy.short_exit]
    for group in groups_for_cols:
        for cond in group.conditions:
            for side in (cond.left, cond.right):
                if isinstance(side, str) and side in BASE_COLUMNS:
                    required_cols.add(side)

    entry_expr = _group_expr(strategy.entry.model_dump())
    exit_expr = _group_expr(strategy.exit.model_dump())
    if strategy.has_short_logic:
        short_entry_expr = _group_expr(strategy.short_entry.model_dump())
        short_exit_expr = _group_expr(strategy.short_exit.model_dump())
    else:
        short_entry_expr = None
        short_exit_expr = None

    required_cols_sorted = sorted(required_cols) or ["close"]
    base_col_lines = [f'{c} = df["{c}"]' for c in ["open", "high", "low", "close", "volume"] if c in required_cols_sorted]
    if "close" not in required_cols_sorted:
        base_col_lines.append('close = df["close"]  # always available for reference')

    domain_lines = ",\n            ".join(f'"{k}": {v!r}' for k, v in domain.items())
    default_params_lines = ",\n            ".join(f'"{k}": {v!r}' for k, v in default_params.items())
    validate_lines = "\n        ".join(validate_asserts) if validate_asserts else "pass  # no numeric bounds to check"
    indicator_block = "\n        ".join(code_lines)

    rationale = strategy.rationale.replace('"""', "'''")

    if strategy.has_short_logic:
        short_signal_lines = (
            f"se = ({short_entry_expr}).fillna(False)\n"
            f"        sx = ({short_exit_expr}).fillna(False)"
        )
    else:
        short_signal_lines = (
            "# Always-False: source strategy has no short-side logic (see class docstring).\n"
            "        se = pd.Series(False, index=df.index)\n"
            "        sx = pd.Series(False, index=df.index)"
        )

    class_short_note = (
        "Trades both long and short (short_entry/short_exit are real rules from the source strategy)."
        if strategy.has_short_logic else
        "Long-only: source strategy defines no short_entry/short_exit, so short_entry/short_exit "
        "below are always-False -- NOT inferred by mirroring the long conditions."
    )

    source = f'''from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

Signals = Tuple[pd.Series, pd.Series, pd.Series, pd.Series]


class {class_name}:
    """
    {strategy.strategy_name} ({strategy.symbol}, {strategy.timeframe})

    {rationale}

    Auto-generated from a local_quant_ai strategy JSON. {class_short_note}

    stop_loss_pct={strategy.stop_loss_pct}, take_profit_pct={strategy.take_profit_pct},
    position_sizing={strategy.position_sizing!r}, risk_per_trade={strategy.risk_per_trade}
    (risk/exit-price management is not part of this class -- wire it through
    your own backtester's position sizing / stop-loss handling as usual.)
    """

    MODE = "{mode}"
    TAG = "{tag}"
    REQUIRED_COLS = {required_cols_sorted!r}
    PRICE_COL = "close"

    @staticmethod
    def domain() -> Dict[str, List[Any]]:
        """Search neighborhood around the source strategy's actual parameter
        values (not just the single value it used) -- multiplicative and
        small additive steps, clamped to local_quant_ai's own bounds for
        each parameter. See _widen_int_domain/_widen_float_domain in
        local_quant_ai's src/strategy_export.py for exactly how these were
        generated."""
        return {{
            {domain_lines}
        }}

    @staticmethod
    def default_params() -> Dict[str, Any]:
        """The source strategy's actual concrete parameter values (not a
        widened neighborhood). Use this -- not `{{k: v[0] for k, v in
        domain().items()}}` -- when you want to reproduce/sanity-check the
        strategy as originally generated: domain()[0] is just the smallest
        widened candidate for each parameter independently and can land on a
        degenerate combination (e.g. a stochastic smooth_d of 1 makes %K and
        %D identical, so crossover conditions can never fire)."""
        return {{
            {default_params_lines}
        }}

    @staticmethod
    def validate(p: Dict[str, Any]) -> bool:
        {validate_lines}
        return True

    @staticmethod
    def signals(df: pd.DataFrame, p: Dict[str, Any]) -> Signals:
        {chr(10).join("        " + l for l in base_col_lines).strip()}

        {indicator_block}

        le = ({entry_expr}).fillna(False)
        lx = ({exit_expr}).fillna(False)
        {short_signal_lines}

        return (
            le.astype(bool),
            lx.astype(bool),
            se.astype(bool),
            sx.astype(bool)
        )
'''
    return source


def export_strategy_file(strategy_json_path: str | Path, output_path: str | Path | None = None) -> Path:
    strategy_json_path = Path(strategy_json_path)
    raw = json.loads(strategy_json_path.read_text(encoding="utf-8"))
    source = generate_class_source(raw)

    if output_path is None:
        slug = _slug(raw.get("strategy_name", "strategy"))
        output_path = strategy_json_path.parent / f"strat_gen_{slug}.py"
    output_path = Path(output_path)
    output_path.write_text(source, encoding="utf-8")
    return output_path
