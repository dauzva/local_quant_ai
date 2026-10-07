"""Compact strategy generation: prompts, snippet parsing, and the wrapper that
turns an LLM-written snippet into a complete, standalone strategy class.

Why snippets instead of whole files: the old contract made the model write ~1000
tokens of boilerplate (10 class constants, domain()/default_params()/validate())
per strategy, which is where truncation, formatting mistakes and most of the
token bill came from. Now the model writes only the idea in ~150-250 tokens:

    ===S===
    name: rsi_dip_in_uptrend
    family: meanrev
    idea: buy oversold dips above the long trend
    sl: 0.02
    tp: 0.04
    params: {"rsi_n": 5, "lo": 25, "trend_n": 100}
    ```python
    r = rsi(close, p["rsi_n"])
    le = (close > sma(close, p["trend_n"])) & (r < p["lo"])
    lx = r > 60
    ```

`build_source` wraps that into the exact class contract the sandbox
(src/strategy_sandbox.py) already enforces -- the helper indicators the snippet
calls (rsi, sma, atr, ...) are inlined into signals() so every saved strategy.py
stays fully standalone (numpy/pandas only). Ten strategies fit comfortably in
one completion, so one request yields 10 candidates instead of 3.
"""
from __future__ import annotations

import ast
import json
import random
import re
import textwrap
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Helper indicator library (inlined into signals() on demand)
# --------------------------------------------------------------------------

# name -> (signature shown to the LLM, source, dependencies)
_HELPER_DEFS: dict[str, tuple[str, str, list[str]]] = {
    "sma": ("sma(s, n)", '''def sma(s, n):
    return s.rolling(int(n), min_periods=int(n)).mean()''', []),
    "ema": ("ema(s, n)", '''def ema(s, n):
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()''', []),
    "stdev": ("stdev(s, n)", '''def stdev(s, n):
    return s.rolling(int(n), min_periods=int(n)).std(ddof=0)''', []),
    "zscore": ("zscore(s, n)  # (s-mean)/std over n bars", '''def zscore(s, n):
    m = s.rolling(int(n), min_periods=int(n)).mean()
    sd = s.rolling(int(n), min_periods=int(n)).std(ddof=0)
    return (s - m) / sd.replace(0.0, np.nan)''', []),
    "roc": ("roc(s, n)  # n-bar fractional return", '''def roc(s, n):
    return s.pct_change(int(n))''', []),
    "rsi": ("rsi(s, n)  # Wilder RSI 0-100", '''def rsi(s, n):
    d = s.diff()
    g = d.clip(lower=0.0)
    l = -d.clip(upper=0.0)
    ag = g.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean()
    al = l.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean()
    rs = ag / al.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).where(al != 0, 100.0)''', []),
    "atr": ("atr(n)  # Wilder ATR in price units", '''def atr(n):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean()''', []),
    "adx": ("adx(n)  # Wilder ADX 0-100 (trend strength)", '''def adx(n):
    up = high.diff()
    dn = -low.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    a = atr(n)
    pdi = 100.0 * pdm.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean() / a
    mdi = 100.0 * mdm.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean() / a
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / int(n), adjust=False, min_periods=int(n)).mean()''', ["atr"]),
    "macd": ("macd(s, fast, slow, sig)  # -> (line, signal, hist)", '''def macd(s, fast, slow, sig):
    line = ema(s, fast) - ema(s, slow)
    signal = line.ewm(span=int(sig), adjust=False, min_periods=int(sig)).mean()
    return line, signal, line - signal''', ["ema"]),
    "stoch": ("stoch(n, k=3)  # smoothed %K 0-100", '''def stoch(n, k=3):
    ll_ = low.rolling(int(n), min_periods=int(n)).min()
    hh_ = high.rolling(int(n), min_periods=int(n)).max()
    raw = 100.0 * (close - ll_) / (hh_ - ll_).replace(0.0, np.nan)
    return raw.rolling(int(k), min_periods=int(k)).mean()''', []),
    "hh": ("hh(n)  # highest high of the PRIOR n bars (excludes today)", '''def hh(n):
    return high.shift(1).rolling(int(n), min_periods=int(n)).max()''', []),
    "ll": ("ll(n)  # lowest low of the PRIOR n bars (excludes today)", '''def ll(n):
    return low.shift(1).rolling(int(n), min_periods=int(n)).min()''', []),
    "clv": ("clv()  # close location in today's range, 0-1", '''def clv():
    return ((close - low) / (high - low).replace(0.0, np.nan)).fillna(0.5)''', []),
    "crossover": ("crossover(a, b)  # a crosses above b (b may be a number)", '''def crossover(a, b):
    b = pd.Series(b, index=a.index) if np.isscalar(b) else b
    return (a > b) & (a.shift(1) <= b.shift(1))''', []),
    "crossunder": ("crossunder(a, b)  # a crosses below b (b may be a number)", '''def crossunder(a, b):
    b = pd.Series(b, index=a.index) if np.isscalar(b) else b
    return (a < b) & (a.shift(1) >= b.shift(1))''', []),
    "pct_rank": ("pct_rank(s, n)  # 0-1 rank of today's value within last n bars", '''def pct_rank(s, n):
    n = int(n)
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        w = np.lib.stride_tricks.sliding_window_view(v, n)
        out[n - 1:] = (w <= w[:, -1:]).mean(axis=1)
    out[np.isnan(v)] = np.nan
    return pd.Series(out, index=s.index)''', []),
    "eff_ratio": ("eff_ratio(s, n)  # Kaufman efficiency ratio 0-1 (trendiness)", '''def eff_ratio(s, n):
    return s.diff(int(n)).abs() / s.diff().abs().rolling(int(n), min_periods=int(n)).sum().replace(0.0, np.nan)''', []),
    "up_streak": ("up_streak(s)  # consecutive bars s rose", '''def up_streak(s):
    x = s.diff() > 0
    return x.groupby((~x).cumsum()).cumsum()''', []),
    "down_streak": ("down_streak(s)  # consecutive bars s fell", '''def down_streak(s):
    x = s.diff() < 0
    return x.groupby((~x).cumsum()).cumsum()''', []),
}

HELPER_SIGNATURES = "\n".join(f"  {sig}" for sig, _src, _deps in _HELPER_DEFS.values())


def _helpers_for(body: str) -> tuple[str, list[str]]:
    """(source, names) of exactly the helpers `body` calls (plus their
    dependencies), in definition order, indented for placement inside signals()."""
    needed: set[str] = set()

    def add(name: str) -> None:
        if name in needed:
            return
        needed.add(name)
        for dep in _HELPER_DEFS[name][2]:
            add(dep)

    for name in _HELPER_DEFS:
        if re.search(rf"(?<![\w.]){name}\s*\(", body):
            add(name)
    ordered = [n for n in _HELPER_DEFS if n in needed]
    parts = [_HELPER_DEFS[n][1] for n in ordered]
    return (textwrap.indent("\n".join(parts), " " * 8) if parts else ""), ordered


# --------------------------------------------------------------------------
# Strategy spec: parsing + assembly
# --------------------------------------------------------------------------

FAMILIES = ["trend", "breakout", "meanrev", "momentum", "volatility", "pattern", "seasonal"]

# Models drift from the requested "===S===" delimiter: a bare "===", "=== S ===",
# "===STRATEGY_1===", "=== S1 ===" were all seen in real output (a bare "==="
# used to lose the whole batch). Accept them all.
_BLOCK_SPLIT_RE = re.compile(r"^[ \t]*={3,}[ \t]*(?:(?:S|STRATEGY)[ _]*\d*)?[ \t]*(?:={3,})?[ \t]*$", re.IGNORECASE | re.MULTILINE)
_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
_HEADER_RE = re.compile(r"^[ \t]*([a-z_]+)[ \t]*:[ \t]*(.*?)[ \t]*$", re.IGNORECASE | re.MULTILINE)


@dataclass
class StrategySpec:
    name: str
    family: str
    idea: str
    sl: float | None
    tp: float | None
    params: dict
    body: str
    parent: str | None = None
    mutation: str | None = None   # set when produced by the local mutation engine (src/mutation.py)
    problems: list[str] = field(default_factory=list)  # parse-time problems (spec unusable if non-empty)

    def to_record(self) -> dict:
        return {"name": self.name, "family": self.family, "idea": self.idea, "sl": self.sl, "tp": self.tp,
                "params": self.params, "body": self.body, "parent": self.parent, "mutation": self.mutation}

    @classmethod
    def from_record(cls, r: dict) -> "StrategySpec":
        return cls(name=r["name"], family=r.get("family", "other"), idea=r.get("idea", ""), sl=r.get("sl"),
                   tp=r.get("tp"), params=dict(r.get("params") or {}), body=r.get("body", ""), parent=r.get("parent"),
                   mutation=r.get("mutation"))


def _sanitize_name(raw: str) -> str:
    name = re.sub(r"[^a-z0-9_]+", "_", raw.strip().lower()).strip("_")
    name = re.sub(r"_+", "_", name)[:48]
    return name


def _parse_float_or_none(raw: str | None, lo: float, hi: float, default: float | None) -> float | None:
    if raw is None:
        return default
    text = raw.strip().lower().split("#")[0].strip().split()[0] if raw.strip() else ""
    if text in ("none", "null", "no", "-", ""):
        return None
    try:
        value = float(text)
    except ValueError:
        return default
    return value if lo <= value <= hi else default


def _parse_params(raw: str | None) -> tuple[dict | None, str | None]:
    if raw is None or not raw.strip():
        return {}, None
    text = raw.strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
            break
        except (ValueError, SyntaxError):
            value = None
    else:
        value = None
    if not isinstance(value, dict):
        return None, "params is not a JSON object"
    clean = {}
    for k, v in value.items():
        if not isinstance(k, str) or not k.isidentifier():
            return None, f"bad param key {k!r}"
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            if isinstance(v, bool):
                clean[k] = v
                continue
            return None, f"param {k!r} must be a number/string/bool"
        clean[k] = v
    if len(clean) > 12:
        return None, "too many params"
    return clean, None


def parse_specs(raw_text: str) -> list[StrategySpec]:
    """Parse a delimited multi-strategy response. Unusable blocks come back
    with `.problems` set (so the caller can count/learn from them) rather than
    being silently dropped."""
    chunks = _BLOCK_SPLIT_RE.split(raw_text)[1:]  # text before the first marker is chatter
    specs: list[StrategySpec] = []
    for chunk in chunks:
        fence = _FENCE_RE.search(chunk)
        header_text = chunk[: fence.start()] if fence else chunk
        fields = {m.group(1).lower(): m.group(2) for m in _HEADER_RE.finditer(header_text)}
        problems: list[str] = []

        name = _sanitize_name(fields.get("name", ""))
        if not name:
            problems.append("missing name")
        body = textwrap.dedent(fence.group(1)).strip("\n") if fence else ""
        if not fence:
            problems.append("no closed ```python code block (truncated or unformatted)")
        elif not body.strip():
            problems.append("empty code block")
        params, perr = _parse_params(fields.get("params"))
        if perr:
            problems.append(perr)
        family = fields.get("family", "other").strip().lower().split("|")[0].split()[0] if fields.get("family", "").strip() else "other"
        if family not in FAMILIES:
            family = "other"
        parent = fields.get("parent")
        parent = _sanitize_name(parent) if parent else None
        specs.append(StrategySpec(
            name=name or "unnamed", family=family, idea=(fields.get("idea", "") or "").strip()[:160],
            sl=_parse_float_or_none(fields.get("sl"), 0.005, 0.1, 0.02),
            tp=_parse_float_or_none(fields.get("tp"), 0.01, 0.5, 0.04),
            params=params or {}, body=body, parent=parent or None, problems=problems,
        ))
    return specs


class _PrecedenceFixer(ast.NodeTransformer):
    """Repairs the most common LLM operator-precedence bug in boolean-Series
    code: `a > b | c` parses as `a > (b | c)` (and `a & b > c` as `(a & b) > c`),
    which crashes with float-vs-bool errors or silently computes nonsense.
    The intended meaning is almost always `(a > b) | c` / `a & (b > c)`; a bitwise
    and/or *inside* a comparison is never legitimate in these snippets."""

    _BIT = (ast.BitAnd, ast.BitOr)
    fixes = 0

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if len(node.ops) == 1:
            left, right = node.left, node.comparators[0]
            if isinstance(right, ast.BinOp) and isinstance(right.op, self._BIT):
                self.fixes += 1   # a > (b | c)  ->  (a > b) | c
                return ast.BinOp(left=ast.Compare(left=left, ops=node.ops, comparators=[right.left]),
                                 op=right.op, right=right.right)
            if isinstance(left, ast.BinOp) and isinstance(left.op, self._BIT):
                self.fixes += 1   # (a & b) > c  ->  a & (b > c)
                return ast.BinOp(left=left.left, op=left.op,
                                 right=ast.Compare(left=left.right, ops=node.ops, comparators=node.comparators))
        return node


def fix_precedence(body: str) -> tuple[str, int]:
    """Return (possibly rewritten body, number of fixes). Leaves unparsable
    code untouched (the sandbox will report the SyntaxError)."""
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return body, 0
    fixer = _PrecedenceFixer()
    tree = fixer.visit(tree)
    if not fixer.fixes:
        return body, 0
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), fixer.fixes


_FRIENDLY = [
    (re.compile(r"free variable '(le|lx|se|sx)'"), "snippet never assigned `{0}` (assign le/lx/se/sx as boolean Series)"),
    (re.compile(r"name '(\w+)' is not defined"), "used undefined name `{0}` (only df, p, open_, high, low, close, volume, helpers and your own earlier variables exist)"),
    (re.compile(r"too many values to unpack|not enough values to unpack"), "helper returned a different number of values than unpacked (macd returns 3; others return 1 Series)"),
    (re.compile(r"Cannot perform '(r?and_|r?or_|r?xor)' with|unsupported operand type\(s\) for [&|]"), "boolean operator & | applied to a non-boolean (wrap EVERY comparison in parentheses: (a > b) & (c < d))"),
    (re.compile(r"takes (\d+) positional argument|got an unexpected keyword"), "helper called with the wrong number of arguments (see HELPERS signatures)"),
    (re.compile(r"int\(\) argument must be .* 'Series'"), "a Series was passed where a window length (number) is expected"),
    (re.compile(r"KeyError\('(\w+)'\)"), "read p[\"{0}\"] but it is missing from params"),
    (re.compile(r"LOOKAHEAD"), "LOOKAHEAD: used future data (shift(-k), center=True, or whole-series max/min/mean/rank)"),
    (re.compile(r"forbidden name '(\w+)'"), "used forbidden name `{0}` (open price is open_)"),
]


def friendly_problem(problem: str) -> str:
    """Normalise a raw sandbox/runtime error into the instruction the model
    should have followed -- used for the lessons shown in later prompts."""
    for pattern, template in _FRIENDLY:
        m = pattern.search(problem)
        if m:
            return template.format(*m.groups())
    return re.sub(r"\s+", " ", problem)[:110]


def _domain_for(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, bool) or isinstance(v, str):
            out[k] = [v]
        elif isinstance(v, int):
            out[k] = sorted({max(1, round(v * 0.6)), v, max(2, round(v * 1.5))})
        else:
            out[k] = sorted({round(v * 0.7, 6), v, round(v * 1.4, 6)})
    return out


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_") if part) or "Strategy"


def build_source(spec: StrategySpec, risk_per_trade: float = 0.01) -> str:
    """Assemble the complete standalone strategy file from a spec.

    The snippet becomes the body of a nested `_body()` inside signals(); OHLCV
    series are passed as default args so the snippet may freely reassign them.
    A snippet may either assign le/lx/se/sx or end with its own `return`."""
    body, _ = fix_precedence(spec.body)
    tree = ast.parse(body)  # SyntaxError propagates -> caller reports it as a validation problem
    has_return = any(isinstance(n, ast.Return) for n in tree.body)
    body_lines = body + ("" if has_return else "\nreturn le, lx, se, sx")
    body_src = textwrap.indent(body_lines, " " * 12)
    helpers, used = _helpers_for(body)
    helper_block = helpers + "\n" if helpers else ""
    # Helpers are also bound as default args of _body so a snippet may rebind
    # them (`adx = adx(14)` is natural and would otherwise raise UnboundLocalError).
    helper_args = "".join(f", {n}={n}" for n in used)
    idea = (spec.idea or spec.name).replace("\\", "/").replace('"', "'")
    class_name = "StratGen" + _camel(spec.name)
    tag = re.sub(r"[^a-z0-9]", "", spec.family)[:8] or "gen"

    return f'''from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class {class_name}:
    """{idea}"""

    STRATEGY_NAME = {spec.name!r}
    RATIONALE = {idea!r}
    MODE = {spec.name!r}
    TAG = {tag!r}
    REQUIRED_COLS = ["open", "high", "low", "close", "volume"]
    PRICE_COL = "close"
    STOP_LOSS_PCT = {spec.sl!r}
    TAKE_PROFIT_PCT = {spec.tp!r}
    RISK_PER_TRADE = {risk_per_trade!r}
    POSITION_SIZING = "risk_based_contracts"

    @staticmethod
    def domain() -> Dict[str, List[Any]]:
        return {_domain_for(spec.params)!r}

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {spec.params!r}

    @staticmethod
    def validate(p: Dict[str, Any]) -> bool:
        return True

    @staticmethod
    def signals(df: pd.DataFrame, p: Dict[str, Any]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        open_, high, low, close, volume = df["open"], df["high"], df["low"], df["close"], df["volume"]
{helper_block}        def _body(open_=open_, high=high, low=low, close=close, volume=volume{helper_args}):
            lx = se = sx = pd.Series(False, index=df.index)
{body_src}

        def _b(x):
            return pd.Series(x, index=df.index).fillna(False).astype(bool)

        le, lx, se, sx = _body()
        return (_b(le), _b(lx), _b(se), _b(sx))
'''


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

THEMES = [
    "N-day (Donchian) channel breakout with ATR-based exit",
    "RSI(2-5) pullback inside a long-term SMA/EMA trend",
    "Bollinger-band mean reversion (z-score of close)",
    "volatility squeeze (low ATR percentile) followed by range expansion breakout",
    "gap fade or gap continuation (open vs prior close, ATR-normalised)",
    "NR4/NR7 narrow-range or inside-day breakout",
    "time-series momentum: sign of 20/60/120-bar return, with volatility filter",
    "dual EMA crossover filtered by ADX trend strength",
    "Keltner channel (EMA +/- k*ATR) trend or reversion",
    "consecutive up/down closes exhaustion reversal",
    "stochastic oversold/overbought cross with trend filter",
    "MACD histogram turn / zero-line cross",
    "proximity to 52-week (250-bar) high or low: continuation or fade",
    "close-location-value (CLV) / candle-body momentum",
    "Kaufman efficiency-ratio regime switch (trend-follow if trendy, fade if choppy)",
    "ATR trailing-stop trend following (chandelier / supertrend-like)",
    "day-of-week or turn-of-month seasonality (df.index.dayofweek / df.index.day)",
    "range-expansion bar (true range > k*ATR) follow-through or fade",
    "volatility-regime filter: trade only when ATR percentile is high (or low)",
    "short-term reversal after N-bar extreme return (z-scored roc)",
    "moving-average slope + pullback to EMA",
    "price vs rolling VWAP-like mean (typical price) deviation reversion",
    "momentum rank within own history (pct_rank of roc) as entry trigger",
    "asymmetric long/short: long trend-follow, short only on crash-style breakdowns",
    "mean reversion to a fast EMA after a volatility spike, exit at the mean",
    "swing-high/low breakout with confirmation close and time-stop exit",
]




def _fmt_example() -> str:
    """Format demonstration only. Deliberately an idea nobody should resubmit
    (models copied the previous, realistic example verbatim)."""
    return '''===S===
name: example_do_not_reuse_prior_range_fade
family: pattern
idea: FORMAT DEMO ONLY - fade a close outside the prior day's range
sl: 0.02
tp: 0.03
params: {"k": 0.5, "n": 20}
```python
rng = (high - low).rolling(int(p["n"]), min_periods=int(p["n"])).mean()
le = close < low.shift(1) - p["k"] * rng
lx = close > close.shift(1)
se = close > high.shift(1) + p["k"] * rng
sx = close < close.shift(1)
```'''


def _mandate(min_tpy: float) -> str:
    return f"""THE #1 REQUIREMENT -- TRADE OFTEN. Every strategy must average at least {min_tpy:.0f} round-trip trades per year per market
(hundreds of trades per market over 20+ years). Strategies firing 2-5 times a year are lucky coin flips: they are measured,
penalised and thrown away automatically, and so is everything that does not clear {min_tpy:.0f}/yr. How to do it:
- short lookbacks (2-20 bars), shallow thresholds (RSI < 35, not < 15; z < -1, not < -2.5; breakout of 5-20 bars, not 100),
- at most 2-3 AND-ed conditions (each extra AND divides the trade count),
- signals that are true on roughly 5-15% of bars,
- ALWAYS an exit (lx/sx) that closes trades within 1-15 bars (mean/trend/time exit), otherwise one position blocks everything,
- use both directions where sensible (se/sx): it doubles the opportunities."""


_ENV_AND_RULES = f"""SNIPPET ENVIRONMENT (already defined -- never import or redefine): df, p (your params dict), open_, high, low,
close, volume (pandas Series, daily bars), np, pd, and these vectorised, lookahead-free HELPERS:
{HELPER_SIGNATURES}
Assign boolean Series: le (long entry, REQUIRED), lx (long exit), se (short entry), sx (short exit). Unassigned lx/se/sx default
to False. A signal on bar t is filled at bar t+1's open. The stop `sl` and target `tp` are applied automatically.

MANDATORY CODE RULES (every violation = candidate discarded; these are the exact mistakes seen in earlier batches):
1. PARENTHESES: wrap EVERY comparison inside & | ~ .   RIGHT: (a > b) & (c < d)   WRONG: a > b & c < d   WRONG: x > 5 | y
2. le/lx/se/sx must be BOOLEAN Series (comparisons, crossover/crossunder, & | ~ of those). Never floats, never if/else/for/while
   on a Series, never `lx = 1`. A plain `False` is fine for an unused side.
3. Exact helper signatures: atr(n), adx(n), stoch(n, k), hh(n), ll(n), clv() take NO series argument; sma/ema/stdev/zscore/roc/rsi/
   pct_rank/eff_ratio take (series, n). macd returns THREE values: line, signal, hist = macd(close, f, s, g). Window lengths are
   plain numbers (p["n"]), never Series.
4. Only names that exist: df, p, open_, high, low, close, volume, np, pd, the helpers, and variables you assigned on an EARLIER
   line. You MUST assign `le`. Every p["key"] you read must appear in params, and every key in params must be used.
5. NO LOOKAHEAD (checked by a dynamic test): never shift(-k), never rolling(center=True), never max/min/mean/std/rank/quantile
   over the WHOLE series -- only rolling/ewm/expanding windows and shift(k>=1). Channel breakouts: use hh(n)/ll(n) (they exclude
   today); `close > high.rolling(n).max()` can never be true.
6. No imports, no loops, no randomness, no recursion, no print. Forbidden names: open (the open price is open_), input, vars,
   exit, eval, exec, compile, getattr, setattr, hasattr, globals, locals, help.
7. params = ONE-LINE JSON object of plain numbers (no lists, null, comments, trailing commas). Put every tunable number in params.
8. Compare like with like: never an oscillator (RSI, z-score) against a price level; normalise distances by atr(n) or close.
9. Don't use volume (zero on many markets). Scale-free logic only: the same code runs on equities, bonds, FX, metals, grains."""

_OUTPUT_FORMAT = """OUTPUT FORMAT: your reply starts with the characters ===S=== and contains ONLY blocks of this exact shape (each block opens with
the line ===S===). No thinking, no preface, no explanations, no summary, no markdown outside the blocks:
{example}
family = trend|breakout|meanrev|momentum|volatility|pattern|seasonal; sl 0.005-0.1 and tp 0.01-0.5 (fractions of entry price)."""

_WHY_BLOCK = """HOW YOU ARE SCORED: score = mean over markets of the average walk-forward Sharpe (5 time folds), multiplied down if the
strategy trades fewer than {min_tpy:.0f} times/yr. Acceptance needs >= {accept_pct:.0%} of markets profitable AND passing every fold
(positive Sharpe in each period, enough trades in each period, drawdown limits). One lucky market does not count; breadth and
steady trade flow do."""


def build_explore_prompt(n: int, n_symbols: int, memory_text: str, rng: random.Random, n_themes: int = 4,
                         min_tpy: float = 10.0, accept_pct: float = 0.6) -> str:
    themes = "; ".join(rng.sample(THEMES, min(n_themes, len(THEMES))))
    return f"""You are an automated quantitative strategy generator inside a backtesting harness. You write {n} NEW daily-bar futures trading
strategies as short Python snippets. The harness backtests each on {n_symbols} futures markets (equity indices, rates, FX, metals,
energy, grains, softs, livestock...), 20+ years of daily bars, and learns from the results. Be systematic and literal: the output is
parsed by a program.

{_mandate(min_tpy)}

{_WHY_BLOCK.format(min_tpy=min_tpy, accept_pct=accept_pct)}

THEMES FOR THIS BATCH (cover them, vary the details, add your own twists; every strategy needs a DIFFERENT core mechanism): {themes}

{_ENV_AND_RULES}

WHAT THE HARNESS HAS LEARNED SO FAR (real backtest results -- use them):
{memory_text}

{_OUTPUT_FORMAT.format(example=_fmt_example())}
The example above is a format demo: do not reuse or lightly vary it. Write exactly {n} blocks, each with a unique snake_case name.
"""


def build_evolve_prompt(parents: list[dict], variants_per_parent: int, n_symbols: int, memory_text: str,
                        min_tpy: float = 10.0, accept_pct: float = 0.6) -> str:
    """`parents`: dicts with keys name, family, idea, sl, tp, params, body, results (one-line stats), diagnosis."""
    blocks = []
    for p in parents:
        blocks.append(
            f"### PARENT {p['name']} (family={p['family']}) -- {p['idea']}\n"
            f"sl={p['sl']} tp={p['tp']} params={json.dumps(p['params'])}\n"
            f"```python\n{p['body']}\n```\n"
            f"MEASURED: {p['results']}\n"
            f"DIAGNOSIS: {p['diagnosis']}"
            + (f"\nHISTORY (changes that led here, with their measured effect on score): {p['history']}" if p.get("history") else "")
        )
    parent_text = "\n\n".join(blocks)
    total = variants_per_parent * len(parents)
    example = _fmt_example().replace("family: pattern", "family: pattern\nparent: <the PARENT's name>")
    return f"""You are an automated quantitative strategy generator inside a backtesting harness. Below are existing strategies with their
MEASURED results across {n_symbols} futures markets. Improve them: for each PARENT write {variants_per_parent} variants as short
Python snippets. Be systematic and literal: the output is parsed by a program.

{_mandate(min_tpy)}

{_WHY_BLOCK.format(min_tpy=min_tpy, accept_pct=accept_pct)}

HOW TO IMPROVE: read MEASURED and DIAGNOSIS for each parent, fix the weakness named first. Keep the part that works.
 variant 1: targeted fix of the diagnosed weakness (more trades if it trades < {min_tpy:.0f}/yr; a regime/volatility filter if it
            loses in some asset classes or periods; drop a losing long/short leg; change thresholds/lookbacks);
 variant 2: materially different exit/risk design (time stop, trailing or mean exit, different sl/tp);
 variant 3: a simpler or faster version of the same core edge (shorter lookbacks, fewer conditions => more trades).
Variants must change the logic or its parameters meaningfully, not nudge one number by 5%.

{parent_text}

{_ENV_AND_RULES}

OTHER LEARNINGS:
{memory_text}

{_OUTPUT_FORMAT.format(example=example)}
Write exactly {total} blocks ({variants_per_parent} per parent), each with a NEW unique snake_case name and a `parent:` line.
"""
