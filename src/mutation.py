"""Code-level mutation operators for the local (zero-token) strategy search.

A mutation takes a StrategySpec and returns a modified copy: numeric parameters
rescaled, stop/target changed, a long or short leg dropped, the signal inverted,
or a filter/exit appended to the snippet. Operators work on the snippet text
(appending lines that rewrite le/lx/se/sx), so they apply to ANY strategy,
whether it came from the LLM or from a seed template, and the result goes
through the normal sandbox (including the lookahead test).

Which operator to try is not random: `choose_op` weights them by
- their measured success rate so far (memory.mutation_stats: share of
  candidates that beat their parent), and
- the diagnosis of the parent's results (a losing short leg boosts
  drop_shorts, no edge before costs boosts invert, costs eating a real edge
  boosts the trade-reducing filters, ...).
That is the "learning" in the loop: results -> operator priors -> next candidates.
"""
from __future__ import annotations

import ast
import random
import re

from src.strategy_codegen import StrategySpec

OP_LABELS = {
    "perturb": "re-tune parameters",
    "sltp": "change stop/target",
    "drop_shorts": "drop the short leg (long-only)",
    "drop_longs": "drop the long leg (short-only)",
    "invert": "invert the signal (swap long/short)",
    "trend_filter": "trade only with the SMA trend",
    "contra_trend": "trade only against the SMA trend",
    "vol_calm": "only in calm (low ATR percentile) regimes",
    "vol_wild": "only in volatile (high ATR percentile) regimes",
    "adx_trend": "only when ADX shows a strong trend",
    "adx_chop": "only when ADX shows chop (weak trend)",
    "time_stop": "add a time-based exit",
    "flip_exit": "exit when the opposite signal fires",
}
ALL_OPS = list(OP_LABELS)
# Base sampling weights before learning/diagnosis adjust them.
BASE_WEIGHT = {"perturb": 3.0, "sltp": 1.5}


def _has_top_level_return(body: str) -> bool:
    try:
        return any(isinstance(n, ast.Return) for n in ast.parse(body).body)
    except SyntaxError:
        return True


def _assigns(body: str, name: str) -> bool:
    return re.search(rf"^\s*{name}\s*=", body, re.MULTILINE) is not None


def _clone(spec: StrategySpec, op: str, body: str | None = None, params: dict | None = None,
           sl: float | None = None, tp: float | None = None) -> StrategySpec:
    return StrategySpec(
        name=spec.name, family=spec.family, idea=spec.idea, sl=spec.sl if sl is None else sl,
        tp=spec.tp if tp is None else tp, params=dict(spec.params if params is None else params),
        body=spec.body if body is None else body, parent=spec.name, mutation=op,
    )


def _perturb_params(spec: StrategySpec, rng: random.Random) -> dict | None:
    params = dict(spec.params)
    keys = [k for k, v in params.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0]
    if not keys:
        return None
    changed = False
    for k in keys:
        if rng.random() < 0.55:
            v = params[k]
            nv = v * rng.uniform(0.55, 1.7)
            if isinstance(v, int):
                nv = max(1, int(round(nv)))
                if nv == v:
                    nv = v + 1 if v < 3 or rng.random() < 0.5 else v - 1
            else:
                nv = round(nv, 4)
                if 0 < abs(v) <= 1:
                    nv = max(0.01, min(1.0, abs(nv))) * (1 if v > 0 else -1)
            if nv != v:
                params[k] = nv
                changed = True
    if not changed:
        k = rng.choice(keys)
        v = params[k]
        params[k] = v + 1 if isinstance(v, int) else round(v * 1.25, 4)
    return params


def _append(spec: StrategySpec, op: str, lines: list[str], extra_params: dict | None = None) -> StrategySpec | None:
    """Append `lines` to the snippet (first line tagged with a marker so the same
    operator is never stacked twice); None if the snippet has its own return."""
    if f"# mut:{op}" in spec.body or _has_top_level_return(spec.body):
        return None
    params = dict(spec.params)
    for k, v in (extra_params or {}).items():
        if k in params:
            return None
        params[k] = v
    lines = list(lines)
    lines[0] = f"{lines[0]}  # mut:{op}"
    return _clone(spec, op, body=spec.body.rstrip("\n") + "\n" + "\n".join(lines), params=params)


def mutate(spec: StrategySpec, op: str, rng: random.Random) -> StrategySpec | None:
    """Apply operator `op`; None if it does not apply to this strategy."""
    body = spec.body
    if op == "perturb":
        params = _perturb_params(spec, rng)
        return _clone(spec, op, params=params) if params is not None else None
    if op == "sltp":
        sl = rng.choice([0.01, 0.015, 0.02, 0.03, 0.05, 0.08])
        tp = rng.choice([0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3])
        if (sl, tp) == (spec.sl, spec.tp):
            return None
        return _clone(spec, op, sl=sl, tp=tp)
    if op == "drop_shorts":
        return _append(spec, op, ["se = False", "sx = False"]) if _assigns(body, "se") else None
    if op == "drop_longs":
        return _append(spec, op, ["le = False", "lx = False"]) if _assigns(body, "se") else None
    if op == "invert":
        return _append(spec, op, ["le, se = se, le", "lx, sx = sx, lx"])
    if op in ("trend_filter", "contra_trend"):
        up, down = (">", "<") if op == "trend_filter" else ("<", ">")
        return _append(spec, op, ['_mtf = sma(close, p["mut_tf"])', f"le = le & (close {up} _mtf)", f"se = se & (close {down} _mtf)"],
                       {"mut_tf": rng.choice([50, 100, 150, 200])})
    if op in ("vol_calm", "vol_wild"):
        cmp = "<" if op == "vol_calm" else ">"
        return _append(spec, op, ["_mvr = pct_rank(atr(14) / close, 250)", f'le = le & (_mvr {cmp} p["mut_vq"])',
                                  f'se = se & (_mvr {cmp} p["mut_vq"])'], {"mut_vq": rng.choice([0.3, 0.5, 0.7])})
    if op in ("adx_trend", "adx_chop"):
        cmp = ">" if op == "adx_trend" else "<"
        return _append(spec, op, ["_mad = adx(14)", f'le = le & (_mad {cmp} p["mut_adx"])', f'se = se & (_mad {cmp} p["mut_adx"])'],
                       {"mut_adx": rng.choice([15, 20, 25, 30])})
    if op == "time_stop":
        return _append(spec, op, ['lx = lx | pd.Series(le, index=df.index).shift(int(p["mut_ts"]), fill_value=False)',
                                  'sx = sx | pd.Series(se, index=df.index).shift(int(p["mut_ts"]), fill_value=False)'],
                       {"mut_ts": rng.choice([2, 3, 5, 8, 12])})
    if op == "flip_exit":
        return _append(spec, op, ["lx = lx | pd.Series(se, index=df.index)", "sx = sx | pd.Series(le, index=df.index)"]) \
            if _assigns(body, "se") else None
    raise ValueError(f"unknown mutation operator {op!r}")


def diagnosis_boosts(summary: dict, min_tpy: float = 10.0) -> dict[str, float]:
    """Operator multipliers implied by the parent's measured results."""
    b: dict[str, float] = {}
    if not summary:
        return b
    long_pnl, short_pnl = summary.get("long_pnl_pct", 0.0), summary.get("short_pnl_pct", 0.0)
    gross, cost = summary.get("gross_pnl_pct", 0.0), summary.get("cost_pct", 0.0)
    if short_pnl < -0.3 and long_pnl > 0:
        b["drop_shorts"] = 4.0
    if long_pnl < -0.3 and short_pnl > 0:
        b["drop_longs"] = 4.0
    if gross < 0 and summary.get("pct_positive", 1.0) < 0.45:
        b["invert"] = 4.0
    if gross > 0.2 and gross - cost < 0.2:          # real edge, eaten by costs: trade less
        for op in ("trend_filter", "adx_trend", "vol_calm", "vol_wild"):
            b[op] = 2.5
    if summary.get("pct_positive", 0.0) >= 0.55 and summary.get("pass_rate", 1.0) < 0.15:   # profitable but unstable over time
        for op in ("vol_calm", "vol_wild", "adx_trend", "adx_chop", "time_stop"):
            b[op] = max(b.get(op, 1.0), 2.0)
    if summary.get("trades_per_year", 99.0) < min_tpy:
        b["perturb"] = 2.0
        b["time_stop"] = 2.5
    if summary.get("avg_hold_bars", 0.0) > 25:
        b["time_stop"] = max(b.get("time_stop", 1.0), 2.5)
    return b


def choose_op(rng: random.Random, stats: dict, boosts: dict[str, float], exclude: set[str] | None = None) -> str:
    """Sample an operator: weight = base * learned success rate * diagnosis boost.
    The learned rate is a smoothed win fraction ((wins+1)/(tried+2)) so untried
    operators keep a fair chance and a floor keeps every operator alive."""
    weights = []
    ops = [o for o in ALL_OPS if not exclude or o not in exclude]
    for op in ops:
        st = stats.get(op, {})
        rate = (st.get("wins", 0) + 1.0) / (st.get("tried", 0) + 2.0)
        weights.append(max(0.05, BASE_WEIGHT.get(op, 1.0) * (0.25 + 1.5 * rate) * boosts.get(op, 1.0)))
    return rng.choices(ops, weights=weights, k=1)[0]
