"""Turns universe results into compact, LLM-readable evidence -- computed in
plain code, so interpreting results costs zero tokens and zero API requests
(the old pipeline spent one LLM request per iteration on a vague prose
"analysis" of numbers the program already had).

Two outputs per strategy:
- `results_line`: one line of headline numbers;
- `diagnose`: rule-based reading of those numbers into concrete things to fix
  ("short leg loses money", "works on equity indices but not grains", "too few
  trades", "edge exists but unstable across time", ...).
"""
from __future__ import annotations

_FAIL_TEXT = {
    "low_sharpe": "average Sharpe too low",
    "negative_fold": "a fold goes clearly negative",
    "inconsistent": "worst fold gives back most of the edge",
    "drawdown": "drawdown too deep",
    "low_profit_factor": "profit factor < 1",
    "low_trades": "too few trades in some fold",
    "error": "runtime error",
}


def _top_groups(group_mean: dict, n: int = 2, reverse: bool = True) -> str:
    items = sorted(group_mean.items(), key=lambda kv: kv[1], reverse=reverse)[:n]
    return ", ".join(f"{g} {v:+.2f}" for g, v in items)


def results_line(s: dict) -> str:
    if not s or not s.get("n_eval"):
        return "no evaluable symbols"
    best = ", ".join(f"{sym} {sh:+.2f}" for sym, sh in s.get("best_symbols", []))
    return (f"score {s['score']:+.2f} | {s['pct_positive']:.0%} of {s['n_eval']} symbols profitable | "
            f"passes all criteria on {s['n_pass']} | {s['trades_per_year']:.1f} trades/yr/symbol (~{s.get('trades_per_symbol', 0)} per symbol), avg hold {s['avg_hold_bars']:.0f} bars | "
            f"avg PnL%% long {s['long_pnl_pct']:+.1f} / short {s['short_pnl_pct']:+.1f}, gross {s.get('gross_pnl_pct', 0):+.1f} costs {s.get('cost_pct', 0):.1f} | "
            f"best groups: {_top_groups(s['group_mean_sharpe'])}; worst: {_top_groups(s['group_mean_sharpe'], reverse=False)} | "
            f"best symbols: {best}").replace("%%", "%")


def diagnose(s: dict, min_tpy: float = 10.0) -> str:
    """Rule-based weaknesses, most important first. Always returns something.
    `min_tpy` is the required trades per year per symbol."""
    if not s or not s.get("n_eval"):
        return "Could not be evaluated; simplify the logic."
    hints: list[str] = []
    n = s["n_eval"]
    fails = s.get("fail_counts", {})
    gm = s.get("group_mean_sharpe", {})

    if s["trades_per_year"] < min_tpy or s.get("no_trade_symbols", 0) > 0.3 * n:
        hints.append(f"TOO FEW TRADES: {s['trades_per_year']:.1f}/yr per symbol (~{s.get('trades_per_symbol', 0)} per symbol in total; "
                     f"need >= {min_tpy:.0f}/yr = hundreds per symbol; {s.get('no_trade_symbols', 0)} symbols never traded) "
                     "-> top priority: shorter lookbacks, shallower thresholds, fewer AND conditions, add the short side, "
                     "exit within 1-15 bars so the next trade can start")
    elif s["trades_per_year"] > 60:
        hints.append(f"over-trades ({s['trades_per_year']:.0f}/yr) -> costs eat the edge; add a filter or hold longer")

    gross, cost = s.get("gross_pnl_pct", 0.0), s.get("cost_pct", 0.0)
    if s["score"] <= 0.05 and s["trades_per_year"] >= min_tpy:
        if gross > 0.2 and gross - cost < 0.2:
            hints.append(f"edge is eaten by costs (gross PnL {gross:+.1f}% vs costs {cost:.1f}% per period) -> trade less often, "
                         "hold longer, wider targets, or stricter entries")
        elif gross < 0:
            hints.append(f"no edge even before costs (gross {gross:+.1f}%) -> the signal itself is wrong: change the mechanism, "
                         "or trade the opposite side")

    long_pnl, short_pnl = s["long_pnl_pct"], s["short_pnl_pct"]
    if short_pnl < -0.3 and long_pnl > 0:
        hints.append(f"short leg loses money (short {short_pnl:+.1f}% vs long {long_pnl:+.1f}%) -> long-only or a much stricter short filter")
    elif long_pnl < -0.3 and short_pnl > 0:
        hints.append(f"long leg loses money (long {long_pnl:+.1f}% vs short {short_pnl:+.1f}%) -> short-only or a stricter long filter")

    if gm:
        best_g, best_v = max(gm.items(), key=lambda kv: kv[1])
        worst_g, worst_v = min(gm.items(), key=lambda kv: kv[1])
        if best_v - worst_v > 0.4:
            hints.append(f"edge is asset-class specific: strong in {best_g} ({best_v:+.2f}), weak in {worst_g} ({worst_v:+.2f}) "
                         "-> add a volatility/trend-regime filter that generalises, or lean into what works there")

    if s["pct_positive"] >= 0.6 and s["pass_rate"] < 0.2:
        dominant = max(fails.items(), key=lambda kv: kv[1])[0] if fails else None
        if dominant == "negative_fold" or dominant == "inconsistent":
            hints.append("broadly profitable but unstable across time (some period always loses) -> regime filter, "
                         "volatility-scaled thresholds, or faster exits to cut the bad period")
        elif dominant == "low_trades":
            hints.append("profitable but fails the minimum-trades-per-fold rule -> trade more often (looser entries, shorter holds)")
        elif dominant == "drawdown":
            hints.append("profitable but drawdowns too deep -> tighter sl, trailing/mean exit, or filter out chaotic regimes")
        elif dominant == "low_profit_factor":
            hints.append("many small wins, few large losses -> widen tp / tighten sl / add a loss-cutting exit")
    if s["pct_positive"] < 0.45:
        hints.append("loses on most symbols -> the core signal likely has no edge; try the opposite side or a different mechanism")
    if not hints:
        top = max(fails.items(), key=lambda kv: kv[1])[0] if fails else None
        hints.append(f"no glaring flaw; main blocker: {_FAIL_TEXT.get(top, top)}" if top else "solid; try refinements for higher Sharpe")
    return "; ".join(hints[:3])
