"""Chart and Markdown report generation (matplotlib + pandas only).

`write_universe_report` is the live one (IS / OSS1 reporting for the universe pipeline).
The older single-symbol helpers below (`generate_all_charts`, `write_markdown_report`) belong
to the legacy pnl_viewer notebook flow and still use the old search-fold / holdout wording."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("report")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _chain_equities(equities: list[pd.Series], initial_capital: float) -> pd.Series:
    """Chain several equity curves (one per fold, each independently starting
    from initial_capital) into one continuous series for charting -- each
    subsequent fold's curve is offset so it visually continues where the
    previous one left off. Sharpe/metrics are still computed per-fold
    independently elsewhere; this is purely a plotting convenience."""
    chained = []
    offset = 0.0
    for eq in equities:
        eq = eq.dropna()
        if eq.empty:
            continue
        chained.append(eq + offset)
        offset += eq.iloc[-1] - initial_capital
    return pd.concat(chained) if chained else pd.Series(dtype=float)


def plot_equity_curve(equity: pd.Series, benchmark: pd.Series | None, out_path: Path,
                       fold_boundaries: list[pd.Timestamp] | None = None,
                       holdout_start: pd.Timestamp | None = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    if benchmark is not None and not benchmark.empty:
        scaled = benchmark / benchmark.iloc[0] * equity.iloc[0]
        ax.plot(scaled.index, scaled.values, color="grey", alpha=0.35, linewidth=1.2, label="Buy & Hold", zorder=1)
    ax.plot(equity.index, equity.values, color="tab:blue", linewidth=1.5, label="Strategy (search folds)", zorder=2)
    for b in (fold_boundaries or []):
        ax.axvline(b, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    if holdout_start is not None:
        ax.axvline(holdout_start, color="black", linestyle="--", linewidth=1.2)
        ax.text(holdout_start, ax.get_ylim()[1] if ax.get_ylim()[1] else 0, "  final holdout \u2192",
                va="top", fontsize=8)
    ax.set_title("Equity Curve (search folds chained; dotted lines = fold boundaries)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left")
    _save(fig, out_path)


def plot_drawdown(equity: pd.Series, out_path: Path) -> None:
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.fill_between(dd.index, dd.values * 100, 0, color="firebrick", alpha=0.6)
    ax.set_title("Drawdown (%) -- search folds chained")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    _save(fig, out_path)


def plot_monthly_returns(equity: pd.Series, out_path: Path) -> None:
    ret = equity.pct_change().dropna()
    if ret.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No returns to display", ha="center")
        _save(fig, out_path)
        return
    monthly = (1 + ret).resample("ME").prod() - 1
    table = monthly.to_frame("ret")
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = table.pivot(index="year", columns="month", values="ret")
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_title("Monthly Returns (%) -- search folds chained")
    fig.colorbar(im, ax=ax, label="% return")
    _save(fig, out_path)


def plot_trade_pnl_histogram(trades: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if trades is not None and not trades.empty:
        ax.hist(trades["pnl"], bins=30, color="steelblue", edgecolor="white")
    else:
        ax.text(0.5, 0.5, "No trades", ha="center")
    ax.set_title("Trade PnL Distribution -- search folds")
    ax.set_xlabel("PnL")
    ax.set_ylabel("Count")
    _save(fig, out_path)


def plot_fold_sharpes(fold_sharpes: list[float], holdout_sharpe: float | None,
                       min_avg_fold_sharpe: float, min_fold_sharpe_floor: float, out_path: Path) -> None:
    """Bar chart of each search fold's Sharpe, with the (separately, once)
    checked holdout fold's Sharpe shown in a distinct color -- the single
    most direct visual for "are these folds actually consistent"."""
    labels = [f"Fold {i+1}" for i in range(len(fold_sharpes))]
    colors = ["tab:blue"] * len(fold_sharpes)
    values = list(fold_sharpes)
    if holdout_sharpe is not None:
        labels.append("Holdout")
        colors.append("tab:orange")
        values.append(holdout_sharpe)

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(values)), 4.5))
    ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(min_avg_fold_sharpe, color="green", linestyle="--", linewidth=1, label=f"min_avg_fold_sharpe ({min_avg_fold_sharpe})")
    ax.axhline(min_fold_sharpe_floor, color="orange", linestyle=":", linewidth=1, label=f"min_fold_sharpe_floor ({min_fold_sharpe_floor})")
    ax.set_title("Sharpe by Fold (blue = search folds used for acceptance, orange = untouched holdout)")
    ax.set_ylabel("Sharpe")
    ax.legend(loc="best", fontsize=8)
    _save(fig, out_path)


def generate_all_charts(fold_eval, holdout, benchmark_close: pd.Series | None,
                         fold_boundaries: list, holdout_start, initial_capital: float,
                         min_avg_fold_sharpe: float, min_fold_sharpe_floor: float, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    paths = {
        "equity_curve": out_dir / "equity_curve.png",
        "drawdown": out_dir / "drawdown.png",
        "monthly_returns": out_dir / "monthly_returns.png",
        "trade_pnl_histogram": out_dir / "trade_pnl_histogram.png",
        "fold_sharpes": out_dir / "fold_sharpes.png",
    }

    fold_equities = [r.equity_curve["equity"] for r in fold_eval.fold_results]
    chained = _chain_equities(fold_equities, initial_capital)

    # Trim the benchmark to the same date range actually plotted (search
    # folds only) -- otherwise the benchmark visually continues into the
    # holdout period while the strategy line stops, which reads as "the
    # strategy fell off a cliff" rather than "the holdout isn't shown here".
    benchmark_trimmed = None
    if benchmark_close is not None and not benchmark_close.empty and not chained.empty:
        benchmark_trimmed = benchmark_close.loc[chained.index.min():chained.index.max()]

    plot_equity_curve(chained, benchmark_trimmed, paths["equity_curve"], fold_boundaries, None)
    plot_drawdown(chained, paths["drawdown"])
    plot_monthly_returns(chained, paths["monthly_returns"])

    all_trades = pd.concat([r.trades for r in fold_eval.fold_results], ignore_index=True) \
        if any(r.trades is not None and not r.trades.empty for r in fold_eval.fold_results) else pd.DataFrame()
    plot_trade_pnl_histogram(all_trades, paths["trade_pnl_histogram"])

    fold_sharpes = [m["sharpe"] for m in fold_eval.fold_metrics]
    holdout_sharpe = holdout.metrics["sharpe"] if holdout is not None else None
    plot_fold_sharpes(fold_sharpes, holdout_sharpe, min_avg_fold_sharpe, min_fold_sharpe_floor, paths["fold_sharpes"])

    return paths


def write_markdown_report(strategy, symbol: str, source_filename: str, timeframe: str,
                           fold_eval, holdout, model_used: str, cfg, out_path: Path) -> None:
    """`strategy` is a src.strategy_sandbox.CompiledStrategy: strategy logic
    is a full Python file (see strategy.source), not a structured JSON DSL,
    so the report embeds the source itself rather than a rule breakdown."""
    def fmt_metrics(m: dict) -> str:
        rows = "\n".join(f"| {k} | {v:.4f} |" if isinstance(v, (int, float)) else f"| {k} | {v} |" for k, v in m.items())
        return f"| Metric | Value |\n|---|---|\n{rows}"

    lines = [
        f"# Strategy Report: {strategy.strategy_name}",
        "",
        f"- **Symbol**: {symbol}",
        f"- **Source filename**: {source_filename}",
        f"- **Timeframe**: {timeframe} (daily)",
        f"- **LLM model used**: {model_used}",
        f"- **Required columns**: {strategy.required_cols}",
        "",
        "## Rationale",
        strategy.rationale or "(none provided)",
        "",
        "## Parameters used",
        f"`{strategy.params}`",
        "",
        "## Parameter search domain (for future optimization; not searched by this run)",
        f"`{strategy.domain}`",
        "",
        f"- Stop loss: {strategy.stop_loss_pct}",
        f"- Take profit: {strategy.take_profit_pct}",
        f"- Position sizing: {strategy.position_sizing}, risk_per_trade={strategy.risk_per_trade}",
        "",
        "## Strategy source (see also strat_gen_*.py alongside this report)",
        "```python",
        strategy.source,
        "```",
        "",
        "## Per-Fold Metrics (search folds only -- what acceptance was decided on)",
    ]
    for i, m in enumerate(fold_eval.fold_metrics, start=1):
        lines.append(f"### Fold {i}")
        lines.append(fmt_metrics(m))
        lines.append("")

    lines += [
        f"## Cross-Fold Summary",
        f"- Average fold Sharpe: {fold_eval.avg_fold_sharpe:.4f} (min required: {cfg.validation.min_avg_fold_sharpe})",
        f"- Worst fold Sharpe: {fold_eval.min_fold_sharpe:.4f} (floor required: {cfg.validation.min_fold_sharpe_floor})",
        f"- Cross-fold Sharpe consistency: {fold_eval.sharpe_consistency:.4f} (min required: {cfg.validation.min_sharpe_consistency})",
        "",
        f"## Acceptance Result: {'ACCEPTED' if fold_eval.accepted else 'REJECTED'}",
    ]
    if fold_eval.reasons:
        lines.append("### Rejection reasons")
        for r in fold_eval.reasons:
            lines.append(f"- {r}")

    if holdout is not None:
        lines += [
            "",
            "## Final Holdout Result (genuinely untouched during search -- see ASSUMPTIONS.md #27)",
            "This fold was NEVER backtested, shown to the LLM, or used in any acceptance decision",
            "until this single check, run once after the strategy was already accepted.",
            fmt_metrics(holdout.metrics),
            "",
            f"Holdout Sharpe vs. average search-fold Sharpe consistency: {holdout.sharpe_consistency_vs_search:.4f}",
        ]

    lines += [
        "",
        "## Warnings / Overfitting Notes",
        "- Backtest uses next-bar-open execution and simple cost assumptions; results are indicative only.",
        "- Small per-fold trade counts reduce statistical confidence in that fold's metrics.",
        "- Acceptance is based on the search folds only; the holdout result (if present) is the more",
        "  trustworthy generalization check precisely because it was never used to guide the search.",
        "",
        "## Futures-Specific Notes",
        "- Data is treated as a continuous futures contract; any roll/adjustment is assumed already applied upstream in the input .dat file.",
        "- Position sizing and PnL use instrument `point_value`/`tick_size` from config when available.",
        "",
        "## Disclaimer",
        "This report is for local research purposes only and is not financial advice. No live trades are placed by this project.",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


OSS2_PLACEHOLDER = "Not evaluated. Run `python main.py final-test --file <strategy> --confirm` once, at the very end."


def _per_symbol_chart(df: pd.DataFrame, title: str, out_path: Path) -> None:
    """IS bars + OSS1 dots (+ OSS2 diamonds when the test column is present)."""
    try:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.17 * len(df))))
        y = np.arange(len(df))
        colors = ["#2a9d8f" if p else "#9aa5b1" for p in df["passed"]]
        ax.barh(y, df["avg_fold_sharpe"], color=colors, label="IS avg fold Sharpe (green = passes all criteria)")
        ax.scatter(df["oss1_sharpe"], y, color="#e76f51", s=14, zorder=3, label="OSS1 (validation) Sharpe")
        if "oss2_sharpe" in df.columns:
            ax.scatter(df["oss2_sharpe"], y, color="#264653", marker="D", s=14, zorder=4, label="OSS2 (test) Sharpe")
        ax.set_yticks(y)
        ax.set_yticklabels(df["symbol"], fontsize=6)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(f"{title}: per-symbol Sharpe")
        ax.legend(fontsize=7, loc="lower right")
        _save(fig, out_path)
    except Exception as exc:  # a chart failure must never lose the strategy
        logger.warning("Could not draw per-symbol chart for %s: %s", title, exc)


def write_universe_report(strat_dir: Path, spec, compiled, result, oss1: dict, universe, accepted: bool = True) -> None:
    """Per-symbol table (CSV), an IS-vs-OSS1 bar chart and a short Markdown report.
    OSS2 (test) is not evaluated here; `append_oss2` adds it at the very end of a pipeline
    run (or `main.py final-test` does it by hand)."""
    from src import insights  # local import: report.py is imported by older code paths without src.insights

    strat_dir = Path(strat_dir)
    hold = {r["symbol"]: r for r in oss1.get("rows", [])}
    rows = []
    for r in result.per_symbol:
        h = hold.get(r.symbol, {})
        rows.append({
            "symbol": r.symbol, "group": r.group, "avg_fold_sharpe": round(r.avg_sharpe, 4),
            "min_fold_sharpe": round(r.min_sharpe, 4), "consistency": round(r.consistency, 3),
            "fold_sharpes": r.fold_sharpes, "fold_trades": r.fold_trades, "trades_per_year": round(r.trades_per_year, 2),
            "max_drawdown": round(r.max_dd, 4), "long_pnl_pct": round(r.long_pnl_pct, 2),
            "short_pnl_pct": round(r.short_pnl_pct, 2), "passed": r.passed, "fail_codes": ",".join(r.fail_codes),
            "oss1_sharpe": round(h.get("sharpe", float("nan")), 4), "oss1_trades": h.get("trades"),
        })
    df = pd.DataFrame(rows).sort_values("avg_fold_sharpe", ascending=False)
    df.to_csv(strat_dir / "per_symbol.csv", index=False)

    _per_symbol_chart(df, spec.name, strat_dir / "per_symbol.png")

    s = result.summary
    lines = [
        f"# {spec.name}",
        "",
        "**ACCEPTED**: cleared the IS universe screen and the OSS1 validation gate." if accepted else
        "**Top-ranked but NOT accepted** (did not clear every `universe.accept` gate, IS or OSS1); saved for inspection.",
        "",
        f"*{spec.idea}* (family: {spec.family}{', parent: ' + spec.parent if spec.parent else ''})",
        "",
        "## Universe result (IS walk-forward folds only)",
        insights.results_line(s),
        "",
        f"- Symbols evaluated: {s.get('n_eval')} | profitable: {s.get('pct_positive', 0):.0%} | pass all criteria: {s.get('n_pass')}",
        f"- Diagnosis: {insights.diagnose(s)}",
        "",
        "## OSS1 (validation segment; the search never optimises on it)",
        f"- Median Sharpe: {oss1.get('median_sharpe'):+.2f} | mean (clipped to +-1.5): {oss1.get('mean_sharpe'):+.2f} "
        f"| symbols profitable: {oss1.get('pct_positive', 0):.0%} | gate: "
        + ("passed" if oss1.get("passed") else f"FAILED ({oss1.get('fail')})" if "passed" in oss1 else "n/a"),
        "",
        "## OSS2 (test segment)",
        OSS2_PLACEHOLDER,
        "",
        "## Parameters",
        f"`{spec.params}` | stop {compiled.stop_loss_pct} | target {compiled.take_profit_pct}",
        "",
        "See `per_symbol.csv` / `per_symbol.png` for the breakdown and `strat_gen_*.py` for the standalone code.",
        "",
        "_Research only; not financial advice._",
    ]
    (strat_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def append_oss2(strat_dir: Path, name: str, oss2: dict, verdict: str) -> None:
    """Add the one-shot OSS2 (test) result to an already saved strategy folder: metadata.json,
    a per-symbol column in per_symbol.csv, the chart and the report.md section."""
    import json

    strat_dir = Path(strat_dir)
    meta_path = strat_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["oss2"] = {k: v for k, v in oss2.items() if k != "rows"} | {"verdict": verdict}
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    csv_path = strat_dir / "per_symbol.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        by_sym = {r["symbol"]: r for r in oss2.get("rows", [])}
        df["oss2_sharpe"] = [round(by_sym.get(sym, {}).get("sharpe", float("nan")), 4) for sym in df["symbol"]]
        df["oss2_trades"] = [by_sym.get(sym, {}).get("trades") for sym in df["symbol"]]
        df.to_csv(csv_path, index=False)
        _per_symbol_chart(df, name, strat_dir / "per_symbol.png")

    report_path = strat_dir / "report.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        section = (f"Median Sharpe: {oss2['median_sharpe']:+.2f} | mean (clipped to +-1.5): {oss2['mean_sharpe']:+.2f} "
                   f"| symbols profitable: {oss2['pct_positive']:.0%}\n\n**{verdict}**\n\n"
                   "Evaluated exactly once, after the last generation. Do not tune this strategy further based on this number: "
                   "it would turn OSS2 into a second validation set.")
        report_path.write_text(text.replace(OSS2_PLACEHOLDER, section), encoding="utf-8")
