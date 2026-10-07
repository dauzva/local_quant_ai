"""Multi-fold walk-forward evaluation for daily futures strategies.

Replaces the earlier single in-sample/out-of-sample split (see
ASSUMPTIONS.md #19-21, #26 for that history) with K time-ordered,
non-overlapping folds spanning the full available history:

    [ search fold 1 | search fold 2 | ... | search fold K-1 | HOLDOUT fold K ]

- Search folds (all but the last) are what the research loop actually
  optimizes against: acceptance and the near-miss leaderboard are driven by
  cross-fold Sharpe consistency (are these folds' Sharpe ratios close to
  each other?) computed ONLY over the search folds.
- The final fold (most recent data) is a genuinely untouched holdout: it is
  never backtested during the search loop, never appears in any prompt or
  near-miss record, and is only ever evaluated once, after the loop
  finishes, against whatever strategy the loop already decided to accept
  (see run_final_holdout). This is what makes it a real held-out check
  rather than another number the LLM can indirectly optimize against by
  repeatedly seeing its value across iterations (see ASSUMPTIONS.md #27).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtester import BacktestResult, run_backtest
from src.metrics import compute_metrics
from src.utils import get_logger

logger = get_logger("evaluation")


def split_folds(df: pd.DataFrame, cfg) -> list[pd.DataFrame]:
    """Split df into cfg.validation.num_folds contiguous, time-ordered,
    non-overlapping folds spanning the entire available date range -- never
    shuffled, since this is time-series data. Folds are as equal-length as
    possible (numpy distributes any remainder across the earliest folds).
    The LAST fold (most recent data) is always the final holdout by
    convention -- callers should slice `folds[:-1]` for search folds and
    `folds[-1]` for the holdout.
    """
    num_folds = int(cfg.validation.num_folds)
    if num_folds < 2:
        raise ValueError("validation.num_folds must be >= 2 (at least one search fold + one holdout fold)")
    n = len(df)
    if n < num_folds:
        raise ValueError(f"Not enough bars ({n}) to split into {num_folds} folds")
    groups = np.array_split(np.arange(n), num_folds)
    return [df.iloc[g] for g in groups]


def cross_fold_sharpe_consistency(fold_sharpes: list[float]) -> float:
    """What fraction of the average fold Sharpe survives in the WORST fold,
    as a bounded [0, 1] score:

        consistency = min(fold_sharpes) / mean(fold_sharpes)

    1.0 = every fold matches the average exactly (or there's only one fold).
    Falls toward 0 as the worst fold gives back more of the average, and
    clips to 0 if the average itself is <= 0 (already a failure on other
    grounds) or the worst fold has flipped sign relative to a positive
    average. This directly answers "how much of my edge is left in my
    worst period?" without reference to how good the BEST fold happens to
    be, so a strategy that is strongly profitable in every fold isn't
    penalized just because one fold is exceptionally good (the previous
    range-based formula, `1 - (max-min)/max(|max|,|min|)`, was scale-biased
    this way: the higher a strategy's ceiling, the tighter its folds had to
    cluster to pass, even when every fold cleared the absolute bars by a
    wide margin -- see ASSUMPTIONS.md for the concrete case that surfaced
    this: a strategy dominating another's Sharpe in every single fold still
    scored a lower "consistency" and lost the selection).
    """
    if not fold_sharpes:
        return 1.0
    avg = sum(fold_sharpes) / len(fold_sharpes)
    if avg <= 0:
        return 0.0
    worst = min(fold_sharpes)
    return max(0.0, min(1.0, worst / avg))


@dataclass
class FoldEvaluationResult:
    fold_metrics: list[dict]          # per search-fold compute_metrics() dicts, in time order
    fold_results: list[BacktestResult]  # per search-fold BacktestResult, in time order
    avg_fold_sharpe: float
    min_fold_sharpe: float
    sharpe_consistency: float          # cross-fold consistency across search folds only
    accepted: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class HoldoutResult:
    metrics: dict
    result: BacktestResult
    sharpe_consistency_vs_search: float  # how close holdout Sharpe is to avg_fold_sharpe


def evaluate_strategy_folds(df_full_signals: pd.DataFrame, strategy, cfg, symbol: str) -> FoldEvaluationResult:
    """Backtest a strategy across every SEARCH fold (all folds except the
    reserved final holdout), and score it on cross-fold Sharpe consistency.

    `df_full_signals` must be the FULL history (not pre-sliced) with
    indicators and entry/exit/short_entry/short_exit signal columns already
    computed on the complete series (see src.orchestrator for why this is
    both faster and more correct than computing indicators separately per
    fold -- ASSUMPTIONS.md #27). This function does the fold slicing itself.
    """
    periods_per_year = int(cfg.data.get("periods_per_year", 252))
    v = cfg.validation

    folds = split_folds(df_full_signals, cfg)
    search_folds = folds[:-1]  # last fold is the reserved holdout, never touched here

    fold_metrics = []
    fold_results = []
    for fold_df in search_folds:
        result = run_backtest(fold_df, strategy, cfg, symbol)
        metrics = compute_metrics(result.equity_curve["equity"], result.trades, periods_per_year)
        fold_metrics.append(metrics)
        fold_results.append(result)

    fold_sharpes = [m["sharpe"] for m in fold_metrics]
    fold_trades = [int(m["num_trades"]) for m in fold_metrics]
    fold_drawdowns = [m["max_drawdown"] for m in fold_metrics]
    # profit_factor is +inf when a fold has zero losing trades (genuinely the
    # best possible outcome, not an undefined one) -- cap it at a large but
    # finite value for averaging instead of dropping it. Dropping it is
    # actively backwards: if EVERY fold happens to have zero losing trades
    # (plausible with few trades in a strong clean trend), filtering all of
    # them out leaves an empty list, and averaging an empty list would
    # otherwise silently fail a genuinely excellent strategy on this check.
    PROFIT_FACTOR_CAP = 10.0
    fold_profit_factors = [
        (pf if np.isfinite(pf) else PROFIT_FACTOR_CAP) for pf in (m["profit_factor"] for m in fold_metrics)
    ]

    avg_sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
    min_fold_sharpe = float(min(fold_sharpes)) if fold_sharpes else 0.0
    consistency = cross_fold_sharpe_consistency(fold_sharpes)
    worst_drawdown = float(min(fold_drawdowns)) if fold_drawdowns else 0.0  # most negative
    avg_profit_factor = float(np.mean(fold_profit_factors)) if fold_profit_factors else 0.0

    reasons = []
    if avg_sharpe < float(v.min_avg_fold_sharpe):
        reasons.append(f"avg fold Sharpe {avg_sharpe:.3f} < {v.min_avg_fold_sharpe}")
    if min_fold_sharpe < float(v.min_fold_sharpe_floor):
        reasons.append(
            f"worst fold Sharpe {min_fold_sharpe:.3f} < {v.min_fold_sharpe_floor} floor "
            f"(fold Sharpes: {[round(s, 3) for s in fold_sharpes]})"
        )
    if consistency < float(v.min_sharpe_consistency):
        reasons.append(
            f"cross-fold Sharpe consistency {consistency:.3f} < {v.min_sharpe_consistency} "
            f"(fold Sharpes: {[round(s, 3) for s in fold_sharpes]})"
        )
    if abs(worst_drawdown) > float(v.max_fold_drawdown):
        reasons.append(f"worst fold drawdown {worst_drawdown:.3f} exceeds {v.max_fold_drawdown}")
    if avg_profit_factor < float(v.min_avg_profit_factor):
        reasons.append(f"avg fold profit factor {avg_profit_factor:.3f} < {v.min_avg_profit_factor}")
    if any(t < int(v.min_trades_per_fold) for t in fold_trades):
        worst_i = min(range(len(fold_trades)), key=lambda i: fold_trades[i])
        reasons.append(
            f"fold {worst_i + 1} trades {fold_trades[worst_i]} < {v.min_trades_per_fold} "
            f"(fold trades: {fold_trades})"
        )

    accepted = len(reasons) == 0

    return FoldEvaluationResult(
        fold_metrics=fold_metrics,
        fold_results=fold_results,
        avg_fold_sharpe=avg_sharpe,
        min_fold_sharpe=min_fold_sharpe,
        sharpe_consistency=consistency,
        accepted=accepted,
        reasons=reasons,
    )


def run_final_holdout(df_full_signals: pd.DataFrame, strategy, cfg, symbol: str, avg_fold_sharpe: float) -> HoldoutResult:
    """Run the backtest ONCE on the reserved final holdout fold (the last
    fold from split_folds). Call this only after the search loop has
    already decided to accept a strategy -- never during the loop, and
    never feed its result back into the prompt or near-miss leaderboard,
    or it stops being a genuine holdout (see ASSUMPTIONS.md #27).
    """
    periods_per_year = int(cfg.data.get("periods_per_year", 252))
    folds = split_folds(df_full_signals, cfg)
    holdout_df = folds[-1]

    result = run_backtest(holdout_df, strategy, cfg, symbol)
    metrics = compute_metrics(result.equity_curve["equity"], result.trades, periods_per_year)
    consistency_vs_search = cross_fold_sharpe_consistency([avg_fold_sharpe, metrics["sharpe"]])

    return HoldoutResult(metrics=metrics, result=result, sharpe_consistency_vs_search=consistency_vs_search)
