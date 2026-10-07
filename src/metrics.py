"""Performance metrics computed manually (no external backtesting libs)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return float(dd.min()) if not dd.empty else 0.0


def compute_metrics(equity: pd.Series, trades: pd.DataFrame, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> dict:
    """Compute the full metrics dict for one equity curve + trade list."""
    equity = equity.dropna()
    if equity.empty:
        return _empty_metrics()

    initial_equity = float(equity.iloc[0])
    final_equity = float(equity.iloc[-1])
    total_return = (final_equity / initial_equity - 1.0) if initial_equity else 0.0

    n_periods = len(equity)
    years = n_periods / periods_per_year if periods_per_year else np.nan
    cagr = (final_equity / initial_equity) ** (1 / years) - 1 if (initial_equity and years and years > 0) else 0.0

    returns = _daily_returns(equity)
    ann_vol = float(returns.std(ddof=0) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0

    mean_excess = returns.mean() - risk_free_rate / periods_per_year if len(returns) else 0.0
    std_ret = returns.std(ddof=0)
    sharpe = float((mean_excess / std_ret) * np.sqrt(periods_per_year)) if std_ret and std_ret > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0)
    sortino = float((mean_excess / downside_std) * np.sqrt(periods_per_year)) if downside_std and downside_std > 0 else 0.0

    mdd = max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd not in (0.0, None) and abs(mdd) > 1e-9 else 0.0

    if trades is None or trades.empty:
        win_rate = 0.0
        profit_factor = 0.0
        num_trades = 0
        avg_trade_pnl = 0.0
        avg_trade_pnl_pct = 0.0
        avg_holding_bars = 0.0
        best_trade = 0.0
        worst_trade = 0.0
    else:
        pnl = trades["pnl"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        num_trades = len(trades)
        win_rate = float(len(wins) / num_trades) if num_trades else 0.0
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        avg_trade_pnl = float(pnl.mean())
        avg_trade_pnl_pct = float(trades["pnl_pct"].mean()) if "pnl_pct" in trades else 0.0
        avg_holding_bars = float(trades["holding_bars"].mean()) if "holding_bars" in trades else 0.0
        best_trade = float(pnl.max())
        worst_trade = float(pnl.min())

    exposure = 0.0
    if trades is not None and not trades.empty and "holding_bars" in trades and n_periods:
        exposure = float(trades["holding_bars"].sum() / n_periods)

    return {
        "total_return": total_return,
        "cagr": float(cagr) if cagr == cagr else 0.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": num_trades,
        "avg_trade_pnl": avg_trade_pnl,
        "avg_trade_pnl_pct": avg_trade_pnl_pct,
        "avg_holding_bars": avg_holding_bars,
        "exposure": exposure,
        "annualized_volatility": ann_vol,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "final_equity": final_equity,
    }


def _empty_metrics() -> dict:
    keys = [
        "total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar",
        "win_rate", "profit_factor", "num_trades", "avg_trade_pnl",
        "avg_trade_pnl_pct", "avg_holding_bars", "exposure",
        "annualized_volatility", "best_trade", "worst_trade", "final_equity",
    ]
    return {k: 0.0 for k in keys}


def fast_metrics(equity: np.ndarray, pnl: np.ndarray, periods_per_year: int = 252) -> dict:
    """The screening subset of `compute_metrics`, straight from numpy arrays
    (no pandas): sharpe, max_drawdown, profit_factor, num_trades, total_return.
    Numerically identical to compute_metrics for these keys (see tests)."""
    n_trades = int(len(pnl))
    out = {"sharpe": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0,
           "num_trades": n_trades, "total_return": 0.0}
    if equity.size == 0:
        return out
    first, last = float(equity[0]), float(equity[-1])
    out["total_return"] = (last / first - 1.0) if first else 0.0
    if equity.size > 1:
        prev = equity[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = equity[1:] / prev - 1.0
        rets = rets[np.isfinite(rets)]
        if rets.size:
            std = float(rets.std())
            if std > 0:
                out["sharpe"] = float(rets.mean() / std * np.sqrt(periods_per_year))
    running_max = np.maximum.accumulate(equity)
    out["max_drawdown"] = float(((equity - running_max) / running_max).min())
    if n_trades:
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl < 0].sum())
        out["profit_factor"] = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    return out
