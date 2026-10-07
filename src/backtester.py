"""Local daily futures backtester.

Rules implemented (see project spec):
- Signals computed at bar close; trades executed at NEXT bar's open
  (fallback to next bar's close if open unavailable).
- Long and short positions supported if config allows.
- Optional stop loss / take profit; if both could trigger in the same bar,
  stop loss takes priority.
- No lookahead, no fabricated missing-day fills.
- Position sizing: risk_based_contracts when a stop loss exists and
  instrument point_value is known, else default_contracts fallback.

PERFORMANCE: the per-bar loop lives in `_simulate`, a numba-compiled kernel
over raw numpy arrays (falls back to the same code interpreted if numba is not
installed). The universe pipeline calls it hundreds of times per strategy
(symbols x folds), so it must stay allocation-light and pandas-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils import get_logger

try:  # numba is an optional accelerator: same code path, ~100x faster when present
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba installed
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

logger = get_logger("backtester")


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    summary: dict


def _instrument_meta(cfg, symbol: str) -> dict:
    instruments = cfg.get("instruments", {}) or {}
    return dict(instruments.get(symbol, {})) if symbol in instruments else {}


# Exit-reason codes returned by the simulation kernel.
REASON_SIGNAL, REASON_STOP, REASON_TARGET, REASON_END = 0, 1, 2, 3
_REASON_NAMES = {REASON_SIGNAL: "signal_exit", REASON_STOP: "stop_loss",
                 REASON_TARGET: "take_profit", REASON_END: "end_of_data"}


@dataclass
class SimSettings:
    """Everything the kernel needs that does not depend on the bars."""
    allow_short: bool
    stop_loss_pct: float      # 0.0 == none
    take_profit_pct: float    # 0.0 == none
    initial_capital: float
    point_value: float        # 0.0 == unknown (return-based PnL fallback)
    cost_fixed: float         # $ per contract per side
    cost_pct: float           # fraction of price per contract per side
    risk_per_trade: float
    max_contracts: int
    default_contracts: int


def build_settings(cfg, instrument: dict, strategy, initial_capital: float | None = None) -> SimSettings:
    """Resolve config + instrument + strategy overrides into kernel inputs.

    Cost model: with point_value and tick_size known, commission +
    slippage_ticks*tick*point_value per contract per side; otherwise
    commission plus bps-of-price costs. Fractional contracts are not
    supported by the kernel (the shipped config has them disabled)."""
    bt = cfg.backtest
    sl = strategy.stop_loss_pct if strategy.stop_loss_pct is not None else bt.get("stop_loss_pct_default")
    tp = strategy.take_profit_pct if strategy.take_profit_pct is not None else bt.get("take_profit_pct_default")
    rpt = getattr(strategy, "risk_per_trade", None)
    if rpt is None:
        rpt = bt.get("risk_per_trade", 0.01)
    pv = instrument.get("point_value")
    tick = instrument.get("tick_size")
    commission = float(bt.get("commission_per_side_usd", 0.0))
    if pv and tick:
        fixed = commission + float(bt.get("slippage_ticks", 0)) * float(tick) * float(pv)
        pct = 0.0
    else:
        fixed = commission
        pct = (float(bt.get("slippage_bps", 0.0)) + float(bt.get("fee_bps", 0.0))) / 10_000.0
    return SimSettings(
        allow_short=bool(bt.get("allow_short", True)),
        stop_loss_pct=float(sl) if sl else 0.0,
        take_profit_pct=float(tp) if tp else 0.0,
        initial_capital=float(initial_capital if initial_capital is not None else bt.get("initial_capital", 100_000)),
        point_value=float(pv) if pv else 0.0,
        cost_fixed=fixed,
        cost_pct=pct,
        risk_per_trade=float(rpt),
        max_contracts=int(bt.get("max_contracts", 1)),
        default_contracts=int(bt.get("default_contracts", 1)),
    )


@njit(cache=True, nogil=True)
def _size(equity, entry_price, sl, pv, risk_pt, max_c, def_c):
    """Whole-contract position size: risk budget / (stop distance * point value),
    capped at max_c, falling back to default contracts when the budget is too
    small for even one contract."""
    if sl > 0.0:
        risk_amount = equity * risk_pt
        stop_distance = entry_price * sl
        if stop_distance > 0.0:
            if pv > 0.0:
                size = int(math.floor(risk_amount / stop_distance / pv))
            else:
                size = int(math.floor(risk_amount / stop_distance))
        else:
            size = def_c
        if size < 0:
            size = 0
    else:
        size = def_c
    if size > max_c:
        size = max_c
    if size > 0:
        return size
    return min(def_c, max_c)


@njit(cache=True, nogil=True)
def _simulate(open_, high, low, close, le, lx, se, sx, allow_short, sl, tp, init_cap,
              pv, cost_fixed, cost_pct, risk_pt, max_c, def_c):
    """Single-position long/short next-bar-open simulation over raw arrays.

    Signals computed on bar i-1 are filled at bar i's open (close if open is
    NaN); signal exits are processed before entries; stop beats target when
    both are touched in one bar; any open position is closed at the final
    close. Returns (equity[n], n_trades, then per-trade arrays of length n)."""
    n = close.shape[0]
    equity_curve = np.empty(n)
    t_entry = np.zeros(n, dtype=np.int64)
    t_exit = np.zeros(n, dtype=np.int64)
    t_dir = np.zeros(n, dtype=np.int64)
    t_con = np.zeros(n, dtype=np.int64)
    t_epx = np.zeros(n)
    t_xpx = np.zeros(n)
    t_pnl = np.zeros(n)
    t_cost = np.zeros(n)
    t_reason = np.zeros(n, dtype=np.int64)
    nt = 0

    equity = init_cap
    pos = 0
    contracts = 0
    entry_price = np.nan
    entry_idx = -1

    for i in range(n):
        px = open_[i]
        if np.isnan(px):
            px = close[i]
        bar_high = high[i]
        bar_low = low[i]
        bar_close = close[i]

        # 1) signal exit (from the previous bar's signal)
        if pos != 0 and i > 0:
            do_exit = lx[i - 1] if pos == 1 else sx[i - 1]
            if do_exit:
                cost = contracts * (cost_fixed + px * cost_pct)
                if pv > 0.0:
                    pnl = (px - entry_price) * pos * contracts * pv - cost
                else:
                    pnl = (px - entry_price) / entry_price * pos * equity - cost
                equity += pnl
                t_entry[nt] = entry_idx
                t_exit[nt] = i
                t_dir[nt] = pos
                t_con[nt] = contracts
                t_epx[nt] = entry_price
                t_xpx[nt] = px
                t_pnl[nt] = pnl
                t_cost[nt] = cost
                t_reason[nt] = 0
                nt += 1
                pos = 0
                contracts = 0
                entry_price = np.nan
                entry_idx = -1

        # 2) entry from the previous bar's signal (long wins a same-bar tie)
        if pos == 0 and i > 0:
            direction = 0
            if le[i - 1]:
                direction = 1
            elif allow_short and se[i - 1]:
                direction = -1
            if direction != 0:
                pos = direction
                entry_price = px
                entry_idx = i
                contracts = _size(equity, entry_price, sl, pv, risk_pt, max_c, def_c)
                if contracts <= 0:
                    pos = 0
                    entry_price = np.nan
                    entry_idx = -1

        # 3) stop / target management, including on the entry bar
        if pos != 0:
            stop_price = np.nan
            target_price = np.nan
            if sl > 0.0:
                stop_price = entry_price * (1 - sl) if pos == 1 else entry_price * (1 + sl)
            if tp > 0.0:
                target_price = entry_price * (1 + tp) if pos == 1 else entry_price * (1 - tp)
            hit_stop = False
            hit_target = False
            if not np.isnan(bar_low) and not np.isnan(bar_high):
                if not np.isnan(stop_price):
                    hit_stop = (bar_low <= stop_price) if pos == 1 else (bar_high >= stop_price)
                if not np.isnan(target_price):
                    hit_target = (bar_high >= target_price) if pos == 1 else (bar_low <= target_price)
            if hit_stop or hit_target:
                xpx = stop_price if hit_stop else target_price
                cost = contracts * (cost_fixed + xpx * cost_pct)
                if pv > 0.0:
                    pnl = (xpx - entry_price) * pos * contracts * pv - cost
                else:
                    pnl = (xpx - entry_price) / entry_price * pos * equity - cost
                equity += pnl
                t_entry[nt] = entry_idx
                t_exit[nt] = i
                t_dir[nt] = pos
                t_con[nt] = contracts
                t_epx[nt] = entry_price
                t_xpx[nt] = xpx
                t_pnl[nt] = pnl
                t_cost[nt] = cost
                t_reason[nt] = 1 if hit_stop else 2
                nt += 1
                pos = 0
                contracts = 0
                entry_price = np.nan
                entry_idx = -1

        # 4) mark to market
        if pos != 0 and not np.isnan(bar_close):
            if pv > 0.0:
                unreal = (bar_close - entry_price) * pos * contracts * pv
            else:
                unreal = ((bar_close - entry_price) / entry_price) * pos * equity
            equity_curve[i] = equity + unreal
        else:
            equity_curve[i] = equity

    if pos != 0:
        xpx = close[n - 1]
        cost = contracts * (cost_fixed + xpx * cost_pct)
        if pv > 0.0:
            pnl = (xpx - entry_price) * pos * contracts * pv - cost
        else:
            pnl = (xpx - entry_price) / entry_price * pos * equity - cost
        equity += pnl
        t_entry[nt] = entry_idx
        t_exit[nt] = n - 1
        t_dir[nt] = pos
        t_con[nt] = contracts
        t_epx[nt] = entry_price
        t_xpx[nt] = xpx
        t_pnl[nt] = pnl
        t_cost[nt] = cost
        t_reason[nt] = 3
        nt += 1
        equity_curve[n - 1] = equity

    return equity_curve, nt, t_entry, t_exit, t_dir, t_con, t_epx, t_xpx, t_pnl, t_cost, t_reason


def simulate_arrays(open_, high, low, close, le, lx, se, sx, s: SimSettings):
    """Run the kernel on raw numpy arrays (float64 prices, bool signals)."""
    return _simulate(open_, high, low, close, le, lx, se, sx, s.allow_short,
                     s.stop_loss_pct, s.take_profit_pct, s.initial_capital, s.point_value,
                     s.cost_fixed, s.cost_pct, s.risk_per_trade, s.max_contracts, s.default_contracts)


def run_backtest(df: pd.DataFrame, strategy, cfg, symbol: str, instrument: dict | None = None,
                 initial_capital: float | None = None) -> BacktestResult:
    """Run a single-position, long/short, next-bar-open backtest and return
    full pandas artifacts (equity curve, trades table).

    `df` must already contain `entry_signal` / `exit_signal` (long) and
    `short_entry_signal` / `short_exit_signal` (short) boolean columns plus
    OHLCV columns. Shorts are only taken when `cfg.backtest.allow_short` is
    true (a strategy that never emits short_entry simply never shorts).

    `instrument` overrides the config's per-symbol instrument table (the
    universe pipeline passes tick/point-value read from the .dat header).
    """
    instrument = instrument if instrument is not None else _instrument_meta(cfg, symbol)
    settings = build_settings(cfg, instrument, strategy, initial_capital)

    df = df.reset_index()
    n = len(df)
    nan_arr = np.full(n, np.nan)
    false_arr = np.zeros(n, dtype=bool)

    def col(name, default, dtype):
        return df[name].to_numpy(dtype=dtype) if name in df.columns else default

    datetime_arr = df["datetime"].to_numpy()
    (equity_curve, nt, t_entry, t_exit, t_dir, t_con, t_epx, t_xpx, t_pnl, t_cost, t_reason) = simulate_arrays(
        col("open", nan_arr, float), col("high", nan_arr, float), col("low", nan_arr, float),
        col("close", nan_arr, float),
        col("entry_signal", false_arr, bool), col("exit_signal", false_arr, bool),
        col("short_entry_signal", false_arr, bool), col("short_exit_signal", false_arr, bool),
        settings,
    )

    initial = settings.initial_capital
    trades_df = pd.DataFrame({
        "strategy_name": strategy.strategy_name,
        "symbol": symbol,
        "entry_time": datetime_arr[t_entry[:nt]],
        "exit_time": datetime_arr[t_exit[:nt]],
        "direction": np.where(t_dir[:nt] == 1, "long", "short"),
        "entry_price": t_epx[:nt],
        "exit_price": t_xpx[:nt],
        "contracts": t_con[:nt],
        "size": t_con[:nt],
        "pnl": t_pnl[:nt],
        "pnl_pct": (t_pnl[:nt] / initial) if initial else np.nan,
        "fees_paid": t_cost[:nt],
        "holding_bars": t_exit[:nt] - t_entry[:nt],
        "exit_reason": [_REASON_NAMES[int(r)] for r in t_reason[:nt]],
    })
    equity_df = pd.DataFrame({"datetime": datetime_arr, "equity": equity_curve}).set_index("datetime")
    summary = {
        "final_equity": float(equity_curve[-1]) if n else initial,
        "num_trades": int(nt),
    }
    return BacktestResult(equity_curve=equity_df, trades=trades_df, positions=pd.DataFrame(), summary=summary)
