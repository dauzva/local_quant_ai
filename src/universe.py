"""Multi-symbol ("universe") evaluation.

A generated strategy is no longer judged on one instrument: it is run through
every symbol in the data folder (~60) with the same multi-fold walk-forward
scheme, and the per-symbol outcomes are aggregated into a universe score. A
real edge should show up broadly; a lucky fit to one chart should not.

Everything in the hot path works on raw numpy arrays (see
src.backtester._simulate) -- the pandas DataFrame is only touched to run the
strategy's own signals() once per symbol.

The history of every symbol is cut in time order into three segments, exactly
like train / validation / test in machine learning:

    [ IS: walk-forward folds | OSS1: validation | OSS2: test ]

- IS is what the search optimises on (`evaluate_symbol`): the local tuner's fitness and
  the walk-forward acceptance criteria use IS only.
- OSS1 is validation (`run_universe_oss1`): the best strategies of every generation are
  run on it, and the result decides acceptance, parent selection, what the LLM is told
  about its parents, and the final ranking. It never enters the tuner's fitness.
- OSS2 is the test set. `run_universe_oss2` is called from exactly one place,
  src.final_test (`python main.py pipeline` at its very end, or `final-test`); the
  generation loop itself never evaluates it.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtester import SimSettings, build_settings, simulate_arrays
from src.data_loader import discover_dat_files, read_trade_station_dat
from src.metrics import fast_metrics
from src.utils import get_logger

logger = get_logger("universe")

PROFIT_FACTOR_CAP = 10.0

# (group, regex over "<description> <symbol>") -- first match wins.
_GROUP_RULES = [
    ("crypto", r"bitcoin|\bbtc\b"),
    ("volatility", r"volatility|\bvix\b"),
    ("rates", r"treasury|t-note|t-bond|bund|bobl|schatz|buxl|gilt|euribor|eurodollar|\bnotes?\b|\bbonds?\b"),
    ("equity_index", r"index|e-?mini|dax|nikkei|ftse|msci|stoxx|russell|dow|nasdaq|s&p|midcap"),
    ("fx", r"dollar|euro fx|yen|pound|peso|franc|real\b|canadian|australian|zealand|cross rate|currency"),
    ("metals", r"gold|silver|platinum|palladium|copper"),
    ("energy", r"crude|brent|heating|rbob|blendstock|natural gas|gasoil|gasoline|wti"),
    ("livestock", r"cattle|hogs?\b|milk"),
    ("softs", r"cocoa|coffee|cotton|sugar|juice|\boj\b|robusta|lumber"),
    ("grains", r"corn|soy|wheat|oats?|rice|rapeseed|canola"),
]


def classify_group(description: str, symbol: str) -> str:
    text = f"{description} {symbol}".lower()
    for group, pattern in _GROUP_RULES:
        if re.search(pattern, text):
            return group
    return "other"


@dataclass
class SymbolData:
    symbol: str
    group: str
    df: pd.DataFrame
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    bounds: list[tuple[int, int]]      # walk-forward fold slices INSIDE the IS segment
    oss1: tuple[int, int]              # validation segment (bar slice)
    oss2: tuple[int, int]              # test segment (bar slice); only src.final_test reads it
    instrument: dict
    initial_capital: float
    years: float

    @property
    def n(self) -> int:
        return len(self.close)


@dataclass
class SymbolResult:
    symbol: str
    group: str
    avg_sharpe: float = 0.0
    min_sharpe: float = 0.0
    consistency: float = 0.0
    fold_sharpes: list[float] = field(default_factory=list)
    fold_trades: list[int] = field(default_factory=list)
    total_trades: int = 0
    trades_per_year: float = 0.0
    avg_hold: float = 0.0
    max_dd: float = 0.0
    long_pnl_pct: float = 0.0      # net PnL of long / short trades, % of initial capital
    short_pnl_pct: float = 0.0
    cost_pct: float = 0.0          # commissions + slippage paid, % of initial capital (per fold, averaged)
    gross_pnl_pct: float = 0.0     # net PnL + costs: does the signal have an edge BEFORE costs?
    passed: bool = False
    fail_codes: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class EvalSettings:
    """Thresholds + backtest knobs resolved once from the config."""
    num_folds: int
    min_trades_per_fold: int
    min_trades_per_year: float
    min_avg_fold_sharpe: float
    min_fold_sharpe_floor: float
    min_sharpe_consistency: float
    max_fold_drawdown: float
    min_avg_profit_factor: float
    periods_per_year: int
    beta_adjust: bool = False     # score = Sharpe minus what the same market exposure earns by luck (see exposure_baseline)


def build_eval_settings(cfg) -> EvalSettings:
    v = cfg.validation
    return EvalSettings(
        num_folds=int(v.num_folds),
        min_trades_per_fold=int(v.min_trades_per_fold),
        min_trades_per_year=float(v.get("min_trades_per_year", 0.0) or 0.0),
        min_avg_fold_sharpe=float(v.min_avg_fold_sharpe),
        min_fold_sharpe_floor=float(v.min_fold_sharpe_floor),
        min_sharpe_consistency=float(v.min_sharpe_consistency),
        max_fold_drawdown=float(v.max_fold_drawdown),
        min_avg_profit_factor=float(v.min_avg_profit_factor),
        periods_per_year=int(cfg.data.get("periods_per_year", 252)),
        beta_adjust=bool((cfg.get("scoring", {}) or {}).get("beta_adjust", False)),
    )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _fold_bounds(n: int, num_folds: int) -> list[tuple[int, int]]:
    """Contiguous, time-ordered folds covering bars [0, n) (np.array_split semantics)."""
    sizes = [n // num_folds + (1 if i < n % num_folds else 0) for i in range(num_folds)]
    bounds, start = [], 0
    for s in sizes:
        bounds.append((start, start + s))
        start += s
    return bounds


def segment_edges(index: pd.DatetimeIndex, split_cfg) -> tuple[int, int]:
    """Bar positions (is_n, oss1_n) such that IS = [0, is_n), OSS1 = [is_n, oss1_n),
    OSS2 = [oss1_n, len). With `split.is_end` / `split.oss1_end` set, the cutoffs are
    calendar dates shared by every symbol; otherwise per-symbol fractions."""
    n = len(index)
    is_end, oss1_end = split_cfg.get("is_end"), split_cfg.get("oss1_end")
    if is_end and oss1_end:
        def pos(date) -> int:
            ts = pd.Timestamp(date)
            if index.tz is not None and ts.tzinfo is None:
                ts = ts.tz_localize(index.tz)
            return int(index.searchsorted(ts, side="right"))     # bars dated <= cutoff
        a, b = pos(is_end), pos(oss1_end)
        if a > b:
            raise ValueError(f"split.is_end ({is_end}) must be before split.oss1_end ({oss1_end})")
        return a, b
    is_frac = float(split_cfg.get("is_frac", 0.6))
    oss1_frac = float(split_cfg.get("oss1_frac", 0.2))
    if not 0 < is_frac < 1 or not 0 < oss1_frac < 1 or is_frac + oss1_frac >= 1:
        raise ValueError("split.is_frac and split.oss1_frac must be in (0, 1) and leave room for OSS2")
    a = int(round(n * is_frac))
    return a, int(round(n * (is_frac + oss1_frac)))


def select_symbols(cfg, available: list[str], requested: list[str] | None = None) -> list[str]:
    """Apply universe.exclude / universe.symbols / CLI request to the symbols
    that have a data file. An explicit CLI request bypasses the exclude list."""
    ucfg = cfg.get("universe", {}) or {}
    if requested:
        missing = [s for s in requested if s not in available]
        if missing:
            raise FileNotFoundError(f"No data file for symbol(s) {missing}. Available: {sorted(available)}")
        return list(requested)
    configured = ucfg.get("symbols", "all")
    chosen = sorted(available) if configured in (None, "all") else [s for s in configured if s in available]
    exclude = set(ucfg.get("exclude", []) or [])
    return [s for s in chosen if s not in exclude]


def _data_quality_problem(df: pd.DataFrame, ucfg) -> str | None:
    """Percent-based stops/targets/signals are meaningless on back-adjusted
    continuous series whose prices go to or below zero (observed on cocoa,
    heating oil, RBOB, soy meal, gasoil...: they produced fake Sharpe 5-7
    results) or that carry thousands of 20%+ daily jumps. Such series are
    excluded automatically."""
    close = df["close"].to_numpy(dtype=float)
    nonpos = float((df[["open", "high", "low", "close"]].to_numpy(dtype=float) <= 0).any(axis=1).mean())
    if nonpos > float(ucfg.get("max_nonpositive_frac", 0.0005)):
        return f"{nonpos:.1%} of bars have non-positive prices (back-adjusted series)"
    with np.errstate(all="ignore"):
        moves = np.abs(np.diff(close) / close[:-1])
    moves = moves[np.isfinite(moves)]
    # A single bar that more than doubles/halves the price is a data error, not a market move
    # (@RP=13INC jumps from 0.97 to 697 for six weeks around Dec 2003). One such bar is rare
    # enough to slip under the fraction test below yet it dominates every PnL number.
    max_move = float(ucfg.get("max_single_move", 1.0) or 0.0)
    if max_move > 0 and len(moves) and float(moves.max()) > max_move:
        return f"a single daily move of {float(moves.max()):.0%} (corrupt bar / contract mix-up)"
    big = float((moves > 0.2).mean()) if len(moves) else 0.0
    if big > float(ucfg.get("max_big_move_frac", 0.003)):
        return f"{big:.1%} of daily moves exceed 20% (broken/adjusted series)"
    return None


def load_universe(cfg, requested: list[str] | None = None) -> list[SymbolData]:
    """Load every selected symbol once (header tick/point-value included).

    Per-symbol initial capital: max(backtest.initial_capital,
    backtest.capital_per_notional * median 1-contract notional), so one
    contract of an index future is not a 3x-levered bet on a $100k account
    while a grain contract is a sensible position -- keeps drawdown and
    sizing comparable across the universe."""
    ucfg = cfg.get("universe", {}) or {}
    min_bars = int(ucfg.get("min_bars", 2500))
    max_symbols = ucfg.get("max_symbols")
    capital_per_notional = float(cfg.backtest.get("capital_per_notional", 0.0) or 0.0)
    base_capital = float(cfg.backtest.get("initial_capital", 100_000))
    num_folds = int(cfg.validation.num_folds)
    split_cfg = cfg.get("split", {}) or {}
    min_is_bars = int(split_cfg.get("min_is_bars", num_folds * 100))
    min_oss_bars = int(split_cfg.get("min_oss_bars", 250))

    file_map = discover_dat_files(cfg)
    symbols = select_symbols(cfg, list(file_map.keys()), requested)

    universe: list[SymbolData] = []
    for symbol in symbols:
        meta_fn = file_map[symbol]
        try:
            df, meta = read_trade_station_dat(Path(meta_fn.file_path), filename_meta=meta_fn, cfg=cfg)
        except Exception as exc:
            logger.warning("Skipping %s: failed to load (%s)", symbol, exc)
            continue
        if len(df) < min_bars and not requested:
            logger.info("Skipping %s: only %d bars (< universe.min_bars=%d)", symbol, len(df), min_bars)
            continue
        is_n, oss1_n = segment_edges(df.index, split_cfg)
        if is_n < min_is_bars or oss1_n - is_n < min_oss_bars or len(df) - oss1_n < min_oss_bars:
            logger.info("Skipping %s: too little history for the IS/OSS1/OSS2 split (%d / %d / %d bars; need >= %d / %d / %d)",
                        symbol, is_n, oss1_n - is_n, len(df) - oss1_n, min_is_bars, min_oss_bars, min_oss_bars)
            continue
        bad = _data_quality_problem(df, ucfg)
        if bad and not requested:
            logger.info("Skipping %s: %s", symbol, bad)
            continue
        if bad:
            logger.warning("%s has data-quality problems (%s); results on it are unreliable", symbol, bad)
        instrument = dict(cfg.get("instruments", {}).get(symbol, {}) or {})
        if not instrument.get("point_value") and meta.bpv:
            instrument["point_value"] = float(meta.bpv)
        if not instrument.get("tick_size") and meta.tick:
            instrument["tick_size"] = float(meta.tick)
        capital = base_capital
        if capital_per_notional > 0 and instrument.get("point_value"):
            notional = float(np.median(df["close"].to_numpy())) * float(instrument["point_value"])
            capital = max(base_capital, capital_per_notional * notional)
        universe.append(SymbolData(
            symbol=symbol,
            group=classify_group(meta.description or "", symbol),
            df=df,
            open_=df["open"].to_numpy(dtype=float),
            high=df["high"].to_numpy(dtype=float),
            low=df["low"].to_numpy(dtype=float),
            close=df["close"].to_numpy(dtype=float),
            bounds=_fold_bounds(is_n, num_folds),
            oss1=(is_n, oss1_n),
            oss2=(oss1_n, len(df)),
            instrument=instrument,
            initial_capital=capital,
            years=is_n / float(cfg.data.get("periods_per_year", 252)),
        ))
    if max_symbols:
        universe = universe[: int(max_symbols)]
    return universe


def pick_probe(universe: list[SymbolData], n: int) -> list[SymbolData]:
    """A deterministic, group-diverse subset: round-robin one symbol per
    asset group, so a quick pre-screen sees equities, rates, FX, commodities."""
    if n <= 0 or n >= len(universe):
        return list(universe)
    by_group: dict[str, list[SymbolData]] = {}
    for sd in sorted(universe, key=lambda s: s.symbol):
        by_group.setdefault(sd.group, []).append(sd)
    picked: list[SymbolData] = []
    i = 0
    while len(picked) < n:
        progressed = False
        for g in sorted(by_group):
            if i < len(by_group[g]) and len(picked) < n:
                picked.append(by_group[g][i])
                progressed = True
        if not progressed:
            break
        i += 1
    return picked


# --------------------------------------------------------------------------
# Per-symbol evaluation
# --------------------------------------------------------------------------

def compute_signal_arrays(compiled, sd: SymbolData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    le, lx, se, sx = compiled.signals(sd.df)
    return tuple(np.ascontiguousarray(s.to_numpy(dtype=bool)) for s in (le, lx, se, sx))


def position_vector(compiled, universe: list[SymbolData], cfg, sig_cache: dict | None = None) -> np.ndarray:
    """What a strategy actually DOES: +1 (long) / -1 (short) / 0 (flat) for every bar of every
    symbol, concatenated. Two strategies whose vectors correlate highly are the same bet with
    different parameters, whatever their code or family label says."""
    parts = []
    for sd in universe:
        sigs = (sig_cache or {}).get(sd.symbol) or compute_signal_arrays(compiled, sd)
        le, lx, se, sx = sigs
        settings = build_settings(cfg, sd.instrument, compiled, sd.initial_capital)
        _eq, nt, ent, ext, tdir, *_ = simulate_arrays(sd.open_, sd.high, sd.low, sd.close, le, lx, se, sx, settings)
        pos = np.zeros(sd.n, dtype=np.float32)
        for i in range(nt):
            pos[ent[i]:ext[i]] = tdir[i]
        parts.append(pos)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def position_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or a.size != b.size or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def exposure_baseline(close: np.ndarray, t_entry: np.ndarray, t_exit: np.ndarray, t_dir: np.ndarray, ppy: int) -> float:
    """Sharpe that the SAME market exposure would earn with no timing skill.

    A long-only strategy that sits in the market a fraction f of the time earns, with random
    timing, a mean of f*mu and a volatility of sqrt(f)*sigma, i.e. Sharpe = sqrt(f) * SharpeBuyHold.
    With long exposure eL and short exposure eS (fractions of the bars) the general form is
    (eL - eS) / sqrt(eL + eS) * SharpeBuyHold. Generated strategies were found to be ~all long-only
    and to score exactly what that formula gives (they harvested the up-drift of futures markets
    in the IS/OSS1 years, which was gone in OSS2), so a strategy is credited only for the part of
    its Sharpe above this baseline."""
    n = len(close)
    if n < 3 or len(t_entry) == 0:
        return 0.0
    held = np.clip(t_exit - t_entry, 0, None)
    e_long = float(held[t_dir == 1].sum()) / n
    e_short = float(held[t_dir == -1].sum()) / n
    if e_long + e_short <= 0:
        return 0.0
    with np.errstate(all="ignore"):
        rets = np.diff(close) / close[:-1]
    rets = rets[np.isfinite(rets)]
    if rets.size < 2 or rets.std() <= 0:
        return 0.0
    sharpe_bh = float(rets.mean() / rets.std() * np.sqrt(ppy))
    return (e_long - e_short) / np.sqrt(e_long + e_short) * sharpe_bh


def _run_fold(sd: SymbolData, sigs, settings: SimSettings, a: int, b: int, ppy: int,
              beta_adjust: bool = False) -> tuple[dict, dict]:
    le, lx, se, sx = sigs
    equity, nt, _ent, t_exit, t_dir, _con, _epx, _xpx, t_pnl, _cost, _rsn = simulate_arrays(
        sd.open_[a:b], sd.high[a:b], sd.low[a:b], sd.close[a:b],
        le[a:b], lx[a:b], se[a:b], sx[a:b], settings,
    )
    pnl = t_pnl[:nt]
    m = fast_metrics(equity, pnl, ppy)
    m["sharpe_raw"] = m["sharpe"]
    m["baseline"] = exposure_baseline(sd.close[a:b], _ent[:nt], t_exit[:nt], t_dir[:nt], ppy)
    if beta_adjust:
        m["sharpe"] = m["sharpe_raw"] - m["baseline"]
    direction = t_dir[:nt]
    extra = {
        "long_pnl": float(pnl[direction == 1].sum()),
        "short_pnl": float(pnl[direction == -1].sum()),
        "hold_sum": float((t_exit[:nt] - _ent[:nt]).sum()),
        "cost": float(_cost[:nt].sum()),
        "net": float(pnl.sum()),
    }
    return m, extra


def _cross_fold_consistency(sharpes: list[float]) -> float:
    if not sharpes:
        return 1.0
    avg = sum(sharpes) / len(sharpes)
    if avg <= 0:
        return 0.0
    return max(0.0, min(1.0, min(sharpes) / avg))


def evaluate_symbol(compiled, sd: SymbolData, ev: EvalSettings, cfg, sigs=None) -> SymbolResult:
    """Walk-forward evaluation of one strategy on one symbol over the IS folds only."""
    res = SymbolResult(symbol=sd.symbol, group=sd.group)
    try:
        if sigs is None:
            sigs = compute_signal_arrays(compiled, sd)
        settings = build_settings(cfg, sd.instrument, compiled, sd.initial_capital)
        metrics, extras = [], []
        for a, b in sd.bounds:
            m, x = _run_fold(sd, sigs, settings, a, b, ev.periods_per_year, ev.beta_adjust)
            metrics.append(m)
            extras.append(x)
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
        res.fail_codes = ["error"]
        return res

    sharpes = [m["sharpe"] for m in metrics]
    trades = [int(m["num_trades"]) for m in metrics]
    pfs = [m["profit_factor"] if np.isfinite(m["profit_factor"]) else PROFIT_FACTOR_CAP for m in metrics]
    res.fold_sharpes = [round(s, 4) for s in sharpes]
    res.fold_trades = trades
    res.avg_sharpe = float(np.mean(sharpes))
    res.min_sharpe = float(min(sharpes))
    res.consistency = _cross_fold_consistency(sharpes)
    res.max_dd = float(min(m["max_drawdown"] for m in metrics))
    res.total_trades = int(sum(trades))
    search_years = sum(b - a for a, b in sd.bounds) / max(1, ev.periods_per_year)
    res.trades_per_year = res.total_trades / search_years if search_years else 0.0
    res.avg_hold = (sum(x["hold_sum"] for x in extras) / res.total_trades) if res.total_trades else 0.0
    res.long_pnl_pct = sum(x["long_pnl"] for x in extras) / (settings.initial_capital * len(extras)) * 100.0
    res.short_pnl_pct = sum(x["short_pnl"] for x in extras) / (settings.initial_capital * len(extras)) * 100.0
    res.cost_pct = sum(x["cost"] for x in extras) / (settings.initial_capital * len(extras)) * 100.0
    res.gross_pnl_pct = sum(x["net"] + x["cost"] for x in extras) / (settings.initial_capital * len(extras)) * 100.0

    codes = []
    if res.avg_sharpe < ev.min_avg_fold_sharpe:
        codes.append("low_sharpe")
    if res.min_sharpe < ev.min_fold_sharpe_floor:
        codes.append("negative_fold")
    if res.consistency < ev.min_sharpe_consistency:
        codes.append("inconsistent")
    if abs(res.max_dd) > ev.max_fold_drawdown:
        codes.append("drawdown")
    if float(np.mean(pfs)) < ev.min_avg_profit_factor:
        codes.append("low_profit_factor")
    # Frequency requirement scales with fold length: a 4-year fold must hold
    # min_trades_per_year*4 trades (floor: min_trades_per_fold). A strategy that
    # trades 2-5x a year is a handful of lucky positions, not a strategy.
    needed = [max(ev.min_trades_per_fold, int(round(ev.min_trades_per_year * (b - a) / max(1, ev.periods_per_year))))
              for a, b in sd.bounds]
    if any(t < n for t, n in zip(trades, needed)):
        codes.append("low_trades")
    res.fail_codes = codes
    res.passed = not codes
    return res


def segment_symbol(compiled, sd: SymbolData, ev: EvalSettings, cfg, segment: tuple[int, int], sigs=None) -> dict:
    """Evaluate one strategy on one symbol over a single out-of-sample bar slice.
    Signals come from the full history (indicators warm up on earlier data), exactly
    as in the IS folds."""
    try:
        if sigs is None:
            sigs = compute_signal_arrays(compiled, sd)
        settings = build_settings(cfg, sd.instrument, compiled, sd.initial_capital)
        a, b = segment
        m, _ = _run_fold(sd, sigs, settings, a, b, ev.periods_per_year, ev.beta_adjust)
        out = {"symbol": sd.symbol, "group": sd.group, "sharpe": m["sharpe"], "raw_sharpe": m["sharpe_raw"],
               "trades": int(m["num_trades"]), "max_dd": m["max_drawdown"], "total_return": m["total_return"]}
        mid = (a + b) // 2
        if mid - a >= 120 and b - mid >= 120:        # two halves: an edge should not live in only one of them
            out["sharpe_h1"] = _run_fold(sd, sigs, settings, a, mid, ev.periods_per_year, ev.beta_adjust)[0]["sharpe"]
            out["sharpe_h2"] = _run_fold(sd, sigs, settings, mid, b, ev.periods_per_year, ev.beta_adjust)[0]["sharpe"]
        return out
    except Exception as exc:
        return {"symbol": sd.symbol, "group": sd.group, "sharpe": 0.0, "trades": 0, "max_dd": 0.0,
                "total_return": 0.0, "error": str(exc)}


def behavior_signature(compiled, symbols: list[SymbolData], sig_cache: dict | None = None) -> str:
    """Hash of the actual entry/exit signal arrays on a few reference symbols:
    catches strategies whose code differs but whose behavior is identical."""
    h = hashlib.sha1()
    for sd in symbols:
        sigs = compute_signal_arrays(compiled, sd)
        if sig_cache is not None:
            sig_cache[sd.symbol] = sigs
        for arr in sigs:
            h.update(np.packbits(arr).tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# Universe aggregation
# --------------------------------------------------------------------------

@dataclass
class UniverseResult:
    per_symbol: list[SymbolResult]
    summary: dict
    probe_only: bool = False
    seconds: float = 0.0


def summarize(results: list[SymbolResult], accept_cfg: dict) -> dict:
    """Aggregate per-symbol results into the universe score + acceptance.

    score = mean of per-symbol avg fold Sharpe, each clipped to [-1.5, 1.5] so
    one freak symbol cannot carry a strategy. Acceptance needs breadth: enough
    symbols individually passing the walk-forward criteria AND a majority
    profitable AND a decent score."""
    ok = [r for r in results if r.error is None]
    if not ok:
        return {"n_symbols": len(results), "n_eval": 0, "score": -9.0, "accepted": False,
                "error_symbols": len(results)}
    sharpes = np.array([r.avg_sharpe for r in ok])
    clipped = np.clip(sharpes, -1.5, 1.5)
    n_pass = sum(1 for r in ok if r.passed)
    pass_rate = n_pass / len(ok)
    pct_pos = float((sharpes > 0).mean())
    sharpe_score = float(clipped.mean())
    tpy = float(np.mean([r.trades_per_year for r in ok]))
    min_tpy = float(accept_cfg.get("min_trades_per_year", 0.0) or 0.0)
    # Frequency-adjusted score: the number everything else ranks by. A strategy
    # trading below the target frequency keeps only tpy/target of its Sharpe.
    freq_factor = min(1.0, tpy / min_tpy) if min_tpy > 0 else 1.0
    score = sharpe_score * freq_factor if sharpe_score > 0 else sharpe_score

    groups: dict[str, list[float]] = {}
    for r in ok:
        groups.setdefault(r.group, []).append(r.avg_sharpe)
    group_mean = {g: round(float(np.mean(v)), 3) for g, v in groups.items()}

    fail_counts: dict[str, int] = {}
    for r in ok:
        for c in r.fail_codes:
            fail_counts[c] = fail_counts.get(c, 0) + 1

    ranked = sorted(ok, key=lambda r: r.avg_sharpe, reverse=True)
    summary = {
        "n_symbols": len(results),
        "n_eval": len(ok),
        "score": round(score, 4),
        "sharpe_score": round(sharpe_score, 4),
        "freq_factor": round(freq_factor, 3),
        "median_sharpe": round(float(np.median(sharpes)), 4),
        "pct_positive": round(pct_pos, 3),
        "n_pass": n_pass,
        "pass_rate": round(pass_rate, 3),
        "trades_per_year": round(tpy, 2),
        "trades_per_symbol": int(np.median([r.total_trades for r in ok])),
        "total_trades": int(sum(r.total_trades for r in ok)),
        "avg_hold_bars": round(float(np.mean([r.avg_hold for r in ok])), 1),
        "cost_pct": round(float(np.mean([r.cost_pct for r in ok])), 2),
        "gross_pnl_pct": round(float(np.mean([r.gross_pnl_pct for r in ok])), 2),
        "long_pnl_pct": round(float(np.mean([r.long_pnl_pct for r in ok])), 2),
        "short_pnl_pct": round(float(np.mean([r.short_pnl_pct for r in ok])), 2),
        "group_mean_sharpe": group_mean,
        "fail_counts": fail_counts,
        "best_symbols": [(r.symbol, round(r.avg_sharpe, 2)) for r in ranked[:3]],
        "worst_symbols": [(r.symbol, round(r.avg_sharpe, 2)) for r in ranked[-3:]],
        "no_trade_symbols": sum(1 for r in ok if r.total_trades == 0),
    }
    # Fitness = what the searches maximise: the frequency-adjusted Sharpe score plus
    # a bonus for breadth of symbols clearing EVERY walk-forward criterion, which is
    # what acceptance actually requires (mean Sharpe alone rewards "okay everywhere").
    summary["fitness"] = round(score + float(accept_cfg.get("fitness_pass_weight", 0.4)) * pass_rate, 4)
    summary["accepted"] = bool(
        pass_rate >= float(accept_cfg.get("min_pass_rate", 0.2))
        and pct_pos >= float(accept_cfg.get("min_pct_positive", 0.6))
        and score >= float(accept_cfg.get("min_score", 0.2))
        and tpy >= min_tpy
    )
    return summary


def evaluate_universe(compiled, universe: list[SymbolData], ev: EvalSettings, cfg,
                      probe: list[SymbolData] | None = None, sig_cache: dict | None = None,
                      max_seconds: float = 120.0) -> UniverseResult:
    """Run `compiled` over the universe. If a probe subset is given it is
    evaluated first and a clearly-bad strategy stops there (saves ~85% of the
    work for the typical reject). Aborts with whatever was computed once
    max_seconds is exceeded (a pathologically slow signals())."""
    ucfg = cfg.get("universe", {}) or {}
    accept_cfg = ucfg.get("accept", {}) or {}
    started = time.monotonic()
    sig_cache = sig_cache if sig_cache is not None else {}

    def run(subset: list[SymbolData]) -> list[SymbolResult]:
        out = []
        for sd in subset:
            if time.monotonic() - started > max_seconds:
                logger.warning("Strategy '%s' exceeded %.0fs eval budget at %s; truncating universe",
                               compiled.strategy_name, max_seconds, sd.symbol)
                break
            out.append(evaluate_symbol(compiled, sd, ev, cfg, sigs=sig_cache.get(sd.symbol)))
        return out

    probe_ids = {sd.symbol for sd in probe} if probe else set()
    results: list[SymbolResult] = []
    probe_only = False
    reject_reason = None
    if probe and len(probe) < len(universe):
        results = run(probe)
        ok = [r for r in results if r.error is None]
        if ok:
            med = float(np.median([r.avg_sharpe for r in ok]))
            pos = float(np.mean([r.avg_sharpe > 0 for r in ok]))
            probe_tpy = float(np.mean([r.trades_per_year for r in ok]))
            min_tpy = float(accept_cfg.get("min_trades_per_year", 0.0) or 0.0)
            if min_tpy and probe_tpy < float(ucfg.get("probe_min_tpy_fraction", 0.4)) * min_tpy:
                probe_only, reject_reason = True, f"low_frequency ({probe_tpy:.1f} trades/yr, need >= {min_tpy:.0f})"
            elif med < float(ucfg.get("probe_min_median_sharpe", -0.05)) and pos < float(ucfg.get("probe_min_pct_positive", 0.45)):
                probe_only, reject_reason = True, "unprofitable on probe symbols"
        if not probe_only:
            results += run([sd for sd in universe if sd.symbol not in probe_ids])
    else:
        results = run(universe)

    summary = summarize(results, accept_cfg)
    if probe_only:
        summary["accepted"] = False
        summary["probe_rejected"] = True
        summary["reject_reason"] = reject_reason
    return UniverseResult(per_symbol=results, summary=summary, probe_only=probe_only,
                          seconds=time.monotonic() - started)


def _segment_summary(rows: list[dict]) -> dict:
    sharpes = np.array([r["sharpe"] for r in rows]) if rows else np.array([0.0])
    out = {
        "rows": rows,
        "median_sharpe": round(float(np.median(sharpes)), 4),
        "mean_sharpe": round(float(np.mean(np.clip(sharpes, -1.5, 1.5))), 4),
        "pct_positive": round(float((sharpes > 0).mean()), 3),
    }
    raw = [r["raw_sharpe"] for r in rows if "raw_sharpe" in r]
    if raw:
        out["raw_mean_sharpe"] = round(float(np.mean(np.clip(raw, -1.5, 1.5))), 4)
    for half in ("h1", "h2"):
        vals = [r[f"sharpe_{half}"] for r in rows if f"sharpe_{half}" in r]
        if vals:
            out[f"mean_sharpe_{half}"] = round(float(np.mean(np.clip(vals, -1.5, 1.5))), 4)
    return out


def oss1_verdict(oss1: dict, accept_cfg: dict) -> tuple[bool, str]:
    """The validation gate: does a strategy that cleared the IS screen also hold up on OSS1?"""
    gate = accept_cfg.get("oss1", {}) or {}
    min_pos = float(gate.get("min_pct_positive", 0.0))
    min_sharpe = float(gate.get("min_mean_sharpe", -9.0))
    min_half = gate.get("min_half_mean_sharpe")
    problems = []
    if min_half is not None and "mean_sharpe_h1" in oss1 and "mean_sharpe_h2" in oss1:
        worst = min(oss1["mean_sharpe_h1"], oss1["mean_sharpe_h2"])
        if worst < float(min_half):
            problems.append(f"OSS1 half-period mean Sharpe {worst:+.2f} < {float(min_half):+.2f} (edge lives in one half only)")
    if oss1["pct_positive"] < min_pos:
        problems.append(f"OSS1 profitable on {oss1['pct_positive']:.0%} of symbols < {min_pos:.0%}")
    if oss1["mean_sharpe"] < min_sharpe:
        problems.append(f"OSS1 mean Sharpe {oss1['mean_sharpe']:+.2f} < {min_sharpe:+.2f}")
    return not problems, "; ".join(problems)


def run_universe_oss1(compiled, universe: list[SymbolData], ev: EvalSettings, cfg, sig_cache: dict | None = None) -> dict:
    """Validation segment for every symbol, plus the OSS1 gate verdict (`passed`, `fail`).
    Used by the research loop for strategies that already cleared the IS screen. Its
    numbers must not be fed back into prompts or the tuner's fitness."""
    sig_cache = sig_cache or {}
    out = _segment_summary([segment_symbol(compiled, sd, ev, cfg, sd.oss1, sigs=sig_cache.get(sd.symbol)) for sd in universe])
    out["passed"], out["fail"] = oss1_verdict(out, (cfg.get("universe", {}) or {}).get("accept", {}) or {})
    return out


def run_universe_oss2(compiled, universe: list[SymbolData], ev: EvalSettings, cfg) -> dict:
    """THE TEST SET. Called only by src.final_test (`python main.py final-test`), never by the
    research loop: every look at OSS2 that informs a decision turns it into more validation data."""
    return _segment_summary([segment_symbol(compiled, sd, ev, cfg, sd.oss2) for sd in universe])
