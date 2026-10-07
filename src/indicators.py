"""Technical indicators implemented in pure pandas/NumPy (no TA-Lib).

All indicators use only past data at each timestamp (right-aligned rolling
windows, no centering, no lookahead). Column names are deterministic so the
strategy rule DSL can reference them reliably.
"""
from __future__ import annotations

import pandas as pd

ALLOWED_INDICATORS = {
    "sma", "ema", "rsi", "macd", "atr", "bollinger_bands",
    "donchian", "roc", "stochastic", "adx",
}


def sma(df: pd.DataFrame, column: str = "close", period: int = 20) -> pd.DataFrame:
    name = f"sma_{column}_{period}"
    out = pd.DataFrame(index=df.index)
    out[name] = df[column].rolling(window=period, min_periods=period).mean()
    return out


def ema(df: pd.DataFrame, column: str = "close", period: int = 20) -> pd.DataFrame:
    name = f"ema_{column}_{period}"
    out = pd.DataFrame(index=df.index)
    out[name] = df[column].ewm(span=period, adjust=False, min_periods=period).mean()
    return out


def rsi(df: pd.DataFrame, column: str = "close", period: int = 14) -> pd.DataFrame:
    name = f"rsi_{column}_{period}"
    delta = df[column].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    result = 100 - (100 / (1 + rs))
    result = result.fillna(100.0).where(avg_loss != 0, 100.0)
    out = pd.DataFrame(index=df.index)
    out[name] = result
    return out


def macd(df: pd.DataFrame, column: str = "close", fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    prefix = f"macd_{fast}_{slow}_{signal}"
    ema_fast = df[column].ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_line"] = macd_line
    out[f"{prefix}_signal"] = signal_line
    out[f"{prefix}_hist"] = hist
    out[prefix] = macd_line  # convenience alias
    return out


def atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    name = f"atr_{period}"
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out = pd.DataFrame(index=df.index)
    out[name] = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return out


def bollinger_bands(df: pd.DataFrame, column: str = "close", period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    prefix = f"bb_{period}_{int(num_std) if float(num_std).is_integer() else num_std}"
    mid = df[column].rolling(window=period, min_periods=period).mean()
    std = df[column].rolling(window=period, min_periods=period).std(ddof=0)
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_middle"] = mid
    out[f"{prefix}_upper"] = mid + num_std * std
    out[f"{prefix}_lower"] = mid - num_std * std
    return out


def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian channel over the PRIOR `period` bars (current bar excluded).

    Excluding the current bar is standard for breakout systems: a channel
    that includes today's own high/low would make "close > channel_upper"
    almost unsatisfiable, since close <= high by definition. Shifting by
    one bar keeps this fully non-lookahead (still only uses past data) while
    making breakout conditions meaningful.
    """
    prefix = f"donchian_{period}"
    upper = df["high"].shift(1).rolling(window=period, min_periods=period).max()
    lower = df["low"].shift(1).rolling(window=period, min_periods=period).min()
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_upper"] = upper
    out[f"{prefix}_lower"] = lower
    out[f"{prefix}_middle"] = (upper + lower) / 2.0
    return out


def roc(df: pd.DataFrame, column: str = "close", period: int = 10) -> pd.DataFrame:
    name = f"roc_{column}_{period}"
    out = pd.DataFrame(index=df.index)
    out[name] = df[column].pct_change(periods=period) * 100.0
    return out


def stochastic(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    prefix = f"stoch_{period}_{smooth_k}_{smooth_d}"
    lowest_low = df["low"].rolling(window=period, min_periods=period).min()
    highest_high = df["high"].rolling(window=period, min_periods=period).max()
    denom = (highest_high - lowest_low).replace(0.0, pd.NA)
    raw_k = 100 * (df["close"] - lowest_low) / denom
    k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(window=smooth_d, min_periods=smooth_d).mean()
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_k"] = k
    out[f"{prefix}_d"] = d
    return out


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    name = f"adx_{period}"
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0.0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0.0)

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_smooth = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smooth
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_smooth

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, pd.NA)
    adx_val = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    out = pd.DataFrame(index=df.index)
    out[name] = adx_val
    out[f"plus_di_{period}"] = plus_di
    out[f"minus_di_{period}"] = minus_di
    return out


INDICATOR_FUNCS = {
    "sma": sma,
    "ema": ema,
    "rsi": rsi,
    "macd": macd,
    "atr": atr,
    "bollinger_bands": bollinger_bands,
    "donchian": donchian,
    "roc": roc,
    "stochastic": stochastic,
    "adx": adx,
}


def compute_indicator(df: pd.DataFrame, name: str, params: dict) -> pd.DataFrame:
    """Dispatch to the right indicator function with validated params only."""
    if name not in INDICATOR_FUNCS:
        raise ValueError(f"Unknown indicator: {name}")
    func = INDICATOR_FUNCS[name]
    return func(df, **params)


def apply_indicators(df: pd.DataFrame, indicator_defs: list[dict]) -> pd.DataFrame:
    """Compute all indicators in `indicator_defs` and join them onto df."""
    result = df.copy()
    for spec in indicator_defs:
        name = spec["name"]
        params = dict(spec.get("params", {}))
        if "column" in spec and "column" not in params:
            # sma/ema/rsi/roc take `column`; others ignore it via **kwargs guard below.
            if name in {"sma", "ema", "rsi", "roc", "macd", "bollinger_bands"}:
                params["column"] = spec["column"]
        cols = compute_indicator(df, name, params)
        result = result.join(cols, how="left")
    return result
