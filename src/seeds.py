"""Classic strategy templates, instantiated with random parameters.

Why this exists: LLM requests are the scarcest resource (50/day on the free
tier) while local backtests are nearly free, and raw LLM batches have mean score
below zero. Sampling known-reasonable templates gives every iteration a supply
of candidate parents that cost no tokens, keeps the leaderboard populated when
the quota is exhausted, and gives the local mutation search (src/tuner.py)
something to improve in offline mode. LLM output remains the source of NEW
ideas; templates are the baseline it has to beat.
"""
from __future__ import annotations

import random
import re

from src.strategy_codegen import StrategySpec

# (family, idea, {param: (lo, hi)}, snippet). Int bounds -> int params, float bounds -> float.
TEMPLATES: list[tuple[str, str, dict, str]] = [
    ("breakout", "donchian channel breakout", {"n": (8, 60), "m": (4, 30)},
     'le = close > hh(p["n"])\nse = close < ll(p["n"])\nlx = close < ll(p["m"])\nsx = close > hh(p["m"])'),
    ("meanrev", "rsi dip in trend", {"r": (2, 7), "lo": (15, 38), "t": (50, 200)},
     'x = rsi(close, p["r"])\nle = (close > sma(close, p["t"])) & (x < p["lo"])\nlx = x > 60\nse = (close < sma(close, p["t"])) & (x > 100 - p["lo"])\nsx = x < 40'),
    ("meanrev", "bollinger zscore reversion", {"n": (8, 40), "z": (1.0, 2.5)},
     'z = zscore(close, p["n"])\nle = z < -p["z"]\nlx = z > 0\nse = z > p["z"]\nsx = z < 0'),
    ("trend", "ema cross with adx", {"f": (4, 20), "s": (25, 100), "a": (12, 30)},
     'f = ema(close, p["f"]); s = ema(close, p["s"])\nok = adx(14) > p["a"]\nle = crossover(f, s) & ok\nlx = crossunder(f, s)\nse = crossunder(f, s) & ok\nsx = crossover(f, s)'),
    ("momentum", "time series momentum", {"n": (10, 120)},
     'm = roc(close, p["n"])\nle = m > 0\nlx = m < 0\nse = m < 0\nsx = m > 0'),
    ("volatility", "squeeze breakout", {"n": (10, 25), "r": (0.15, 0.5), "k": (5, 30)},
     'quiet = pct_rank(atr(p["n"]) / close, 250) < p["r"]\nle = quiet & (close > hh(p["k"]))\nse = quiet & (close < ll(p["k"]))\nlx = close < sma(close, p["k"])\nsx = close > sma(close, p["k"])'),
    ("pattern", "gap fade", {"g": (0.2, 1.0), "n": (10, 30)},
     'gap = (open_ - close.shift(1)) / atr(p["n"])\nle = gap < -p["g"]\nse = gap > p["g"]\nlx = close > open_\nsx = close < open_'),
    ("pattern", "consecutive closes reversal", {"k": (2, 5), "t": (30, 150)},
     'le = (down_streak(close) >= p["k"]) & (close > sma(close, p["t"]))\nlx = up_streak(close) >= 2\nse = (up_streak(close) >= p["k"]) & (close < sma(close, p["t"]))\nsx = down_streak(close) >= 2'),
    ("meanrev", "stochastic cross", {"n": (6, 20), "lo": (15, 35)},
     'k = stoch(p["n"])\nle = crossover(k, p["lo"])\nlx = k > 70\nse = crossunder(k, 100 - p["lo"])\nsx = k < 30'),
    ("momentum", "near rolling high", {"n": (60, 250), "d": (0.01, 0.06)},
     'top = close.rolling(int(p["n"]), min_periods=int(p["n"])).max()\nle = close > top * (1 - p["d"])\nlx = close < top * (1 - 2 * p["d"])'),
    ("seasonal", "turn of month", {"d": (1, 4), "e": (2, 6)},
     'dom = pd.Series(df.index.day, index=df.index)\nle = (dom <= p["d"]) | (dom >= 28)\nlx = (dom > p["e"]) & (dom < 27)'),
    ("trend", "efficiency ratio regime", {"n": (8, 30), "e": (0.2, 0.5), "f": (8, 40)},
     'er = eff_ratio(close, p["n"])\ntrendy = er > p["e"]\nm = ema(close, p["f"])\nle = trendy & (close > m)\nlx = close < m\nse = trendy & (close < m)\nsx = close > m'),
    ("meanrev", "internal bar strength reversion", {"lo": (0.1, 0.35), "t": (50, 200)},
     'ibs = clv()\nle = (ibs < p["lo"]) & (close > sma(close, p["t"]))\nlx = ibs > 0.7\nse = (ibs > 1 - p["lo"]) & (close < sma(close, p["t"]))\nsx = ibs < 0.3'),
    ("meanrev", "cumulative rsi dip", {"r": (2, 4), "c": (20, 60), "t": (60, 200)},
     'cum = rsi(close, p["r"]).rolling(2, min_periods=2).sum()\nle = (cum < p["c"]) & (close > sma(close, p["t"]))\nlx = rsi(close, p["r"]) > 65\nse = (cum > 200 - p["c"]) & (close < sma(close, p["t"]))\nsx = rsi(close, p["r"]) < 35'),
    ("breakout", "range expansion follow-through", {"n": (10, 25), "k": (1.0, 2.0)},
     'tr = (high - low)\nbig = tr > p["k"] * atr(p["n"])\nle = big & (close > open_)\nlx = close < ema(close, 5)\nse = big & (close < open_)\nsx = close > ema(close, 5)'),
    ("trend", "macd histogram turn", {"f": (6, 14), "s": (18, 34), "g": (4, 10)},
     'line, sig, hist = macd(close, p["f"], p["s"], p["g"])\nle = crossover(hist, 0)\nlx = crossunder(hist, 0)\nse = crossunder(hist, 0)\nsx = crossover(hist, 0)'),
    ("pattern", "close beyond prior range fade", {"k": (0.2, 0.8), "n": (10, 30)},
     'rng = (high - low).rolling(int(p["n"]), min_periods=int(p["n"])).mean()\nle = close < low.shift(1) - p["k"] * rng\nlx = close > close.shift(1)\nse = close > high.shift(1) + p["k"] * rng\nsx = close < close.shift(1)'),
]

SL_CHOICES = [0.015, 0.02, 0.03, 0.05]
TP_CHOICES = [0.03, 0.06, 0.1, 0.2]


def sample_seed_specs(n: int, rng: random.Random) -> list[StrategySpec]:
    specs = []
    for _ in range(n):
        family, idea, ranges, body = rng.choice(TEMPLATES)
        params: dict = {}
        for k, (lo, hi) in ranges.items():
            params[k] = rng.randint(lo, hi) if isinstance(lo, int) and isinstance(hi, int) else round(rng.uniform(lo, hi), 3)
        name = "seed_" + re.sub(r"\W+", "_", idea) + f"_{rng.randint(1000, 9999)}"
        specs.append(StrategySpec(name=name, family=family, idea=f"seed template: {idea}", sl=rng.choice(SL_CHOICES),
                                  tp=rng.choice(TP_CHOICES), params=params, body=body))
    return specs
