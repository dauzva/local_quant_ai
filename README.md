# local_quant_ai

Automated research loop for daily futures strategies. An LLM (via OpenRouter)
mass-generates indicator-based strategies, each one is backtested on ~50 futures
symbols from `./data` with multi-fold walk-forward validation and an untouched
final holdout, and the best ones are evolved further, both by the LLM and by a
free local search. Strategies that pass the universe-wide acceptance gate are
saved as standalone Python files.

**Research tool only. No live trades are placed. Not financial advice.**

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your key:

```
OPENROUTER_API_KEY=your-key-here
```

Put TradeStation `.dat` files in `./data`, named like
`A_OHLCV_@TY_minutes_1440.dat` (`minutes_1440` = daily bars).

## Usage

```bash
python main.py inspect-data                      # list discovered .dat files
python main.py list-free-models                  # currently-free OpenRouter models, ranked

python main.py run                               # config.yaml defaults (10 iterations)
python main.py run --iterations 0 --minutes 120  # run for 2 hours
python main.py run --symbols "@TY,@ES,@GC"       # restrict the universe
python main.py tune --minutes 30                 # offline local search, no LLM

# Re-run one saved strategy across the universe
python main.py backtest-strategy --file results/<run_id>/strategies/<name>/strat_gen_<name>.py

python -m pytest tests -q                        # offline tests, no API key needed
```

## How it works

1. **Universe**: loads every symbol in `./data`. Unusable series (e.g.
   back-adjusted contracts with negative prices), near-duplicates and short
   histories are dropped (see `universe:` in `config.yaml`; ~51 symbols remain).
2. **Generate**: each iteration sends several concurrent LLM requests. *Explore*
   requests ask for ~10 fresh strategies seeded with random themes. *Evolve*
   requests send the best strategies back with their code, per-asset-class
   results and a rule-based diagnosis, and ask for improvements.
3. **Validate**: every strategy passes the sandbox (see "Safety") and is
   deduplicated by structural and behavioral fingerprint before any backtest.
4. **Evaluate**: a numba backtester runs 5 walk-forward folds per symbol; a
   12-symbol probe stops clearly bad strategies early. The universe **score** is
   the mean per-symbol fold Sharpe, scaled down for low trade frequency.
5. **Accept**: strategies that clear `universe.accept` (breadth across markets,
   not one lucky chart) get a one-shot holdout evaluation, which is never fed
   back, and are saved as `strat_gen_<name>.py`, `report.md`, `per_symbol.csv/png`
   and `metadata.json`.
6. **Remember**: `results/memory/` stores the leaderboard, fingerprints,
   per-family hit rates and common errors. State is checkpointed after every
   LLM response, so Ctrl-C is safe.

Outputs land in `results/<run_id>/` (`all_experiments.csv`, `top_strategies.csv`,
`llm_usage.json`, `strategies/`). At the end of each run the top
`run.save_top_n` strategies are written out whether or not they were accepted.

### Trade frequency

A strategy that trades 2-5 times a year is a handful of lucky positions. Every
walk-forward fold needs a minimum trade count (default 10 trades/year), the
score is multiplied by `min(1, trades_per_year / 10)`, and the prompts tell the
LLM how to meet the requirement.

### Learning

LLM requests are scarce (50/day on the free tier), so most learning is local
and token-free:

- **Local evolutionary search** (`src/tuner.py`, `src/mutation.py`): after each
  iteration the best strategies are mutated (re-tuned parameters, new
  stop/target, drop or invert a leg, trend/volatility/ADX filters, time stop...)
  and evaluated in parallel on the whole universe. A mutant is promoted if it
  beats its parent.
- **Evidence-based operators**: mutation choice is weighted by each operator's
  historical success rate and by the parent's diagnosis.
- **Seed templates** (`src/seeds.py`): ~17 classic strategies with random
  parameters give the search parents even when the LLM is unavailable.
- **LLM feedback**: prompts include winners, family hit rates, recent failures,
  and the measured effect of earlier changes.
- **Offline fallback**: if the quota or key dies, the run continues with seeds
  and local search (`run.offline_fallback`).

## Strategy format

The LLM writes only a short snippet per strategy:

````
===S===
name: rsi_dip_in_uptrend
family: meanrev
idea: buy short-term oversold dips while above the long trend
sl: 0.02
tp: 0.04
params: {"rsi_n": 5, "lo": 25, "trend_n": 100}
```python
r = rsi(close, p["rsi_n"])
le = (close > sma(close, p["trend_n"])) & (r < p["lo"])
lx = r > 60
```
````

The snippet sees `df, p, open_, high, low, close, volume` plus vectorised,
lookahead-free helpers (`sma ema stdev zscore roc rsi atr adx macd stoch hh ll
clv crossover crossunder pct_rank eff_ratio up_streak down_streak`).

`src/strategy_codegen.build_source` wraps it into a self-contained Python file
(only `numpy`/`pandas`/`typing` imports) with one class exposing
`default_params()`, `validate(p)` and
`signals(df, p) -> (long_entry, long_exit, short_entry, short_exit)`.

## Safety

LLM output is executable Python, so `src/strategy_sandbox.py` checks every
strategy before it touches real data:

1. **Static AST allowlist**: only `typing`, `numpy`, `pandas`, `math` imports;
   one class; no `eval`/`exec`/`open`/`__import__`/`getattr` etc.; no dunder
   attribute access.
2. **Restricted `exec`** with a small safe-builtins dict.
3. **Smoke test** on synthetic data in a worker thread with an 8s timeout.
4. **Lookahead test**: signals on the first 75% of the bars must equal the same
   bars' signals computed on all of them. This catches `shift(-k)`, centered
   rolling windows and whole-series normalisation, which would otherwise
   dominate the winners.

Invalid strategies are logged, counted, and the most common errors are fed back
into later prompts. This is defense-in-depth for a local single-user tool, not
a multi-tenant security boundary. The LLM never computes metrics; those are all
local (`src/metrics.py`).

## LLM models and quota

`list-free-models` ranks OpenRouter's live free models for code generation
(large context, code-capable names first; reasoning-branded models like
nemotron are pushed down because they tend to burn their token budget on
thinking). `llm.models` in `config.yaml` is tried first, then, with
`llm.auto_discover_free_models: true`, the live ranked list as fallback.

**Daily quota:** an OpenRouter account with no credit is limited to **50
free-model requests per day, account-wide**; $10 of one-time credit raises that
to 1000/day (the `:free` models still cost $0). When the quota is hit the run
stops cleanly with progress saved. Re-run after the 24h reset.

**Requests are the scarce resource, not tokens**, so each request asks for
10-12 strategies (`llm.strategies_per_call`), about 500-600 strategies/day on a
free account. For unlimited volume, put a cheap paid model first in
`llm.models` and set `use_only_free_models: false`. `llm.max_cost_usd` caps
spend; `run.iterations: 0` with `--minutes` runs until time, quota or budget
runs out.

## Configuration

Everything lives in `config.yaml`: data paths, instrument metadata, backtest
costs and sizing, walk-forward and acceptance thresholds, universe filters,
LLM settings and the memory path. `ASSUMPTIONS.md` lists defaults chosen where
the spec was ambiguous.
