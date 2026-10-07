import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src import memory  # noqa: E402
from src.backtester import build_settings, run_backtest, simulate_arrays  # noqa: E402
from src.config import load_config  # noqa: E402
from src.llm import AuthError, LLMError, OpenRouterClient, RateLimitError  # noqa: E402
from src.metrics import compute_metrics, fast_metrics  # noqa: E402
from src.strategy_codegen import build_source, parse_specs  # noqa: E402
from src.strategy_sandbox import StrategySandboxError, compile_strategy  # noqa: E402

GOOD = """===S===
name: Fast Slow Cross!
family: trend
idea: fast over slow
sl: 0.02
tp: none
params: {"f": 8, "s": 40}
```python
f = ema(close, p["f"]); s = ema(close, p["s"])
le = crossover(f, s)
lx = crossunder(f, s)
```
===S===
name: truncated_one
family: trend
idea: cut off
params: {"n": 5}
```python
le = close > sma(close, p["n"])
"""


@pytest.fixture(scope="module")
def cfg():
    c = load_config(str(ROOT / "config.yaml"))
    c.data.directory = str(ROOT / "data")
    return c


def _ohlc(n=500, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="B", name="datetime")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0}, index=idx)


# ---------------------------------------------------------------- backtester

def test_fast_metrics_match_compute_metrics(cfg):
    df = _ohlc()
    rng = np.random.default_rng(0)
    for col in ("entry_signal", "exit_signal", "short_entry_signal", "short_exit_signal"):
        df[col] = rng.random(len(df)) < 0.05
    strat = SimpleNamespace(strategy_name="t", stop_loss_pct=0.02, take_profit_pct=0.05, risk_per_trade=0.01)
    res = run_backtest(df, strat, cfg, "X", instrument={"point_value": 50.0, "tick_size": 0.25})
    full = compute_metrics(res.equity_curve["equity"], res.trades, 252)
    s = build_settings(cfg, {"point_value": 50.0, "tick_size": 0.25}, strat)
    eq, nt, *_rest = simulate_arrays(
        df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(),
        df["entry_signal"].to_numpy(), df["exit_signal"].to_numpy(),
        df["short_entry_signal"].to_numpy(), df["short_exit_signal"].to_numpy(), s)
    pnl = _rest[6][:nt]
    fast = fast_metrics(eq, pnl, 252)
    for k in ("sharpe", "max_drawdown", "profit_factor", "num_trades", "total_return"):
        assert fast[k] == pytest.approx(full[k], rel=1e-9, abs=1e-12), k


def test_signals_fill_at_next_open_and_stop_beats_target(cfg):
    df = _ohlc(60)
    df["entry_signal"] = False
    df["exit_signal"] = False
    df["short_entry_signal"] = False
    df["short_exit_signal"] = False
    df.iloc[5, df.columns.get_loc("entry_signal")] = True
    strat = SimpleNamespace(strategy_name="t", stop_loss_pct=0.0001, take_profit_pct=0.0001, risk_per_trade=0.01)
    res = run_backtest(df, strat, cfg, "X", instrument={"point_value": 50.0, "tick_size": 0.25})
    t = res.trades.iloc[0]
    assert t["entry_time"] == df.index[6]                      # signal on bar 5 -> fill on bar 6
    assert t["entry_price"] == pytest.approx(df["open"].iloc[6])
    assert t["exit_reason"] == "stop_loss"                      # both touched in one bar -> stop wins


# ------------------------------------------------------------------- codegen

def test_parse_specs_flags_truncated_block():
    specs = parse_specs(GOOD)
    assert [s.name for s in specs] == ["fast_slow_cross", "truncated_one"]
    assert specs[0].problems == [] and specs[0].sl == 0.02 and specs[0].params == {"f": 8, "s": 40}
    assert specs[1].problems and "closed" in specs[1].problems[0]


def test_build_source_compiles_and_is_standalone():
    spec = parse_specs(GOOD)[0]
    src = build_source(spec)
    assert "def ema(" in src and "def crossover(" in src and "def sma(" not in src   # only used helpers inlined
    comp = compile_strategy(src)
    le, lx, se, sx = comp.signals(_ohlc())
    assert le.dtype == bool and len(le) == 500 and not se.any()


def test_snippet_with_forbidden_name_is_rejected():
    raw = GOOD.split("===S===")[1].replace("le = crossover(f, s)", "le = getattr(close, 'x')")
    with pytest.raises(StrategySandboxError):
        compile_strategy(build_source(parse_specs("===S===" + raw)[0]))


def test_numpy_internal_imports_do_not_trip_sandbox():
    raw = GOOD.split("===S===")[1].replace("le = crossover(f, s)", "le = close.rolling(20).std() > np.std(close.to_numpy())")
    compile_strategy(build_source(parse_specs("===S===" + raw)[0]))


@pytest.mark.parametrize("bad_line", [
    "le = close.shift(-1) > close",
    "le = close > close.rolling(20, center=True).mean()",
    "le = close / close.max() > 0.9",
    "le = close.rank(pct=True) > 0.8",
])
def test_lookahead_is_detected(bad_line):
    raw = GOOD.split("===S===")[1].replace("le = crossover(f, s)", bad_line)
    with pytest.raises(StrategySandboxError) as exc:
        compile_strategy(build_source(parse_specs("===S===" + raw)[0]))
    assert "LOOKAHEAD" in exc.value.problems[0]


# -------------------------------------------------------------------- memory

def test_leaderboard_views_are_distinct_per_code_body():
    state = memory._empty_state()
    for name, score in [("a", 0.5), ("a_t1", 0.4), ("b", 0.3)]:
        spec = {"name": name, "family": "trend", "idea": "i", "sl": 0.02, "tp": 0.04, "params": {}, "body": "le=close>sma(close,5)" if name != "b" else "le=close<1", "parent": None}
        memory.record_result(state, spec=spec, summary={"score": score}, run_id="r", iteration=1, signature=name, behavior=name, accepted=False)
    assert [e["name"] for e in memory.best_ever(state, 5)] == ["a", "b"]      # tuned copy of the same code is not a second slot
    assert [e["name"] for e in memory.pick_parents(state, 5)] == ["a", "b"]


def test_memory_names_leaderboard_and_parents(tmp_path):
    state = memory._empty_state()
    spec = {"name": "alpha", "family": "trend", "idea": "i", "sl": 0.02, "tp": 0.04, "params": {}, "body": "le=close>0", "parent": None}
    assert memory.unique_name(state, "alpha") == "alpha"
    for name, score in [("alpha", 0.3), ("beta", 0.5), ("gamma", -0.1)]:
        memory.record_result(state, spec={**spec, "name": name, "body": f"le=close>{score}"}, summary={"score": score, "n_pass": 1}, run_id="r", iteration=1,
                             signature=name, behavior=name, accepted=False)
    assert memory.unique_name(state, "alpha") == "alpha_v2"
    assert [e["name"] for e in memory.best_ever(state, 5)] == ["beta", "alpha"]     # negative scores never enter the board
    for _ in range(6):
        memory.mark_evolved(state, ["beta"])
    assert memory.pick_parents(state, 1)[0]["name"] == "alpha"                      # over-evolved champion is rotated out
    path = tmp_path / "s.json"
    memory.save_memory(state, path)
    assert memory.load_memory(path)["total_tried"] == 3
    path.write_text('{"version": 1}')
    assert memory.load_memory(path)["version"] == memory.STATE_VERSION
    assert (tmp_path / "state_v1_backup.json").exists()                              # old-format state is archived, not lost


# ----------------------------------------------------------------------- llm

def _client(cfg, models):
    c = OpenRouterClient(cfg)
    c._resolved_models = list(models)
    c.retry_wait = 0
    c.max_cooldown_wait = 0
    return c


def test_llm_rate_limited_model_cools_down_and_next_is_used(cfg):
    c = _client(cfg, ["a:free", "b:free"])
    calls = []

    def fake(model, prompt, max_tokens, temperature):
        calls.append(model)
        if model == "a:free":
            raise RateLimitError("429")
        return "ok"

    c._call_model_once = fake
    assert c.complete_text("x").model_used == "b:free"
    assert c.complete_text("x").model_used == "b:free"
    assert calls == ["a:free", "b:free", "b:free"]            # 'a' was NOT retried on the second call


def test_llm_auth_error_is_fatal_and_empty_replies_fail_over(cfg):
    c = _client(cfg, ["a:free", "b:free"])

    def auth(model, *a):
        raise AuthError("bad key")

    c._call_model_once = auth
    with pytest.raises(AuthError):
        c.complete_text("x")

    c2 = _client(cfg, ["a:free", "b:free"])

    def flaky(model, *a):
        if model == "a:free":
            raise LLMError("returned empty content")
        return "fine"

    c2._call_model_once = flaky
    assert c2.complete_text("x").text == "fine"


# --------------------------------------------------------------- end to end

def test_full_loop_with_fake_llm(cfg, tmp_path, monkeypatch):
    import src.orchestrator as orch
    from fake_llm import FakeClient

    monkeypatch.chdir(tmp_path)
    cfg.memory.path = str(tmp_path / "memory" / "state.json")
    cfg.run.explore_calls_per_iteration = 1
    cfg.run.evolve_calls_per_iteration = 1
    cfg.run.tune_seconds = 4
    cfg.run.tune_processes = False   # threads in tests: spawning workers inside pytest is slow/fragile on Windows
    cfg.run.seed_samples_per_iteration = 4
    cfg.run.save_top_n = 3
    cfg.universe.max_symbols = 12
    cfg.universe.accept = {"min_pass_rate": 0.0, "min_pct_positive": 0.0, "min_score": -1.0}   # force the accept path
    monkeypatch.setattr(orch, "OpenRouterClient", FakeClient)
    run_id = orch.run_research_loop(cfg, iterations=2)
    run_dir = tmp_path / "results" / run_id
    df = pd.read_csv(run_dir / "all_experiments.csv")
    assert {"accepted", "invalid"} <= set(df.status)
    state = memory.load_memory(cfg.memory.path)
    assert state["total_tried"] > 5 and state["leaderboard"]
    accepted = next((run_dir / "strategies").iterdir())
    assert (next(accepted.glob("strat_gen_*.py"))).exists() and (accepted / "per_symbol.csv").exists()
    compile_strategy((next(accepted.glob("strat_gen_*.py"))).read_text(encoding="utf-8"))   # saved file is a valid standalone strategy
    assert (tmp_path / "memory" / "experiments.jsonl").exists()


# ------------------------------------------ regressions from real LLM output

def test_bare_triple_equals_delimiters_are_accepted():
    raw = "===\nname: a\nparams: {}\n```python\nle = close > 1\n```\n\n=== STRATEGY 2 ===\nname: b\nparams: {}\n```python\nle = close > 2\n```\n===S===\nname: c\nparams: {}\n```python\nle = close > 3\n```"
    assert [s.name for s in parse_specs(raw)] == ["a", "b", "c"]


def test_helper_can_be_rebound_and_precedence_bug_is_repaired():
    raw = ('===S===\nname: sh\nfamily: trend\nsl: 0.02\ntp: 0.04\nparams: {"n": 14, "t": 20}\n```python\n'
           'adx = adx(p["n"])\nle = adx > p["t"] & (close > sma(close, 20))\nlx = close < sma(close, 20) | (adx < 10)\n```')
    comp = compile_strategy(build_source(parse_specs(raw)[0]))      # `adx = adx(..)` and `a > b & c` both used to crash
    le, lx, _se, _sx = comp.signals(_ohlc())
    assert le.dtype == bool and le.any()


def test_friendly_problem_maps_raw_errors_to_instructions():
    from src.strategy_codegen import friendly_problem
    assert "never assigned `le`" in friendly_problem("Smoke test raised: NameError(\"cannot access free variable 'le' where it is not associated\")")
    assert "parentheses" in friendly_problem("Smoke test raised: TypeError(\"Cannot perform 'ror_' with a dtyped [float64] array\")")


def test_low_trade_frequency_fails_the_per_symbol_criteria(cfg):
    from src import universe as U
    cfg.universe.max_symbols = 3
    uni = U.load_universe(cfg)
    ev = U.build_eval_settings(cfg)
    rare = "===S===\nname: rare\nfamily: trend\nsl: 0.02\ntp: 0.04\nparams: {\"n\": 250}\n```python\nle = close > hh(p[\"n\"])\nlx = close < ll(20)\n```"
    comp = compile_strategy(build_source(parse_specs(rare)[0]))
    res = U.evaluate_symbol(comp, uni[0], ev, cfg)
    assert res.trades_per_year < ev.min_trades_per_year and "low_trades" in res.fail_codes and not res.passed
    summary = U.summarize([res], {"min_trades_per_year": 10})
    assert summary["freq_factor"] < 1.0 and not summary["accepted"]


# ----------------------------------------------- mutation / tuner / learning

def test_every_mutation_operator_yields_a_valid_lookahead_free_strategy():
    import random
    from src.mutation import ALL_OPS, mutate
    from src.seeds import sample_seed_specs
    rng = random.Random(5)
    applied = 0
    for spec in sample_seed_specs(12, rng):
        for op in ALL_OPS:
            m = mutate(spec, op, rng)
            if m is None:
                continue
            compile_strategy(build_source(m))      # sandbox incl. dynamic lookahead test
            assert m.parent == spec.name and m.mutation == op
            applied += 1
    assert applied > 100


def test_mutation_stacking_is_blocked_and_diagnosis_steers_operator_choice():
    import random
    from src.mutation import choose_op, diagnosis_boosts, mutate
    from src.seeds import sample_seed_specs
    rng = random.Random(1)
    spec = sample_seed_specs(1, rng)[0]
    once = mutate(spec, "trend_filter", rng)
    assert mutate(once, "trend_filter", rng) is None                      # same operator never stacks twice
    boosts = diagnosis_boosts({"long_pnl_pct": 2.0, "short_pnl_pct": -2.0, "gross_pnl_pct": 1.0, "cost_pct": 0.1,
                               "pct_positive": 0.6, "pass_rate": 0.3, "trades_per_year": 20})
    picks = [choose_op(random.Random(i), {}, boosts) for i in range(300)]
    assert picks.count("drop_shorts") > picks.count("drop_longs") * 2    # losing short leg -> drop it
    stats = {"invert": {"tried": 40, "wins": 0}, "perturb": {"tried": 40, "wins": 30}}
    learned = [choose_op(random.Random(i), stats, {}) for i in range(300)]
    assert learned.count("perturb") > learned.count("invert") * 3        # measured success shapes sampling


def test_evolution_log_feeds_the_learning_digest():
    state = memory._empty_state()
    for op, win in [("adx_trend", True)] * 4 + [("invert", False)] * 4:
        memory.record_mutation(state, op, 1, 1 if win else 0, 0.03 if win else 0.0)
    memory.record_evolution(state, "p", "p_g1", "adx_trend", "x", 0.03, 2.0)
    memory.record_evolution(state, "p_g1", "p_g2", "llm", "add atr gate to cut chop", 0.05, -1.0)
    memory.record_evolution(state, "q", "q2", "llm", "tighter threshold", -0.04, 5.0)
    from src.mutation import OP_LABELS
    table = memory.mutation_table(state, OP_LABELS)
    assert table.index("ADX") < table.index("invert") and "4/4" in table
    good, bad = memory.llm_change_examples(state)
    assert "atr gate" in good[0] and "tighter" in bad[0]
    assert memory.lineage(state, "p_g2").startswith("LLM") and "adx_trend" in memory.lineage(state, "p_g2")


def test_offline_tune_runs_without_llm_and_saves_top_strategies(cfg, tmp_path, monkeypatch):
    import src.orchestrator as orch
    monkeypatch.chdir(tmp_path)
    cfg.memory.path = str(tmp_path / "memory" / "state.json")
    cfg.universe.max_symbols = 12
    cfg.run.tune_seconds = 4
    cfg.run.tune_processes = False   # threads in tests: spawning workers inside pytest is slow/fragile on Windows
    cfg.run.seed_samples_per_iteration = 4
    cfg.run.save_top_n = 3
    cfg.universe.accept = {"min_pass_rate": 0.0, "min_pct_positive": 0.0, "min_score": -1.0}

    class Boom:                                   # any LLM call in offline mode is a bug
        def __init__(self, *a, **k):
            from src.llm import UsageTracker
            self.usage = UsageTracker()

        def total_cost(self):
            return 0.0

        def complete_text(self, *a, **k):
            raise AssertionError("LLM called in offline mode")

    monkeypatch.setattr(orch, "OpenRouterClient", Boom)
    run_id = orch.run_research_loop(cfg, iterations=1, offline=True)
    run_dir = tmp_path / "results" / run_id
    top = pd.read_csv(run_dir / "top_strategies.csv")
    assert 1 <= len(top) <= 3
    for name in top["name"]:
        assert (run_dir / "strategies" / name / f"strat_gen_{name}.py").exists()
        assert (run_dir / "strategies" / name / "report.md").exists()
    state = memory.load_memory(cfg.memory.path)
    assert state["total_tried"] > 5 and state["mutation_stats"]       # the search ran and recorded operator outcomes
