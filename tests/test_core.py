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


# ------------------------------------------------ IS / OSS1 / OSS2 data split

_SIMPLE = ("===S===\nname: simple_cross\nfamily: trend\nsl: 0.02\ntp: 0.04\nparams: {\"f\": 5, \"s\": 20}\n```python\n"
           "le = crossover(ema(close, p[\"f\"]), ema(close, p[\"s\"]))\nlx = crossunder(ema(close, p[\"f\"]), ema(close, p[\"s\"]))\n"
           "se = lx\nsx = le\n```")


def test_split_is_ordered_disjoint_and_uses_shared_calendar_dates(cfg):
    from src import universe as U
    cfg.universe.max_symbols = 8
    uni = U.load_universe(cfg)
    assert uni
    for sd in uni:
        folds = sd.bounds
        assert len(folds) == int(cfg.validation.num_folds)
        assert folds[0][0] == 0 and all(folds[i][1] == folds[i + 1][0] for i in range(len(folds) - 1))   # IS folds tile IS exactly
        assert folds[-1][1] == sd.oss1[0] and sd.oss1[1] == sd.oss2[0] and sd.oss2[1] == sd.n             # IS | OSS1 | OSS2 tile the series
        assert sd.df.index[sd.oss1[0] - 1] <= pd.Timestamp(cfg.split.is_end) < sd.df.index[sd.oss1[0]]
        assert sd.df.index[sd.oss2[0] - 1] <= pd.Timestamp(cfg.split.oss1_end) < sd.df.index[sd.oss2[0]]


def test_split_falls_back_to_fractions_when_dates_are_unset():
    from src import universe as U
    idx = pd.date_range("2000-01-03", periods=1000, freq="B")
    assert U.segment_edges(idx, {"is_end": None, "oss1_end": None, "is_frac": 0.6, "oss1_frac": 0.2}) == (600, 800)
    with pytest.raises(ValueError):
        U.segment_edges(idx, {"is_frac": 0.8, "oss1_frac": 0.3})
    with pytest.raises(ValueError):
        U.segment_edges(idx, {"is_end": "2002-01-01", "oss1_end": "2001-01-01"})


def test_is_evaluation_cannot_see_oss_data(cfg):
    """Scrambling every bar after the IS cutoff must not change a single IS number."""
    import dataclasses
    from src import universe as U
    cfg.universe.max_symbols = 2
    ev = U.build_eval_settings(cfg)
    comp = compile_strategy(build_source(parse_specs(_SIMPLE)[0]))
    for sd in U.load_universe(cfg):
        cut = sd.oss1[0]
        df2 = sd.df.copy()
        rng = np.random.default_rng(0)
        for col in ("open", "high", "low", "close"):
            df2.iloc[cut:, df2.columns.get_loc(col)] = rng.uniform(1.0, 1000.0, len(df2) - cut)
        sd2 = dataclasses.replace(sd, df=df2, open_=df2["open"].to_numpy(float), high=df2["high"].to_numpy(float),
                                  low=df2["low"].to_numpy(float), close=df2["close"].to_numpy(float))
        a, b = U.evaluate_symbol(comp, sd, ev, cfg), U.evaluate_symbol(comp, sd2, ev, cfg)
        assert a.fold_sharpes == b.fold_sharpes and a.total_trades == b.total_trades and a.total_trades > 0


def test_oss1_gate_verdict():
    from src import universe as U
    acc = {"oss1": {"min_pct_positive": 0.55, "min_mean_sharpe": 0.1}}
    assert U.oss1_verdict({"pct_positive": 0.6, "mean_sharpe": 0.2}, acc) == (True, "")
    ok, why = U.oss1_verdict({"pct_positive": 0.4, "mean_sharpe": 0.0}, acc)
    assert not ok and "OSS1 profitable" in why and "mean Sharpe" in why
    assert U.oss1_verdict({"pct_positive": 0.0, "mean_sharpe": -3.0}, {})[0]        # no gate configured -> always passes


def test_research_loop_never_reads_oss2_but_does_use_oss1(cfg, tmp_path, monkeypatch):
    import src.orchestrator as orch
    from src import universe as U
    from fake_llm import FakeClient

    monkeypatch.chdir(tmp_path)
    cfg.memory.path = str(tmp_path / "memory" / "state.json")
    cfg.run.explore_calls_per_iteration = 1
    cfg.run.evolve_calls_per_iteration = 1
    cfg.run.tune_seconds = 3
    cfg.run.tune_processes = False
    cfg.run.seed_samples_per_iteration = 4
    cfg.run.save_top_n = 3
    cfg.universe.max_symbols = 8
    cfg.universe.accept = {"min_pass_rate": 0.0, "min_pct_positive": 0.0, "min_score": -1.0,
                           "oss1": {"min_pct_positive": 0.0, "min_mean_sharpe": -9.0}}
    seen: set = set()
    real = U.segment_symbol

    def spy(compiled, sd, ev, cfg_, segment, sigs=None):
        assert segment != sd.oss2, "the research loop evaluated the test set (OSS2)"
        seen.add("oss1" if segment == sd.oss1 else "other")
        return real(compiled, sd, ev, cfg_, segment, sigs)

    def forbidden(*a, **k):
        raise AssertionError("run_universe_oss2 called from the research loop")

    monkeypatch.setattr(U, "segment_symbol", spy)
    monkeypatch.setattr(U, "run_universe_oss2", forbidden)
    monkeypatch.setattr(orch, "OpenRouterClient", FakeClient)
    run_id = orch.run_research_loop(cfg, iterations=1)
    assert "oss1" in seen and seen == {"oss1"}
    top = pd.read_csv(tmp_path / "results" / run_id / "top_strategies.csv")
    assert {"oss1_mean_sharpe", "oss1_pct_positive"} <= set(top.columns)
    assert top["oss1_mean_sharpe"].is_monotonic_decreasing                # saved top-N is ranked by validation, not by IS
    first = next((tmp_path / "results" / run_id / "strategies").iterdir())
    assert "oss1_sharpe" in pd.read_csv(first / "per_symbol.csv").columns
    assert "OSS2 (test segment)" in (first / "report.md").read_text(encoding="utf-8")


def test_oss1_failure_blocks_acceptance(cfg, tmp_path, monkeypatch):
    import src.orchestrator as orch
    from fake_llm import FakeClient

    monkeypatch.chdir(tmp_path)
    cfg.memory.path = str(tmp_path / "memory" / "state.json")
    cfg.run.explore_calls_per_iteration = 1
    cfg.run.evolve_calls_per_iteration = 0
    cfg.run.tune_seconds = 0
    cfg.run.seed_samples_per_iteration = 4
    cfg.universe.max_symbols = 8
    cfg.universe.accept = {"min_pass_rate": 0.0, "min_pct_positive": 0.0, "min_score": -1.0,
                           "oss1": {"min_pct_positive": 1.1, "min_mean_sharpe": 99.0}}      # impossible validation bar
    monkeypatch.setattr(orch, "OpenRouterClient", FakeClient)
    run_id = orch.run_research_loop(cfg, iterations=1)
    df = pd.read_csv(tmp_path / "results" / run_id / "all_experiments.csv")
    assert "accepted" not in set(df.status) and "oss1_failed" in set(df.status)
    assert df.loc[df.status == "oss1_failed", "reasons"].str.contains("OSS1").all()
    assert not list((tmp_path / "results" / run_id / "strategies").glob("*/metadata.json")) or all(
        '"accepted": false' in m.read_text(encoding="utf-8") for m in (tmp_path / "results" / run_id / "strategies").glob("*/metadata.json"))


def test_final_test_requires_confirm_and_logs_every_look(cfg, tmp_path, monkeypatch):
    from src.final_test import ledger_path, read_ledger, run_final_test
    monkeypatch.chdir(tmp_path)
    cfg.memory.path = str(tmp_path / "memory" / "state.json")
    cfg.universe.max_symbols = 3
    src_file = tmp_path / "strat_gen_simple_cross.py"
    src_file.write_text(build_source(parse_specs(_SIMPLE)[0]), encoding="utf-8")

    assert run_final_test(str(src_file), cfg) is None                       # no --confirm: OSS2 stays unread
    assert read_ledger(ledger_path(cfg)) == []

    first = run_final_test(str(src_file), cfg, confirm=True)
    assert first and {"oss2_mean_sharpe", "oss2_pct_positive", "oss1_mean_sharpe"} <= set(first)
    run_final_test(str(src_file), cfg, confirm=True)
    ledger = read_ledger(ledger_path(cfg))
    assert len(ledger) == 2 and ledger[0]["signature"] == ledger[1]["signature"]
    assert ledger[0]["oss2_mean_sharpe"] == ledger[1]["oss2_mean_sharpe"]    # deterministic: a repeat look adds no information


# ------------------------------------------- generations: validate -> adjust -> final test

def _board_entry(name, fitness, oss1=None, freq=1.0):
    e = {"name": name, "fitness": fitness, "score": fitness, "body": f"le = close > {name}", "summary": {"freq_factor": freq}}
    if oss1 is not None:
        e["oss1"] = {"mean_sharpe": oss1, "median_sharpe": oss1, "pct_positive": 0.6, "passed": True}
    return e


def test_validation_reranks_leaderboard_and_parents():
    state = memory._empty_state()
    state["leaderboard"] = [_board_entry("overfit", 0.50), _board_entry("robust", 0.40), _board_entry("meh", 0.20)]
    assert [e["name"] for e in memory.unvalidated(state, 2)] == ["overfit", "robust"]       # best IS fitness first
    assert memory.attach_oss1(state, "overfit", {"mean_sharpe": -0.30, "median_sharpe": -0.2, "pct_positive": 0.3, "passed": False})
    assert memory.attach_oss1(state, "robust", {"mean_sharpe": 0.35, "median_sharpe": 0.3, "pct_positive": 0.7, "passed": True})
    assert not memory.attach_oss1(state, "ghost", {"mean_sharpe": 1.0})
    assert [e["name"] for e in state["leaderboard"]] == ["robust", "meh", "overfit"]         # 0.375 > 0.20 (unvalidated) > 0.10
    assert [e["name"] for e in memory.pick_parents(state, 1)] == ["robust"]                  # the one that generalised becomes the parent
    assert memory.rank_value(state["leaderboard"][0]) == 0.40                                # the tuner's IS fitness is untouched
    assert [e["name"] for e in memory.unvalidated(state, 5)] == ["meh"]


def test_llm_sees_how_its_parents_did_out_of_sample():
    from src import insights
    s = {"n_eval": 40, "score": 0.30, "trades_per_year": 20, "long_pnl_pct": 1, "short_pnl_pct": 1, "pct_positive": 0.7,
         "pass_rate": 0.3, "group_mean_sharpe": {}, "fail_counts": {}, "no_trade_symbols": 0}
    bad = {"mean_sharpe": -0.2, "pct_positive": 0.3}
    assert "FAILS" in insights.validation_line(bad, s) and insights.diagnose(s, 10, oss1=bad).startswith("OVERFIT")
    assert "holds up" in insights.validation_line({"mean_sharpe": 0.25, "pct_positive": 0.7}, s)
    assert "degrades" in insights.validation_line({"mean_sharpe": 0.05, "pct_positive": 0.7}, s)
    assert insights.validation_line(None, s) == "" and not insights.diagnose(s, 10).startswith("OVERFIT")


def test_pipeline_runs_generations_then_tests_oss2_once(cfg, tmp_path, monkeypatch):
    import json
    import src.orchestrator as orch
    from src.final_test import ledger_path, read_ledger
    from fake_llm import FakeClient

    monkeypatch.chdir(tmp_path)
    cfg.memory.path = str(tmp_path / "memory" / "state.json")
    cfg.run.explore_calls_per_iteration = 1
    cfg.run.evolve_calls_per_iteration = 1
    cfg.run.tune_seconds = 3
    cfg.run.tune_processes = False
    cfg.run.seed_samples_per_iteration = 4
    cfg.run.save_top_n = 3
    cfg.run.validate_top_k = 6
    cfg.universe.max_symbols = 8
    cfg.universe.accept = {"min_pass_rate": 0.0, "min_pct_positive": 0.0, "min_score": -1.0,
                           "oss1": {"min_pct_positive": 0.0, "min_mean_sharpe": -9.0}}
    monkeypatch.setattr(orch, "OpenRouterClient", FakeClient)
    run_id = orch.run_research_loop(cfg, iterations=2, final_test=True)
    run_dir = tmp_path / "results" / run_id

    state = memory.load_memory(cfg.memory.path)
    assert any(e.get("oss1") for e in state["leaderboard"])                      # generations validated their leaders on OSS1

    res = pd.read_csv(run_dir / "oss2_results.csv")
    top = pd.read_csv(run_dir / "top_strategies.csv")
    assert 1 <= len(res) <= 3 and list(res["name"]) == list(top["name"])
    assert {"oss2_mean_sharpe", "oss2_pct_positive", "verdict"} <= set(top.columns)
    assert len(read_ledger(ledger_path(cfg))) == len(res)                       # exactly one logged look per strategy
    for d in res["dir"]:
        d = Path(d)
        assert "oss2_sharpe" in pd.read_csv(d / "per_symbol.csv").columns
        assert "Not evaluated" not in (d / "report.md").read_text(encoding="utf-8")
        assert "oss2" in json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        assert next(d.glob("strat_gen_*.py")).exists()


def test_pipeline_command_skips_oss2_with_flag_and_is_registered(cfg, tmp_path, monkeypatch):
    import main as M
    args = M.build_parser().parse_args(["pipeline", "--generations", "3", "--no-final-test", "--offline"])
    assert args.func is M.cmd_pipeline and args.generations == 3 and args.no_final_test and args.offline
    seen = {}
    monkeypatch.setattr("src.orchestrator.run_research_loop", lambda cfg_, **kw: seen.update(kw) or "rid")
    M.cmd_pipeline(args, cfg)
    assert seen["final_test"] is False and seen["offline"] is True and seen["iterations"] == 3
