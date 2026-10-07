"""Research loop (universe edition): generate strategies in bulk -> sandbox ->
evaluate each on ~50 symbols -> remember -> feed results back to the generator.

One iteration:
  1. Plan N generation requests: mostly EXPLORE (fresh, theme-seeded ideas in
     bulk) plus a few EVOLVE requests that send the best strategies found so
     far back to the LLM together with their per-asset-class results and
     rule-based diagnosis, asking for improved variants.
  2. Fire the requests concurrently (thread pool); as each completes, parse the
     snippets, wrap + sandbox-check them, drop structural/behavioral
     duplicates, and run the survivors across the whole universe while the
     other requests are still in flight.
  3. Strategies passing the universe screen get the one-shot reserved-holdout
     check and are saved; everything is written to the cross-run memory.

The optimisation targets are the user's: speed (parallel requests, numba
backtests, probe pre-screen), cheap tokens (snippet format, ~10 strategies per
request, no extra "analysis" request, no repair round-trips), and quantity
(statistically, out of thousands some are good).
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import insights, memory, seeds, tuner
from src.mutation import OP_LABELS
from src.llm import AuthError, DailyQuotaExhaustedError, LLMError, OpenRouterClient
from src.report import write_universe_report
from src.strategy_codegen import (StrategySpec, build_evolve_prompt, build_explore_prompt, build_source,
                                  friendly_problem, parse_specs)
from src.strategy_sandbox import CompiledStrategy, StrategySandboxError, compile_strategy
from src.universe import (SymbolData, UniverseResult, behavior_signature, build_eval_settings, evaluate_universe,
                          load_universe, pick_probe, run_universe_holdout)
from src.utils import get_logger, new_run_id, safe_strategy_dirname

logger = get_logger("orchestrator")

BEHAVIOR_REF_SYMBOLS = 3
NAMES_IN_PROMPT = 40
BEST_IN_EXPLORE_PROMPT = 3


@dataclass
class GenJob:
    kind: str                   # "explore" | "evolve"
    prompt: str
    expected: int
    max_tokens: int
    temperature: float
    parent_names: list[str] = field(default_factory=list)


@dataclass
class JobOutput:
    job: GenJob
    text: str = ""
    model: str = "none"
    error: str | None = None


# --------------------------------------------------------------------------
# Prompt context
# --------------------------------------------------------------------------

def _flat(body: str, limit: int = 420) -> str:
    body = re.sub(r"[ \t]*#[^\n]*", "", body)          # drop comments (incl. '# mut:' markers) so they cannot swallow code
    return re.sub(r"\s*\n\s*", " ; ", body.strip())[:limit]


def _memory_text(state: dict, include_best: bool, include_names: bool) -> str:
    parts = []
    last = state.get("last_batch")
    if last and last.get("n"):
        parts.append(f"- Your previous batch ({last['n']} strategies): {last['too_few_trades']} rejected for too few trades, "
                     f"{last['invalid']} invalid/crashed, {last['profitable']} profitable on most markets, mean score {last['mean_score']:+.2f}. "
                     "Do better: more trades, zero crashes.")
    fam = memory.family_table(state)
    if fam:
        parts.append(f"- Family yields so far: {fam}. Lean toward what pays off, but keep exploring.")
    if include_best:
        best = memory.best_ever(state, BEST_IN_EXPLORE_PROMPT)
        if best:
            lines = [f"  * {e['name']} [{e['family']}] score {e['score']:+.2f}; {e['summary'].get('pct_positive', 0):.0%} of markets profitable; "
                     f"{e['summary'].get('trades_per_year', 0):.0f} trades/yr; sl={e.get('sl')} tp={e.get('tp')} params={json.dumps(e.get('params', {}))}\n"
                     f"    code: {_flat(e.get('body', ''))}" for e in best]
            parts.append("- Best strategies so far, with code. Study WHY they work (trade frequency, exit style, filters), then invent "
                         "DIFFERENT mechanisms that share those traits; never resubmit them:\n" + "\n".join(lines))
    effects = memory.mutation_table(state, OP_LABELS)
    if effects:
        parts.append("- MEASURED EFFECT OF CHANGES (from many real tests; use it when deciding how to improve or build a strategy): " + effects)
    good, bad = memory.llm_change_examples(state)
    if good:
        parts.append("- Changes YOU proposed earlier that improved the score: " + " | ".join(good))
    if bad:
        parts.append("- Changes you proposed that made it WORSE (don't repeat the pattern): " + " | ".join(bad))
    freq = memory.freq_table(state)
    if freq:
        parts.append(f"- Mean score by trade frequency so far: {freq}. (Too many trades burn the edge in costs; too few are luck.)")
    lessons = memory.top_lessons(state, 6)
    if lessons:
        parts.append("- Mistakes that got candidates discarded recently (counts in brackets) -- do NOT repeat any:\n  * "
                     + "\n  * ".join(lessons))
    if include_names:
        names = memory.recent_names(state, NAMES_IN_PROMPT)
        if names:
            parts.append("- Names/ideas already tried (do not repeat): " + ", ".join(names))
    return "\n".join(parts) if parts else "- Nothing yet: this is the first batch."


def _accept_cfg(cfg) -> tuple[float, float]:
    acc = (cfg.get("universe", {}) or {}).get("accept", {}) or {}
    return float(acc.get("min_trades_per_year", 10)), float(acc.get("min_pct_positive", 0.6))


def _plan_jobs(cfg, state: dict, n_symbols: int, rng: random.Random) -> list[GenJob]:
    run = cfg.run
    llm = cfg.llm
    min_tpy, accept_pct = _accept_cfg(cfg)
    per_call = int(llm.get("strategies_per_call", 10))
    tokens_per = int(llm.get("tokens_per_strategy", 330))
    cap = int(llm.get("max_tokens", 6000))
    jobs: list[GenJob] = []

    for _ in range(int(run.get("explore_calls_per_iteration", 4))):
        prompt = build_explore_prompt(per_call, n_symbols, _memory_text(state, True, True), rng,
                                      min_tpy=min_tpy, accept_pct=accept_pct)
        jobs.append(GenJob("explore", prompt, per_call, min(cap, per_call * tokens_per + 600),
                           float(llm.get("explore_temperature", 0.9))))

    evolve_calls = int(run.get("evolve_calls_per_iteration", 2))
    parents_per_call = int(llm.get("parents_per_call", 2))
    variants = int(llm.get("variants_per_parent", 3))
    if evolve_calls and state.get("leaderboard"):
        parents = memory.pick_parents(state, evolve_calls * parents_per_call)
        for i in range(0, len(parents), parents_per_call):
            chunk = parents[i:i + parents_per_call]
            infos = [{
                "name": e["name"], "family": e.get("family", "other"), "idea": e.get("idea", ""), "sl": e.get("sl"),
                "tp": e.get("tp"), "params": e.get("params", {}), "body": e.get("body", ""),
                "results": insights.results_line(e["summary"]), "diagnosis": insights.diagnose(e["summary"], min_tpy),
                "history": memory.lineage(state, e["name"]),
            } for e in chunk]
            n_expected = variants * len(chunk)
            prompt = build_evolve_prompt(infos, variants, n_symbols, _memory_text(state, False, False),
                                         min_tpy=min_tpy, accept_pct=accept_pct)
            jobs.append(GenJob("evolve", prompt, n_expected, min(cap, n_expected * tokens_per + 600),
                               float(llm.get("evolve_temperature", 0.6)), [e["name"] for e in chunk]))
    return jobs


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

class Run:
    """Holds the mutable state of one research run so the per-strategy
    processing can be a method instead of a 15-argument function."""

    def __init__(self, cfg, symbols: list[str] | None, offline: bool = False):
        self.cfg = cfg
        self.offline = offline
        self.llm_ok = not offline          # flips to False when the quota/key dies (offline fallback)
        self.state_lock = threading.Lock()
        self.run_id = new_run_id(cfg.run.get("name", "local_quant_run"))
        self.dir = Path("results") / self.run_id
        (self.dir / "llm_iterations").mkdir(parents=True, exist_ok=True)
        (self.dir / "strategies").mkdir(parents=True, exist_ok=True)

        started = time.monotonic()
        self.universe: list[SymbolData] = load_universe(cfg, symbols)
        if not self.universe:
            raise RuntimeError("No usable symbols after filtering; check the data directory and the universe: section of config.yaml")
        logger.info("Loaded %d symbols in %.1fs: %s", len(self.universe), time.monotonic() - started,
                    ", ".join(sorted({f"{g}={sum(1 for s in self.universe if s.group == g)}" for g in {s.group for s in self.universe}})))
        ucfg = cfg.get("universe", {}) or {}
        self.probe = pick_probe(self.universe, int(ucfg.get("probe_symbols", 12)))
        self.ref_symbols = self.probe[:BEHAVIOR_REF_SYMBOLS]
        self.ev = build_eval_settings(cfg)
        self.max_eval_seconds = float(ucfg.get("max_eval_seconds", 90))

        self.memory_path = Path(cfg.get("memory", {}).get("path", memory.DEFAULT_MEMORY_PATH))
        self.experiments_path = self.memory_path.with_name("experiments.jsonl")
        self.state = memory.load_memory(self.memory_path)
        logger.info("Loaded memory: %d strategies tried in past runs, %d accepted, %d on the leaderboard",
                    self.state.get("total_tried", 0), self.state.get("total_accepted", 0), len(self.state.get("leaderboard", [])))

        self.client = OpenRouterClient(cfg)
        self.rng = random.Random(int(cfg.run.get("random_seed", 42)) + int(self.state.get("total_tried", 0)))
        self.rows: list[dict] = []
        self.used_names: set[str] = set()
        self.risk_per_trade = float(cfg.backtest.get("risk_per_trade", 0.01))
        self.n_accepted = 0
        self.min_tpy, _ = _accept_cfg(cfg)
        self.tunable: list[tuple[StrategySpec, float, dict]] = []   # (spec, score, summary) evaluated this iteration
        self.tuned_count: dict[str, int] = {}
        self.score_of: dict[str, tuple[float, float]] = {e["name"]: (memory.rank_value(e), e["summary"].get("trades_per_year", 0.0))
                                                         for e in self.state.get("leaderboard", [])}
        # fully evaluated strategies of this run, for saving the top N at the end
        self.kept: dict[str, tuple[StrategySpec, CompiledStrategy, UniverseResult, str]] = {}
        self.tune_executor = None          # process pool, created lazily by src.tuner
        self.tune_pool_broken = False

    # ---- per-strategy processing -----------------------------------------

    def _row(self, iteration: int, spec: StrategySpec, status: str, model: str, **extra) -> dict:
        row = {"iteration": iteration, "name": spec.name, "family": spec.family, "parent": spec.parent or "",
               "status": status, "model": model}
        row.update(extra)
        self.rows.append(row)
        return row

    def process_spec(self, spec: StrategySpec, iteration: int, model: str) -> dict:
        state = self.state
        if spec.problems:
            memory.record_invalid(state, [friendly_problem(p) for p in spec.problems])
            return self._row(iteration, spec, "invalid_format", model, reasons="; ".join(spec.problems))

        spec.name = memory.unique_name(state, spec.name, self.used_names)
        self.used_names.add(spec.name)

        try:
            source = build_source(spec, self.risk_per_trade)
            compiled = compile_strategy(source)
        except StrategySandboxError as exc:
            memory.record_invalid(state, [friendly_problem(p) for p in exc.problems])
            return self._row(iteration, spec, "invalid", model, reasons="; ".join(exc.problems)[:300])
        except SyntaxError as exc:
            memory.record_invalid(state, [f"SyntaxError: {exc.msg}"])
            return self._row(iteration, spec, "invalid", model, reasons=f"SyntaxError: {exc.msg}")

        record = spec.to_record()
        if memory.has_signature(state, compiled.signature):
            return self._row(iteration, spec, "duplicate", model, reasons="structural signature already tried")

        sig_cache: dict = {}
        try:
            behavior = behavior_signature(compiled, self.ref_symbols, sig_cache)
        except Exception as exc:  # strategy crashes on real data
            memory.record_invalid(state, [friendly_problem(f"runtime error on real data: {type(exc).__name__}: {exc}")])
            return self._row(iteration, spec, "invalid", model, reasons=f"runtime error on real data: {exc}"[:300])
        no_signal = all(not (s[0].any() or s[2].any()) for s in sig_cache.values())
        if no_signal:
            memory.record_unevaluated(state, record, compiled.signature, behavior)
            memory.record_invalid(state, ["entry condition never true on real data (too strict or logic error): loosen thresholds"])
            return self._row(iteration, spec, "no_signals", model, reasons="no entry signals on reference symbols")
        if memory.has_behavior(state, behavior):
            memory.record_unevaluated(state, record, compiled.signature)
            return self._row(iteration, spec, "duplicate", model, reasons="identical signals to an earlier strategy")

        try:
            result: UniverseResult = evaluate_universe(compiled, self.universe, self.ev, self.cfg, probe=self.probe,
                                                       sig_cache=sig_cache, max_seconds=self.max_eval_seconds)
        except Exception as exc:
            logger.exception("Unexpected error evaluating '%s'", spec.name)
            return self._row(iteration, spec, "error", model, reasons=str(exc)[:300])

        return self._finalize(spec, compiled, result, behavior, iteration, model)

    def _finalize(self, spec: StrategySpec, compiled: CompiledStrategy, result: UniverseResult, behavior: str,
                  iteration: int, model: str, tuned: bool = False) -> dict:
        """Everything after a universe evaluation: holdout + save when accepted,
        memory, experiment log, lessons, and the CSV row."""
        state = self.state
        record = spec.to_record()
        s = result.summary
        accepted = bool(s.get("accepted"))
        holdout = None
        if accepted:
            holdout = run_universe_holdout(compiled, self.universe, self.ev, self.cfg)
            self._save_strategy(compiled, spec, result, holdout, model, accepted=True)
            self.n_accepted += 1
            logger.info("ACCEPTED %-34s score %+.2f | %.0f%% profitable | passes %d/%d | %.0f trades/yr | holdout median Sharpe %+.2f",
                        spec.name, s["score"], 100 * s["pct_positive"], s["n_pass"], s["n_eval"], s["trades_per_year"],
                        holdout["median_sharpe"])

        memory.record_result(state, spec=record, summary=s, run_id=self.run_id, iteration=iteration,
                             signature=compiled.signature, behavior=behavior, accepted=accepted, holdout=holdout)
        if s.get("trades_per_year", 99) < self.min_tpy:
            memory.record_invalid(state, [f"too few trades: {s.get('trades_per_year', 0):.1f}/yr (< {self.min_tpy:.0f} required): "
                                          "shorter lookbacks, shallower thresholds, fewer AND conditions, add exits and the short side"])
        if result.probe_only:
            status = "low_frequency" if str(s.get("reject_reason", "")).startswith("low_frequency") else "probe_rejected"
        else:
            status = "accepted" if accepted else "rejected"
        memory.append_experiment({
            "run_id": self.run_id, "iteration": iteration, "model": model, **record, "tuned": tuned, "status": status,
            "summary": {k: v for k, v in s.items() if k not in ("best_symbols", "worst_symbols")},
            "per_symbol": {r.symbol: [round(r.avg_sharpe, 3), round(r.min_sharpe, 3), r.total_trades] for r in result.per_symbol},
        }, self.experiments_path)
        if not result.probe_only:
            score = float(s.get("fitness", s.get("score", -9.0)))     # fitness drives every search/ranking decision
            if spec.parent in self.score_of:
                p_score, p_tpy = self.score_of[spec.parent]
                memory.record_evolution(state, spec.parent, spec.name, spec.mutation or "llm", spec.idea,
                                        score - p_score, float(s.get("trades_per_year", 0.0)) - p_tpy)
            self.score_of[spec.name] = (score, float(s.get("trades_per_year", 0.0)))
            if s.get("score", -9.0) > 0:
                self.kept[spec.name] = (spec, compiled, result, model)
                if len(self.kept) > 120:
                    for k, _ in sorted(self.kept.items(), key=lambda kv: kv[1][2].summary.get("fitness", -9))[:20]:
                        del self.kept[k]
            if not tuned:
                self.tunable.append((spec, score, s))

        return self._row(
            iteration, spec, status, model, score=s.get("score"), sharpe_score=s.get("sharpe_score"),
            median_sharpe=s.get("median_sharpe"), pct_positive=s.get("pct_positive"), n_pass=s.get("n_pass"),
            n_eval=s.get("n_eval"), trades_per_year=s.get("trades_per_year"), trades_per_symbol=s.get("trades_per_symbol"),
            eval_seconds=round(result.seconds, 2), tuned=tuned,
            reasons=s.get("reject_reason", ""), holdout_median_sharpe=holdout["median_sharpe"] if holdout else None,
        )

    def _save_strategy(self, compiled: CompiledStrategy, spec: StrategySpec, result: UniverseResult,
                       holdout: dict, model: str, accepted: bool) -> Path:
        """Write strategy.py + metadata + per-symbol table/chart + report under
        strategies/<name>/. Used for accepted strategies and for the end-of-run
        top-N (accepted=False)."""
        strat_dir = self.dir / "strategies" / safe_strategy_dirname(spec.name)
        strat_dir.mkdir(parents=True, exist_ok=True)
        (strat_dir / "strategy.py").write_text(compiled.source, encoding="utf-8")
        metadata = {
            "strategy_name": spec.name, "class_name": compiled.class_name, "family": spec.family, "idea": spec.idea,
            "parent": spec.parent, "mutation": spec.mutation, "accepted": accepted, "params": spec.params,
            "stop_loss_pct": compiled.stop_loss_pct,
            "take_profit_pct": compiled.take_profit_pct, "signature": compiled.signature, "model_used": model,
            "universe_symbols": len(self.universe), "universe_summary": result.summary,
            "holdout": {k: v for k, v in holdout.items() if k != "rows"},
        }
        (strat_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        write_universe_report(strat_dir, spec, compiled, result, holdout, self.universe, accepted=accepted)
        return strat_dir

    def save_top(self, n: int) -> list[dict]:
        """Save the top-n strategies of this run (by search score, distinct
        code) into strategies/, accepted or not, so the best results are
        always available as ready-to-use files. The holdout is evaluated for
        the report only; ranking uses search folds exclusively."""
        ranked = sorted(self.kept.values(), key=lambda t: -t[2].summary.get("fitness", t[2].summary.get("score", -9)))
        out, seen = [], set()
        for spec, compiled, result, model in ranked:
            key = re.sub(r"\s+", " ", spec.body).strip()      # one slot per distinct code (not per parameter set)
            if key in seen:
                continue
            seen.add(key)
            d = self.dir / "strategies" / safe_strategy_dirname(spec.name)
            if not d.exists():
                holdout = run_universe_holdout(compiled, self.universe, self.ev, self.cfg)
                d = self._save_strategy(compiled, spec, result, holdout, model, accepted=bool(result.summary.get("accepted")))
            out.append({"name": spec.name, "dir": str(d), "score": result.summary.get("score"),
                        "pct_positive": result.summary.get("pct_positive"), "n_pass": result.summary.get("n_pass"),
                        "trades_per_year": result.summary.get("trades_per_year"),
                        "trades_per_symbol": result.summary.get("trades_per_symbol"),
                        "accepted": bool(result.summary.get("accepted")), "family": spec.family, "mutation": spec.mutation or ""})
            if len(out) >= n:
                break
        return out

    # ---- generation ------------------------------------------------------

    def _run_job(self, job: GenJob) -> JobOutput:
        max_cost = self.cfg.llm.get("max_cost_usd")
        if max_cost is not None and self.client.total_cost() >= float(max_cost):
            return JobOutput(job, error="budget")
        try:
            resp = self.client.complete_text(job.prompt, max_tokens=job.max_tokens, temperature=job.temperature)
            return JobOutput(job, text=resp.text, model=resp.model_used)
        except (DailyQuotaExhaustedError, AuthError):
            raise
        except LLMError as exc:
            return JobOutput(job, error=str(exc)[:300])

    def run_iteration(self, iteration: int) -> dict:
        t0 = time.monotonic()
        memory.decay_lessons(self.state)
        self.tunable = []
        jobs = _plan_jobs(self.cfg, self.state, len(self.universe), self.rng) if self.llm_ok else []
        for j in jobs:
            if j.kind == "evolve":
                memory.mark_evolved(self.state, j.parent_names)
        concurrency = max(1, int(self.cfg.llm.get("concurrency", 3)))
        logger.info("=== Iteration %d: %d LLM requests (%d explore, %d evolve), concurrency %d%s ===", iteration, len(jobs),
                    sum(j.kind == "explore" for j in jobs), sum(j.kind == "evolve" for j in jobs), concurrency,
                    "" if self.llm_ok else " | OFFLINE: seeds + local search only")

        n_parsed = 0
        fatal: Exception | None = None
        raw_dir = self.dir / "llm_iterations"
        rows_before = len(self.rows)
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(self._run_job, j) for j in jobs]
            for k, fut in enumerate(as_completed(futures), start=1):
                try:
                    out = fut.result()
                except CancelledError:   # cancelled after a fatal error elsewhere
                    continue
                except (DailyQuotaExhaustedError, AuthError) as exc:
                    fatal = fatal or exc
                    for f in futures:
                        f.cancel()
                    continue
                if out.error:
                    logger.warning("%s request failed: %s", out.job.kind, out.error)
                    continue
                (raw_dir / f"iter{iteration:02d}_{out.job.kind}_{k}.txt").write_text(out.text, encoding="utf-8")
                specs = parse_specs(out.text)
                n_parsed += len(specs)
                if len(specs) < out.job.expected:
                    logger.info("%s request returned %d/%d usable blocks (model %s)", out.job.kind, len(specs), out.job.expected, out.model)
                job_rows = [self.process_spec(spec, iteration, out.model) for spec in specs]
                usable = sum(1 for r in job_rows if r["status"] not in ("invalid", "invalid_format", "error"))
                getattr(self.client, "report_yield", lambda *a: None)(out.model, usable, out.job.expected)
                self._checkpoint()
        if fatal:
            if bool(self.cfg.run.get("offline_fallback", True)):
                logger.warning("LLM unavailable (%s) -- continuing OFFLINE with seed templates + local mutation search; "
                               "everything learned so far is kept", str(fatal)[:120])
                self.llm_ok = False
                fatal = None
        # Zero-token sources of candidates: classic templates with random parameters.
        n_seed = int(self.cfg.run.get("seed_samples_per_iteration", 10)) * (1 if self.llm_ok else 3)
        for spec in seeds.sample_seed_specs(n_seed, self.rng):
            self.process_spec(spec, iteration, "seed")
        tune_seconds = float(self.cfg.run.get("tune_seconds", 60))
        tuning = tuner.run_tuning(self, iteration, tune_seconds * (1 if self.llm_ok else 2)) if not fatal else {}
        self._checkpoint()

        new_rows = self.rows[rows_before:]
        counts: dict[str, int] = {}
        for r in new_rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        llm_rows = [r for r in new_rows if r["model"] not in ("seed", "local-tuner")]   # what the LLM itself wrote
        scored = [r["score"] for r in new_rows if r.get("score") is not None]
        llm_scored = [r["score"] for r in llm_rows if r.get("score") is not None]
        usage = self.client.usage.summary()
        best_ever = max((e["score"] for e in self.state.get("leaderboard", [])), default=None)
        mean_llm = sum(llm_scored) / len(llm_scored) if llm_scored else 0.0
        # Feedback for the next prompt (what went wrong in THIS batch).
        if llm_rows:
            self.state["last_batch"] = {
                "n": len(llm_rows),
                "too_few_trades": sum(1 for r in llm_rows if r["status"] == "low_frequency"
                                      or (r.get("trades_per_year") is not None and r["trades_per_year"] < self.min_tpy)),
                "invalid": sum(1 for r in llm_rows if r["status"] in ("invalid", "invalid_format", "error", "no_signals")),
                "profitable": sum(1 for r in llm_rows if (r.get("pct_positive") or 0) >= 0.6),
                "mean_score": round(mean_llm, 3),
            }
        info = {"iteration": iteration, "parsed": n_parsed, "statuses": counts, "seconds": round(time.monotonic() - t0, 1),
                "best_score": max(scored) if scored else None, "tokens": usage["total_tokens"], "cost_usd": usage["cost_usd"],
                "mean_llm_score": mean_llm, "best_ever": best_ever, "jobs": len(jobs)}
        logger.info("Iteration %d done in %.0fs: %d parsed -> %s | LLM-written mean score %+.3f | tuning %s | BEST EVER %s | tokens so far %d ($%.4f)",
                    iteration, info["seconds"], n_parsed, counts, mean_llm,
                    (f"{tuning['promoted']} improvements from {tuning['evaluated']} candidates (mean gain +{tuning['mean_gain']:.3f})" if tuning else "n/a"),
                    f"{best_ever:+.3f}" if best_ever is not None else "n/a", usage["total_tokens"], usage["cost_usd"])
        if fatal:
            info["fatal"] = fatal
        return info

    def _checkpoint(self) -> None:
        memory.save_memory(self.state, self.memory_path)
        if self.rows:
            pd.DataFrame(self.rows).to_csv(self.dir / "all_experiments.csv", index=False)


def run_research_loop(cfg, symbols: list[str] | None = None, iterations: int | None = None,
                      minutes: float | None = None, offline: bool = False) -> str:
    """Run the universe research loop. `iterations=0` (or None with
    run.iterations: 0) runs until the time limit, the LLM budget is exhausted,
    or Ctrl-C -- progress is checkpointed after every request. If the LLM quota
    or key dies, the run continues offline (seed templates + local mutation
    search) unless run.offline_fallback is false. `offline=True` skips the LLM
    entirely (CLI: `python main.py tune`)."""
    iterations = int(cfg.run.get("iterations", 10)) if iterations is None else int(iterations)
    minutes = cfg.run.get("max_minutes") if minutes is None else minutes
    run = Run(cfg, symbols, offline=offline)
    deadline = time.monotonic() + float(minutes) * 60 if minutes else None
    logger.info("Starting run %s: %s iterations over %d symbols, probe=%d",
                run.run_id, iterations or "unlimited", len(run.universe), len(run.probe))

    stall = 0
    it = 0
    try:
        while iterations == 0 or it < iterations:
            if deadline and time.monotonic() > deadline:
                logger.info("Time limit reached; stopping")
                break
            it += 1
            info = run.run_iteration(it)
            if info.get("fatal"):
                logger.error("Stopping run: %s", info["fatal"])
                break
            max_cost = cfg.llm.get("max_cost_usd")
            if max_cost is not None and run.client.total_cost() >= float(max_cost):
                logger.error("Stopping run: LLM spend reached max_cost_usd=$%s", max_cost)
                break
            stall = stall + 1 if (run.llm_ok and info["jobs"] and info["parsed"] == 0) else 0
            if stall >= 2:
                logger.error("Stopping run: two iterations in a row produced no usable strategies "
                             "(check the model list / API key / quota)")
                break
    except KeyboardInterrupt:
        logger.warning("Interrupted; saving progress")
    finally:
        run._checkpoint()
        _finish(run)
    return run.run_id


def _finish(run: Run) -> None:
    tuner.shutdown(run)
    run.client.usage.log_summary()
    run.client.usage.save(run.dir / "llm_usage.json")
    usage = run.client.usage.summary()
    evaluated = [r for r in run.rows if r.get("score") is not None]
    # Display AND generate the top strategies: strategy.py + report + per-symbol table for each.
    top = run.save_top(int(run.cfg.run.get("save_top_n", 10)))
    if top:
        pd.DataFrame(top).to_csv(run.dir / "top_strategies.csv", index=False)
    memory.record_run(run.state, {
        "run_id": run.run_id, "strategies": len(run.rows), "evaluated": len(evaluated), "accepted": run.n_accepted,
        "best_score": top[0]["score"] if top else None, "tokens": usage["total_tokens"], "cost_usd": usage["cost_usd"],
    })
    memory.save_memory(run.state, run.memory_path)

    logger.info("Run %s complete: %d candidates -> %d evaluated on %d symbols -> %d accepted",
                run.run_id, len(run.rows), len(evaluated), len(run.universe), run.n_accepted)
    if evaluated and usage["total_tokens"]:
        logger.info("Tokens per evaluated strategy: %.0f | per accepted: %s", usage["total_tokens"] / len(evaluated),
                    f"{usage['total_tokens'] / run.n_accepted:.0f}" if run.n_accepted else "n/a")
    for i, r in enumerate(top, start=1):
        logger.info("  #%-2d %-36s score %+.3f | %.0f%% markets profitable | %s trades/yr (~%s per market) | passes %s/%d | %s",
                    i, r["name"], r["score"], 100 * (r.get("pct_positive") or 0), f"{r['trades_per_year']:.0f}",
                    r.get("trades_per_symbol"), r.get("n_pass"), len(run.universe),
                    "ACCEPTED" if r["accepted"] else "not accepted")
    if top:
        logger.info("Generated %d strategies (strategy.py + report.md + per_symbol.csv/png) in %s", len(top), run.dir / "strategies")


def backtest_single_strategy_file(strategy_path: str, cfg, symbols: list[str] | None = None) -> None:
    """CLI helper: run one saved strategy.py across the universe (or the given
    symbols) and print the per-symbol table + holdout summary."""
    source = Path(strategy_path).read_text(encoding="utf-8")
    try:
        compiled = compile_strategy(source)
    except StrategySandboxError as exc:
        logger.error("Strategy file failed validation: %s", exc.problems)
        return
    universe = load_universe(cfg, symbols)
    ev = build_eval_settings(cfg)
    result = evaluate_universe(compiled, universe, ev, cfg, probe=None)
    s = result.summary
    print(f"\n{compiled.strategy_name}: {insights.results_line(s)}")
    print(f"Diagnosis: {insights.diagnose(s)}\n")
    print(f"{'symbol':14s} {'group':13s} {'avg_sharpe':>10s} {'min_fold':>9s} {'trades':>7s} {'pass':>5s}  fold_sharpes")
    for r in sorted(result.per_symbol, key=lambda r: -r.avg_sharpe):
        print(f"{r.symbol:14s} {r.group:13s} {r.avg_sharpe:10.2f} {r.min_sharpe:9.2f} {r.total_trades:7d} {str(r.passed):>5s}  {r.fold_sharpes}"
              + (f"  ERROR {r.error}" if r.error else ""))
    h = run_universe_holdout(compiled, universe, ev, cfg)
    print(f"\nReserved holdout (never used for selection): median Sharpe {h['median_sharpe']:+.2f}, "
          f"mean (clipped) {h['mean_sharpe']:+.2f}, {h['pct_positive']:.0%} of symbols profitable")
    print(f"Universe screen: {'ACCEPTED' if s.get('accepted') else 'not accepted'}")
