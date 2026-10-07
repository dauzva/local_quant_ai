"""Local evolutionary search over the best strategies -- the zero-token learner.

Each step takes a parent strategy, generates a batch of mutated candidates (see
src/mutation.py), evaluates all of them on the whole universe in parallel, and
promotes the best candidate if it beats the parent's FITNESS by `tune_min_gain`.
The winner becomes a new, separately recorded strategy (`<root>_gN`) and the
search continues from it until `patience` consecutive steps fail or the time
budget runs out.

Parallelism: evaluation is pandas-heavy and therefore GIL-bound, so candidates
are evaluated in worker PROCESSES (each loads the universe once, then receives
only small strategy specs and returns small result objects). If processes are
unavailable it falls back to threads.

What makes it learn rather than random-walk:
- operator choice is weighted by the diagnosis of the parent's results and by
  each operator's measured success rate across the whole history (memory);
- every candidate evaluated updates those success statistics, win or lose;
- every promoted child is logged with its score delta, so later LLM prompts can
  show which kinds of changes actually helped (memory.evolution_log).

Fitness = frequency-adjusted universe score + a bonus for the share of symbols
passing every walk-forward criterion (what acceptance needs). It is computed on
SEARCH folds only. Picking the best of many candidates inflates it a little by
construction; the reserved holdout (evaluated when a strategy is saved) is the
honest check.
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from src import memory
from src.mutation import choose_op, diagnosis_boosts, mutate
from src.strategy_codegen import StrategySpec, build_source
from src.strategy_sandbox import compile_strategy
from src.universe import (behavior_signature, build_eval_settings, evaluate_universe, load_universe, pick_probe)
from src.utils import get_logger

logger = get_logger("tuner")

_FIT = "fitness"


def _fit(summary: dict) -> float:
    return float(summary.get(_FIT, summary.get("score", -9.0)))


def _root_and_gen(name: str) -> tuple[str, int]:
    m = re.match(r"^(.*)_g(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


# --------------------------------------------------------------------------
# Worker side (runs in a separate process; module-level so it can be pickled)
# --------------------------------------------------------------------------

_W: dict = {}


def _worker_init(cfg, symbols, risk_per_trade, max_seconds, ref_n):
    universe = load_universe(cfg, list(symbols))
    _W.update(cfg=cfg, universe=universe, ev=build_eval_settings(cfg), risk=risk_per_trade, max_seconds=max_seconds,
              ref=pick_probe(universe, int((cfg.get("universe", {}) or {}).get("probe_symbols", 12)))[:ref_n])


def _worker_eval(record: dict):
    """-> (behavior_signature, UniverseResult) or None if the candidate is invalid."""
    try:
        spec = StrategySpec.from_record(record)
        compiled = compile_strategy(build_source(spec, _W["risk"]))
        sig_cache: dict = {}
        behavior = behavior_signature(compiled, _W["ref"], sig_cache)
        result = evaluate_universe(compiled, _W["universe"], _W["ev"], _W["cfg"], probe=None, sig_cache=sig_cache,
                                   max_seconds=_W["max_seconds"])
        return behavior, result
    except Exception:
        return None


# --------------------------------------------------------------------------
# Main-process side
# --------------------------------------------------------------------------

def _get_executor(run, workers: int):
    """Lazily create (and cache on `run`) the process pool; None -> use threads."""
    if run.tune_pool_broken or not bool(run.cfg.run.get("tune_processes", True)):
        return None
    if run.tune_executor is None:
        try:
            run.tune_executor = ProcessPoolExecutor(
                max_workers=workers, initializer=_worker_init,
                initargs=(run.cfg, [sd.symbol for sd in run.universe], run.risk_per_trade, run.max_eval_seconds, len(run.ref_symbols)))
            logger.info("Tuner: started %d worker processes", workers)
        except Exception as exc:  # pragma: no cover - platform dependent
            logger.warning("Could not start tuner worker processes (%s); using threads", exc)
            run.tune_pool_broken = True
            return None
    return run.tune_executor


def shutdown(run) -> None:
    ex = getattr(run, "tune_executor", None)
    if ex is not None:
        ex.shutdown(wait=False, cancel_futures=True)
        run.tune_executor = None


def _evaluate_batch(run, cands: list[StrategySpec], workers: int) -> list[tuple[StrategySpec, str, object]]:
    """Evaluate candidates in parallel -> [(spec, behavior, UniverseResult)] for the valid, new ones."""
    outs: list = []
    ex = _get_executor(run, workers)
    if ex is not None:
        try:
            outs = list(ex.map(_worker_eval, [c.to_record() for c in cands]))
        except Exception as exc:  # BrokenProcessPool etc.
            logger.warning("Tuner worker pool failed (%s); falling back to threads", exc)
            run.tune_pool_broken = True
            shutdown(run)
            ex = None
    if ex is None:
        def one(c):
            try:
                compiled = compile_strategy(build_source(c, run.risk_per_trade))
                sig_cache: dict = {}
                behavior = behavior_signature(compiled, run.ref_symbols, sig_cache)
                result = evaluate_universe(compiled, run.universe, run.ev, run.cfg, probe=None, sig_cache=sig_cache,
                                           max_seconds=run.max_eval_seconds)
                return behavior, result
            except Exception:
                return None
        with ThreadPoolExecutor(max_workers=workers) as tex:
            outs = list(tex.map(one, cands))

    valid = []
    for cand, out in zip(cands, outs):
        if out is None:
            continue
        behavior, result = out
        if memory.has_behavior(run.state, behavior):      # identical signals to something already tried
            continue
        memory.remember_behavior(run.state, behavior)
        valid.append((cand, behavior, result))
    return valid


def _pool(run, top_k: int) -> list[tuple[StrategySpec, float, dict]]:
    """Parents for this pass: the best of this iteration's fresh candidates plus
    the best distinct leaderboard strategies (discounted by how often their
    lineage was already tuned, to spread effort), plus a couple of
    clearly-negative-gross strategies as inversion candidates."""
    cands: list[tuple[StrategySpec, float, dict]] = []
    seen: set[str] = set()

    def add(spec: StrategySpec, fitness: float, summary: dict) -> None:
        key = re.sub(r"\s+", " ", spec.body).strip() + json.dumps(spec.params, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        cands.append((spec, fitness, summary))

    for spec, fitness, summary in run.tunable:
        add(spec, fitness, summary)
    for e in memory.pick_parents(run.state, top_k):
        add(StrategySpec.from_record(e), memory.rank_value(e), e["summary"])

    def priority(t) -> float:
        return t[1] - 0.05 * run.tuned_count.get(_root_and_gen(t[0].name)[0], 0)

    ranked = sorted(cands, key=lambda t: -priority(t))
    chosen = ranked[:top_k]
    inversion = sorted((t for t in ranked[top_k:] if t[2].get("gross_pnl_pct", 0) < -0.5 and t[2].get("pct_positive", 1) < 0.4),
                       key=lambda t: t[2].get("gross_pnl_pct", 0))[:2]
    return chosen + inversion


def run_tuning(run, iteration: int, seconds: float) -> dict:
    cfg = run.cfg.run
    top_k = int(cfg.get("tune_top_k", 6))
    n_cand = int(cfg.get("tune_candidates", 24))
    patience = int(cfg.get("tune_patience", 2))
    min_gain = float(cfg.get("tune_min_gain", 0.005))
    workers = max(1, min(int(cfg.get("tune_workers", 16)), max(1, (os.cpu_count() or 4) - 2)))
    if seconds <= 0 or top_k <= 0:
        return {}
    pool = _pool(run, top_k)
    if not pool:
        return {}

    deadline = time.monotonic() + seconds
    active = [{"spec": s, "fit": f, "summary": sm, "fails": 0, "start": f} for s, f, sm in pool]
    all_states = list(active)
    evaluated = promoted = 0
    op_tried: dict[str, int] = {}
    t0 = time.monotonic()
    while active and time.monotonic() < deadline:
        for st in list(active):
            if time.monotonic() >= deadline:
                break
            boosts = diagnosis_boosts(st["summary"], run.min_tpy)
            stats = run.state.get("mutation_stats", {})
            cands, keys = [], set()
            for _ in range(n_cand * 3):
                if len(cands) >= n_cand:
                    break
                op = choose_op(run.rng, stats, boosts)
                m = mutate(st["spec"], op, run.rng)
                if m is None:
                    continue
                key = (op, m.body, json.dumps(m.params, sort_keys=True), m.sl, m.tp)
                if key in keys:
                    continue
                keys.add(key)
                cands.append(m)
            if not cands:
                active.remove(st)
                continue

            results = _evaluate_batch(run, cands, workers)
            evaluated += len(results)

            per_op: dict[str, list[float]] = {}
            for cand, _b, res in results:
                per_op.setdefault(cand.mutation, []).append(_fit(res.summary) - st["fit"])
            for op, deltas in per_op.items():
                wins = [d for d in deltas if d > 1e-9]
                memory.record_mutation(run.state, op, len(deltas), len(wins), sum(wins))
                op_tried[op] = op_tried.get(op, 0) + len(deltas)

            promoted_now = False
            for cand, behavior, res in sorted(results, key=lambda r: -_fit(r[2].summary)):
                if _fit(res.summary) < st["fit"] + min_gain:
                    break
                try:
                    compiled = compile_strategy(build_source(cand, run.risk_per_trade))
                except Exception:
                    continue
                root, gen = _root_and_gen(st["spec"].name)
                cand.name = memory.unique_name(run.state, f"{root}_g{gen + 1}", run.used_names)
                run.used_names.add(cand.name)
                cand.idea = f"{st['spec'].idea.split(' [')[0]} [{cand.mutation}]"
                run._finalize(cand, compiled, res, behavior, iteration, "local-tuner", tuned=True)
                promoted += 1
                run.tuned_count[root] = run.tuned_count.get(root, 0) + 1
                st.update(spec=cand, fit=_fit(res.summary), summary=res.summary, fails=0)
                promoted_now = True
                break
            if not promoted_now:
                st["fails"] += 1
                if st["fails"] >= patience:
                    active.remove(st)

    gains = [st["fit"] - st["start"] for st in all_states if st["fit"] > st["start"]]
    info = {"parents": len(pool), "evaluated": evaluated, "promoted": promoted,
            "seconds": round(time.monotonic() - t0, 1), "mean_gain": (sum(gains) / len(gains)) if gains else 0.0,
            "ops": dict(sorted(op_tried.items(), key=lambda kv: -kv[1])[:5])}
    logger.info("Local tuning: %d parents, %d candidates evaluated in %.0fs (%.1f/s, %d workers), %d promoted to new strategies; "
                "most-tried ops %s", info["parents"], evaluated, info["seconds"], evaluated / max(1e-9, info["seconds"]), workers,
                promoted, info["ops"])
    return info
