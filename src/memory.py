"""Persistent, cross-run learning memory (universe edition).

Everything the generator needs to continue where previous runs stopped lives in
`results/memory/state.json`; every evaluated strategy is additionally appended
(with its per-symbol results) to `results/memory/experiments.jsonl` for later
analysis. The memory is built around universe results: a strategy's record is
its aggregate score over ~50 symbols, its asset-class profile, and its code.

state.json:
- `leaderboard`: the best strategies ever seen by universe score, WITH their
  code and parameters, so they can be sent back to the LLM to improve.
- `signatures` / `behaviors`: structural + behavioral fingerprints of every
  strategy already tried, so near-duplicates are skipped before any backtest.
- `family_stats`: per-family hit rates (which kinds of ideas pay off).
- `error_lessons`: the most frequent validation errors, fed back into the
  prompt so the model stops repeating them.
- `recent_names`: names already used (the prompt asks the model to avoid
  repeats; the orchestrator also forces unique names).

The reserved holdout results are stored on leaderboard entries for YOUR
reading but are never included in anything sent to the LLM.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from src.utils import get_logger

logger = get_logger("memory")

DEFAULT_MEMORY_PATH = Path("results") / "memory" / "state.json"
# v3: scores are frequency-adjusted (trades/yr gate); v2 leaderboards ranked rarely-trading lucky strategies on top.
STATE_VERSION = 3

LEADERBOARD_SIZE = 40
MAX_SIGNATURES = 20000
MAX_NAMES = 4000
MAX_RUNS = 60
MAX_LESSONS = 60
MAX_EVOLUTION_LOG = 400
FREQ_BINS = [(0, 5, "<5"), (5, 10, "5-10"), (10, 20, "10-20"), (20, 40, "20-40"), (40, 1e9, ">40")]


def _empty_state() -> dict:
    return {
        "version": STATE_VERSION,
        "updated_at": None,
        "total_tried": 0,
        "total_accepted": 0,
        "leaderboard": [],
        "signatures": [],
        "behaviors": [],
        "names": [],
        "family_stats": {},
        "error_lessons": {},
        "runs": [],
        "last_batch": None,
        "mutation_stats": {},     # operator -> {tried, wins, gain_sum}: which local changes actually help
        "evolution_log": [],      # parent -> child changes with their measured score deltas
        "freq_bins": {},          # trades/yr bucket -> {n, score_sum}: does trading more pay off?
    }


def load_memory(path: str | Path = DEFAULT_MEMORY_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return _empty_state()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load memory at %s (%s); starting fresh", path, exc)
        return _empty_state()
    if state.get("version") != STATE_VERSION:
        # v1 scored one symbol with a different rubric; its numbers are not
        # comparable to universe scores. Keep the file for reference, start clean.
        backup = path.with_name(f"state_v{state.get('version', 1)}_backup.json")
        try:
            path.replace(backup)
            logger.info("Memory format changed (v%s -> v%d); archived old state to %s", state.get("version"), STATE_VERSION, backup)
        except OSError as exc:
            logger.warning("Could not archive old memory: %s", exc)
        return _empty_state()
    for key, default in _empty_state().items():
        state.setdefault(key, default)
    return state


def save_memory(state: dict, path: str | Path = DEFAULT_MEMORY_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def append_experiment(record: dict, path: str | Path | None = None) -> None:
    path = Path(path) if path else DEFAULT_MEMORY_PATH.with_name("experiments.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# --------------------------------------------------------------------------
# Dedup + names
# --------------------------------------------------------------------------

class _Seen:
    """O(1) membership over the (list-serialised) signature stores."""

    def __init__(self):
        self._cache: dict[tuple[int, str], set] = {}

    def has(self, state: dict, key: str, value: str) -> bool:
        lst = state.get(key, [])
        ck = (id(state), key)
        cached = self._cache.get(ck)
        if cached is None or len(cached) != len(lst):
            cached = set(lst)
            self._cache[ck] = cached
        return value in cached


_seen = _Seen()


def has_signature(state: dict, signature: str) -> bool:
    return _seen.has(state, "signatures", signature)


def has_behavior(state: dict, behavior: str) -> bool:
    return _seen.has(state, "behaviors", behavior)


def name_taken(state: dict, name: str) -> bool:
    return _seen.has(state, "names", name)


def unique_name(state: dict, name: str, extra_taken: set[str] | None = None) -> str:
    taken = extra_taken or set()
    if not name_taken(state, name) and name not in taken:
        return name
    base = re.sub(r"_v\d+$", "", name)
    i = 2
    while name_taken(state, f"{base}_v{i}") or f"{base}_v{i}" in taken:
        i += 1
    return f"{base}_v{i}"


def _remember(state: dict, key: str, value: str | None, cap: int) -> None:
    if not value or _seen.has(state, key, value):
        return
    lst = state.setdefault(key, [])
    lst.append(value)
    if len(lst) > cap:
        del lst[:-cap]


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------

def record_invalid(state: dict, problems: list[str]) -> None:
    """Count (normalised) validation errors so the prompt can warn about the
    most frequent ones."""
    lessons = state.setdefault("error_lessons", {})
    for p in problems:
        key = re.sub(r"\d+(\.\d+)?", "N", str(p))[:110]
        lessons[key] = lessons.get(key, 0) + 1
    if len(lessons) > MAX_LESSONS:
        for k, _ in sorted(lessons.items(), key=lambda kv: kv[1])[: len(lessons) - MAX_LESSONS]:
            del lessons[k]


def record_result(state: dict, *, spec: dict, summary: dict, run_id: str, iteration: int, signature: str,
                  behavior: str | None, accepted: bool, holdout: dict | None = None) -> None:
    """Register one fully evaluated strategy. `spec` is StrategySpec.to_record()."""
    state["total_tried"] = state.get("total_tried", 0) + 1
    if accepted:
        state["total_accepted"] = state.get("total_accepted", 0) + 1
    _remember(state, "signatures", signature, MAX_SIGNATURES)
    _remember(state, "behaviors", behavior, MAX_SIGNATURES)
    _remember(state, "names", spec["name"], MAX_NAMES)

    fam = state.setdefault("family_stats", {}).setdefault(spec.get("family", "other"),
                                                            {"tried": 0, "score_sum": 0.0, "positive": 0, "accepted": 0})
    fam["tried"] += 1
    fam["score_sum"] += float(summary.get("score", 0.0))
    fam["positive"] += 1 if summary.get("score", 0.0) > 0.1 else 0
    fam["accepted"] += 1 if accepted else 0

    tpy = summary.get("trades_per_year")
    if tpy is not None and not summary.get("probe_rejected"):
        for lo, hi, label in FREQ_BINS:
            if lo <= tpy < hi:
                b = state.setdefault("freq_bins", {}).setdefault(label, {"n": 0, "score_sum": 0.0})
                b["n"] += 1
                b["score_sum"] += float(summary.get("score", 0.0))
                break

    entry = {
        **spec,
        "score": summary.get("score", 0.0),
        "fitness": summary.get("fitness", summary.get("score", 0.0)),
        "summary": {k: v for k, v in summary.items() if k not in ("fail_counts",)} | {"fail_counts": summary.get("fail_counts", {})},
        "accepted": accepted,
        "run_id": run_id,
        "iteration": iteration,
        "signature": signature,
        "evolved": 0,
    }
    if holdout:  # for the human; never sent to the LLM
        entry["holdout"] = {k: holdout[k] for k in ("median_sharpe", "mean_sharpe", "pct_positive") if k in holdout}
    if summary.get("score", -9) > 0 and not summary.get("probe_rejected"):
        board = state.setdefault("leaderboard", [])
        board.append(entry)
        board.sort(key=lambda e: -rank_value(e))
        del board[LEADERBOARD_SIZE:]


def record_unevaluated(state: dict, spec: dict, signature: str | None, behavior: str | None = None) -> None:
    """A strategy that compiled but was rejected cheaply (duplicate behaviour,
    no signals): remember its fingerprints so it is not retried."""
    _remember(state, "signatures", signature, MAX_SIGNATURES)
    _remember(state, "behaviors", behavior, MAX_SIGNATURES)
    _remember(state, "names", spec["name"], MAX_NAMES)


def remember_behavior(state: dict, behavior: str | None) -> None:
    _remember(state, "behaviors", behavior, MAX_SIGNATURES)


def decay_lessons(state: dict, factor: float = 0.85) -> None:
    """Age the error counts each iteration so the prompt reflects what the
    model is getting wrong NOW, not a mistake it stopped making long ago."""
    lessons = state.get("error_lessons", {})
    for k in list(lessons):
        lessons[k] *= factor
        if lessons[k] < 0.5:
            del lessons[k]


def record_mutation(state: dict, op: str, tried: int, wins: int, gain_sum: float) -> None:
    st = state.setdefault("mutation_stats", {}).setdefault(op, {"tried": 0, "wins": 0, "gain_sum": 0.0})
    st["tried"] += tried
    st["wins"] += wins
    st["gain_sum"] += gain_sum


def record_evolution(state: dict, parent: str, child: str, kind: str, change: str, d_score: float, d_tpy: float) -> None:
    """`kind` is a mutation operator name or "llm" (a variant the LLM wrote for
    `parent`; `change` is the LLM's own one-line statement of what it changed)."""
    log = state.setdefault("evolution_log", [])
    log.append({"parent": parent, "child": child, "kind": kind, "change": change[:140],
                "d_score": round(d_score, 4), "d_tpy": round(d_tpy, 1)})
    if len(log) > MAX_EVOLUTION_LOG:
        del log[:-MAX_EVOLUTION_LOG]


def record_run(state: dict, run_summary: dict) -> None:
    runs = state.setdefault("runs", [])
    runs.append(run_summary)
    if len(runs) > MAX_RUNS:
        del runs[:-MAX_RUNS]


# --------------------------------------------------------------------------
# Reading (what goes into prompts)
# --------------------------------------------------------------------------

def rank_value(e: dict) -> float:
    """What strategies are ranked by everywhere (leaderboard, parents, top-N):
    fitness (score + pass-rate bonus); falls back to score for older records."""
    return e.get("fitness", e.get("score", 0.0))


def _distinct_by_body(entries: list[dict]) -> list[dict]:
    """Keep the first entry per distinct strategy code: tuned/evolved copies of
    one idea differ only in parameters and would otherwise fill every slot."""
    seen, out = set(), []
    for e in entries:
        key = re.sub(r"\s+", " ", e.get("body", "")).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def best_ever(state: dict, n: int) -> list[dict]:
    ranked = sorted(state.get("leaderboard", []), key=lambda e: -rank_value(e))
    return _distinct_by_body(ranked)[:n]


def pick_parents(state: dict, n: int) -> list[dict]:
    """Parents to improve next: highest score, discounted for how often each
    has already been evolved (so the loop spreads effort instead of
    re-mutating the champion forever)."""
    board = state.get("leaderboard", [])
    ranked = sorted(board, key=lambda e: -(rank_value(e) - 0.07 * e.get("evolved", 0)))
    return _distinct_by_body(ranked)[:n]


def mark_evolved(state: dict, names: list[str]) -> None:
    for e in state.get("leaderboard", []):
        if e["name"] in names:
            e["evolved"] = e.get("evolved", 0) + 1


def recent_names(state: dict, n: int) -> list[str]:
    return state.get("names", [])[-n:]


def top_lessons(state: dict, n: int) -> list[str]:
    items = sorted(state.get("error_lessons", {}).items(), key=lambda kv: -kv[1])[:n]
    return [f"{k} (x{v:.0f})" for k, v in items]


def mutation_table(state: dict, labels: dict[str, str], min_tried: int = 3) -> str:
    """Measured effect of each kind of local change (share of candidates that
    beat their parent, and the mean score change when they did)."""
    rows = []
    for op, st in state.get("mutation_stats", {}).items():
        if st["tried"] < min_tried:
            continue
        rows.append((st["wins"] / st["tried"], f"{labels.get(op, op)}: improved the parent in {st['wins']}/{st['tried']} tries"
                     + (f" (mean gain {st['gain_sum'] / st['wins']:+.3f})" if st["wins"] else "")))
    rows.sort(key=lambda r: -r[0])
    shown = rows if len(rows) <= 8 else rows[:5] + rows[-3:]        # best five and worst three: what to try, what to avoid
    return "; ".join(text for _, text in shown)


def llm_change_examples(state: dict, k: int = 3) -> tuple[list[str], list[str]]:
    """The LLM's own past hypotheses with their measured outcome: best and worst."""
    llm = [e for e in state.get("evolution_log", []) if e["kind"] == "llm"]
    llm.sort(key=lambda e: -e["d_score"])

    def fmt(e: dict) -> str:
        return f"'{e['change']}' -> score {e['d_score']:+.3f}, trades/yr {e['d_tpy']:+.0f}"
    good = [fmt(e) for e in llm[:k] if e["d_score"] > 0.005]
    bad = [fmt(e) for e in llm[::-1][:k] if e["d_score"] < -0.005]
    return good, bad


def lineage(state: dict, name: str, depth: int = 4) -> str:
    """The change history that produced `name` (newest first), e.g.
    'g3 adx_trend +0.021 <- g2 re-tune +0.010 <- original'."""
    by_child = {e["child"]: e for e in state.get("evolution_log", [])}
    steps, cur = [], name
    while cur in by_child and len(steps) < depth:
        e = by_child[cur]
        steps.append(f"{e['kind']} ({e['d_score']:+.3f})" if e["kind"] != "llm" else f"LLM '{e['change'][:60]}' ({e['d_score']:+.3f})")
        cur = e["parent"]
    return " <- ".join(steps) + (" <- original" if steps else "")


def freq_table(state: dict) -> str:
    rows = []
    for lo, hi, label in FREQ_BINS:
        b = state.get("freq_bins", {}).get(label)
        if b and b["n"] >= 4:
            rows.append(f"{label}/yr: mean score {b['score_sum'] / b['n']:+.3f} (n={b['n']})")
    return "; ".join(rows)


def family_table(state: dict) -> str:
    rows = []
    for fam, st in sorted(state.get("family_stats", {}).items(), key=lambda kv: -kv[1]["score_sum"] / max(1, kv[1]["tried"])):
        if st["tried"] < 3:
            continue
        rows.append(f"{fam}: avg score {st['score_sum'] / st['tried']:+.2f}, {st['positive']}/{st['tried']} decent, {st['accepted']} accepted")
    return "; ".join(rows)
