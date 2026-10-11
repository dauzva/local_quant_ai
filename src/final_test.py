"""The ONLY place OSS2 (the test set) is ever evaluated.

`python main.py run` / `tune` / `backtest-strategy` never read OSS2. It is touched here,
once you have finished iterating and picked the strategy (or strategies) you would
actually trade. Like a test set in machine learning, it only stays an unbiased estimate
while nothing is chosen based on it: if you test, dislike the result, go back, tweak and
test again, OSS2 has quietly become a second validation set. So every look is appended
to a ledger and the previous looks are printed before the next one.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src import insights
from src.strategy_sandbox import StrategySandboxError, compile_strategy
from src.universe import (build_eval_settings, evaluate_universe, load_universe, run_universe_oss1,
                          run_universe_oss2)
from src.utils import get_logger

logger = get_logger("final_test")

LEDGER_NAME = "oss2_ledger.jsonl"


def ledger_path(cfg) -> Path:
    return Path(cfg.get("memory", {}).get("path", "results/memory/state.json")).parent / LEDGER_NAME


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def run_final_test(strategy_path: str, cfg, symbols: list[str] | None = None, confirm: bool = False) -> dict | None:
    """Evaluate one saved strategy on IS, OSS1 and OSS2 and log the OSS2 look.
    Without `confirm` nothing is evaluated: the command only explains itself and shows
    the ledger, so OSS2 cannot be burned by a mistyped command."""
    try:
        compiled = compile_strategy(Path(strategy_path).read_text(encoding="utf-8"))
    except StrategySandboxError as exc:
        logger.error("Strategy file failed validation: %s", exc.problems)
        return None

    path = ledger_path(cfg)
    ledger = read_ledger(path)
    same = [e for e in ledger if e.get("signature") == compiled.signature]
    split = cfg.get("split", {}) or {}
    print(f"\nOSS2 is the TEST set (everything after {split.get('oss1_end') or 'the OSS1 fraction'}).")
    print(f"It has been looked at {len(ledger)} time(s) before ({len(same)} time(s) for this exact strategy) -- ledger: {path}")
    for e in ledger[-5:]:
        print(f"  {e['timestamp']}  {e['strategy']:<36s} OSS2 mean Sharpe {e['oss2_mean_sharpe']:+.2f}, {e['oss2_pct_positive']:.0%} profitable")
    if not confirm:
        print("\nNothing was evaluated. Use this only when you are DONE iterating and are not going to change anything\n"
              "based on the number you see. If you then tweak and re-test, OSS2 is no longer an unbiased test.\n"
              "To evaluate and log the look, add --confirm.")
        return None
    if same:
        print("\nWARNING: this exact strategy was already tested on OSS2; a repeat look tells you nothing new.")

    universe = load_universe(cfg, symbols)
    ev = build_eval_settings(cfg)
    is_result = evaluate_universe(compiled, universe, ev, cfg, probe=None)
    oss1 = run_universe_oss1(compiled, universe, ev, cfg)
    oss2 = run_universe_oss2(compiled, universe, ev, cfg)

    is_by_sym = {r.symbol: r for r in is_result.per_symbol}
    o1 = {r["symbol"]: r for r in oss1["rows"]}
    o2 = {r["symbol"]: r for r in oss2["rows"]}
    s = is_result.summary
    print(f"\n{compiled.strategy_name}\nIS   : {insights.results_line(s)}")
    print(f"\n{'symbol':14s} {'group':13s} {'IS sharpe':>10s} {'OSS1':>7s} {'OSS2':>7s} {'OSS2 trades':>12s}")
    for sym in sorted(o2, key=lambda k: -o2[k]["sharpe"]):
        r = is_by_sym.get(sym)
        print(f"{sym:14s} {o2[sym]['group']:13s} {r.avg_sharpe if r else float('nan'):10.2f} {o1[sym]['sharpe']:7.2f} "
              f"{o2[sym]['sharpe']:7.2f} {o2[sym]['trades']:12d}")
    print(f"\nIS   score {s.get('score', 0):+.3f} | {s.get('pct_positive', 0):.0%} of symbols profitable")
    print(f"OSS1 mean (clipped) Sharpe {oss1['mean_sharpe']:+.2f} | median {oss1['median_sharpe']:+.2f} | {oss1['pct_positive']:.0%} profitable")
    print(f"OSS2 mean (clipped) Sharpe {oss2['mean_sharpe']:+.2f} | median {oss2['median_sharpe']:+.2f} | {oss2['pct_positive']:.0%} profitable")

    entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "strategy": compiled.strategy_name, "signature": compiled.signature,
             "file": str(strategy_path), "n_symbols": len(universe),
             "is_score": s.get("score"), "oss1_mean_sharpe": oss1["mean_sharpe"], "oss1_pct_positive": oss1["pct_positive"],
             "oss2_mean_sharpe": oss2["mean_sharpe"], "oss2_median_sharpe": oss2["median_sharpe"],
             "oss2_pct_positive": oss2["pct_positive"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry
