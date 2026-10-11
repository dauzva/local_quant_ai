"""The ONLY place OSS2 (the test set) is ever evaluated.

`python main.py run` / `tune` / `backtest-strategy` never read OSS2. It is touched here,
once you have finished iterating and picked the strategy (or strategies) you would
actually trade: either by hand (`final-test`) or as the last step of `pipeline`
(`run_final_stage`: every strategy the run saved, each tested once). Like a test set in machine learning, it only stays an unbiased estimate
while nothing is chosen based on it: if you test, dislike the result, go back, tweak and
test again, OSS2 has quietly become a second validation set. So every look is appended
to a ledger and the previous looks are printed before the next one.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src import insights
from src.report import append_oss2
from src.strategy_sandbox import StrategySandboxError, compile_strategy
from src.universe import (build_eval_settings, evaluate_universe, load_universe, oss1_verdict, run_universe_oss1,
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


def log_look(path: Path, compiled, file: str, n_symbols: int, is_score, oss1: dict, oss2: dict) -> dict:
    """Append one OSS2 look to the ledger."""
    entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "strategy": compiled.strategy_name, "signature": compiled.signature,
             "file": str(file), "n_symbols": n_symbols, "is_score": is_score,
             "oss1_mean_sharpe": oss1["mean_sharpe"], "oss1_pct_positive": oss1["pct_positive"],
             "oss2_mean_sharpe": oss2["mean_sharpe"], "oss2_median_sharpe": oss2["median_sharpe"],
             "oss2_pct_positive": oss2["pct_positive"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def run_final_stage(run, top: list[dict]) -> list[dict]:
    """The last step of `pipeline`: test every saved strategy on OSS2 exactly once, write the
    result into its strategies/<name>/ folder (report.md, metadata.json, per_symbol.csv/png),
    log each look, write results/<run>/oss2_results.csv and print the table. A strategy whose
    code was already tested in an earlier run is not tested again (its old numbers are shown)."""
    cfg = run.cfg
    path = ledger_path(cfg)
    ledger = read_ledger(path)
    seen = {e["signature"]: e for e in ledger}
    accept_cfg = (cfg.get("universe", {}) or {}).get("accept", {}) or {}
    print("\n" + "=" * 100)
    print(f"FINAL TEST on OSS2 (everything after {(cfg.get('split', {}) or {}).get('oss1_end', 'the OSS1 fraction')}): "
          f"{len(top)} strategies, each evaluated once")
    if ledger:
        print(f"WARNING: OSS2 was looked at {len(ledger)} time(s) in earlier runs ({path}). Picking the best of several "
              "pipeline runs by their OSS2 result turns the test set into a validation set.")

    rows = []
    for rank, t in enumerate(top, start=1):
        spec, compiled, result, _model = run.kept[t["name"]]
        oss1 = run.oss1_of[t["name"]]
        prev = seen.get(compiled.signature)
        row = {"rank": rank, "name": t["name"], "is_score": t["score"], "oss1_mean_sharpe": oss1["mean_sharpe"],
               "oss1_pct_positive": oss1["pct_positive"], "accepted": t["accepted"], "dir": t["dir"]}
        if prev:
            row.update(oss2_mean_sharpe=prev["oss2_mean_sharpe"], oss2_median_sharpe=prev["oss2_median_sharpe"],
                       oss2_pct_positive=prev["oss2_pct_positive"], verdict=f"already tested on {prev['timestamp']}; not re-evaluated")
        else:
            oss2 = run_universe_oss2(compiled, run.universe, run.ev, cfg)
            ok, why = oss1_verdict(oss2, accept_cfg)
            verdict = "holds up on the test set (clears the OSS1 bar)" if ok else f"BELOW the OSS1 bar: {why.replace('OSS1', 'OSS2')}"
            append_oss2(Path(t["dir"]), spec.name, oss2, verdict)
            log_look(path, compiled, t["dir"], len(run.universe), t["score"], oss1, oss2)
            row.update(oss2_mean_sharpe=oss2["mean_sharpe"], oss2_median_sharpe=oss2["median_sharpe"],
                       oss2_pct_positive=oss2["pct_positive"], verdict=verdict)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(run.dir / "oss2_results.csv", index=False)
    pd.DataFrame(top).merge(df[["name", "oss2_mean_sharpe", "oss2_median_sharpe", "oss2_pct_positive", "verdict"]], on="name",
                            how="left").to_csv(run.dir / "top_strategies.csv", index=False)

    print(f"\n{'#':>2s} {'strategy':36s} {'IS score':>9s} {'OSS1 mean':>10s} {'OSS1 +%':>8s} {'OSS2 mean':>10s} {'OSS2 med':>9s} {'OSS2 +%':>8s}  verdict")
    for r in rows:
        print(f"{r['rank']:2d} {r['name'][:36]:36s} {r['is_score']:+9.3f} {r['oss1_mean_sharpe']:+10.2f} {r['oss1_pct_positive']:8.0%} "
              f"{r['oss2_mean_sharpe']:+10.2f} {r['oss2_median_sharpe']:+9.2f} {r['oss2_pct_positive']:8.0%}  {r['verdict']}")
    held = sum(1 for r in rows if r["verdict"].startswith("holds up"))
    print(f"\n{held}/{len(rows)} strategies hold up on OSS2. Mean Sharpe across them: IS score {df['is_score'].mean():+.3f} -> "
          f"OSS1 {df['oss1_mean_sharpe'].mean():+.3f} -> OSS2 {df['oss2_mean_sharpe'].mean():+.3f} "
          "(part of the drop from OSS1 to OSS2 is selection bias from choosing on OSS1, part is a different market regime).")
    print("OSS2 is now spent for these strategies. Do not tune them further based on these numbers.")
    print(f"Per-strategy files: {run.dir / 'strategies'}  |  table: {run.dir / 'oss2_results.csv'}")
    print("=" * 100)
    return rows


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

    return log_look(path, compiled, strategy_path, len(universe), s.get("score"), oss1, oss2)
