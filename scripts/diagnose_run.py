"""Post-mortem of the strategies a run saved: where does the edge go between IS, OSS1 and OSS2?

    .venv\\Scripts\\python.exe scripts/diagnose_run.py results/<run_id> [--out report.md]

For every saved strategy and each segment it reports, averaged over the universe:
net / gross PnL and costs (% of capital per year), long vs short PnL, Sharpe by asset group,
and PnL by calendar year, next to a buy-and-hold benchmark so a regime change is visible.

It re-evaluates strategies that were ALREADY looked at on OSS2 (see src/final_test.py), so it
adds no new selection pressure, but whatever you learn from the OSS2 columns is test-set
knowledge: use it to understand the failure, not to pick or tune a strategy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtester import build_settings, simulate_arrays          # noqa: E402
from src.config import load_config                                   # noqa: E402
from src.metrics import fast_metrics                                 # noqa: E402
from src.strategy_sandbox import compile_strategy                    # noqa: E402
from src.universe import compute_signal_arrays, load_universe       # noqa: E402


def segments(sd) -> dict[str, tuple[int, int]]:
    return {"IS": (0, sd.oss1[0]), "OSS1": sd.oss1, "OSS2": sd.oss2}


def run_segment(compiled, sd, sigs, cfg, seg) -> dict:
    a, b = seg
    le, lx, se, sx = sigs
    settings = build_settings(cfg, sd.instrument, compiled, sd.initial_capital)
    equity, nt, ent, ext, tdir, _c, _e, _x, pnl, cost, _r = simulate_arrays(
        sd.open_[a:b], sd.high[a:b], sd.low[a:b], sd.close[a:b], le[a:b], lx[a:b], se[a:b], sx[a:b], settings)
    m = fast_metrics(equity, pnl[:nt], 252)
    years = (b - a) / 252.0
    cap = settings.initial_capital
    idx = sd.df.index[a:b]
    exit_year = idx[ext[:nt]].year if nt else np.array([], dtype=int)
    by_year = pd.Series(pnl[:nt]).groupby(exit_year).sum() / cap * 100.0 if nt else pd.Series(dtype=float)
    px = sd.close[a:b]
    return {
        "sharpe": float(m["sharpe"]), "trades": int(nt), "years": years,
        "net": float(pnl[:nt].sum() / cap * 100.0 / years), "cost": float(cost[:nt].sum() / cap * 100.0 / years),
        "long": float(pnl[:nt][tdir[:nt] == 1].sum() / cap * 100.0 / years),
        "short": float(pnl[:nt][tdir[:nt] == -1].sum() / cap * 100.0 / years),
        "by_year": by_year,
        "bh": float((px[-1] / px[0] - 1.0) * 100.0 / years) if px[0] > 0 else float("nan"),
        "win": float((pnl[:nt] > 0).mean()) if nt else 0.0,
        "hold": float((ext[:nt] - ent[:nt]).mean()) if nt else 0.0,
    }


def fmt_table(df: pd.DataFrame, floatfmt="{:+.2f}") -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for k, row in df.iterrows():
        lines.append("| " + " | ".join([str(k)] + [floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else "" if isinstance(v, (float, np.floating)) else str(v) for v in row]) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    universe = load_universe(cfg)
    run_dir = Path(args.run_dir)
    files = sorted(run_dir.glob("strategies/*/strat_gen_*.py"))
    out: list[str] = [f"# Diagnosis of {run_dir.name}", "",
                      f"{len(universe)} symbols; per-year numbers are % of capital, averaged over symbols.", ""]

    # benchmark: buy & hold per segment per group
    bh_rows = {}
    for sd in universe:
        for name, (a, b) in segments(sd).items():
            px = sd.close[a:b]
            bh_rows.setdefault(name, []).append((sd.group, (px[-1] / px[0] - 1.0) * 100.0 / ((b - a) / 252.0) if px[0] > 0 else np.nan))
    bh = pd.DataFrame({n: pd.Series([v for _, v in rows], index=[g for g, _ in rows]).groupby(level=0).mean() for n, rows in bh_rows.items()})
    bh_sh = {}
    for sd in universe:
        for name, (a, b) in segments(sd).items():
            r = np.diff(sd.close[a:b]) / sd.close[a:b][:-1]
            r = r[np.isfinite(r)]
            bh_sh.setdefault(name, []).append(float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 1 and r.std() > 0 else 0.0)
    out += ["## Always-long benchmark (what 'long-only beta' alone earns)", "",
            fmt_table(pd.DataFrame({n: {"mean Sharpe (clipped)": float(np.clip(v, -1.5, 1.5).mean()), "% symbols positive": float(np.mean(np.array(v) > 0))}
                                    for n, v in bh_sh.items()}).rename_axis("buy & hold")), ""]
    out += ["## Buy-and-hold drift by group (% per year, unleveraged, for scale only)", "", fmt_table(bh.rename_axis("group")), ""]

    seen: dict[str, str] = {}
    all_rows = []
    year_tables: dict[str, pd.DataFrame] = {}
    for f in files:
        compiled = compile_strategy(f.read_text(encoding="utf-8"))
        if compiled.signature in seen:
            out.append(f"(`{f.parent.name}` has the same signature as `{seen[compiled.signature]}`; skipped)\n")
            continue
        seen[compiled.signature] = f.parent.name
        per: dict[str, list[dict]] = {k: [] for k in ("IS", "OSS1", "OSS2")}
        for sd in universe:
            sigs = compute_signal_arrays(compiled, sd)
            for name, seg in segments(sd).items():
                r = run_segment(compiled, sd, sigs, cfg, seg)
                r.update(symbol=sd.symbol, group=sd.group)
                per[name].append(r)
        name = f.parent.name
        for seg, rows in per.items():
            d = pd.DataFrame(rows)
            all_rows.append({"strategy": name, "segment": seg,
                             "mean_sharpe": float(np.clip(d["sharpe"], -1.5, 1.5).mean()), "pct_pos": float((d["sharpe"] > 0).mean()),
                             "net_%/yr": d["net"].mean(), "gross_%/yr": (d["net"] + d["cost"]).mean(), "cost_%/yr": d["cost"].mean(),
                             "long_%/yr": d["long"].mean(), "short_%/yr": d["short"].mean(),
                             "trades/yr": (d["trades"] / d["years"]).mean(), "win": d["win"].mean(), "hold": d["hold"].mean()})
        # group x segment
        g = pd.concat({seg: pd.DataFrame(rows).groupby("group")["sharpe"].apply(lambda s: np.clip(s, -1.5, 1.5).mean())
                       for seg, rows in per.items()}, axis=1)
        out += [f"## {name}", "", "Mean Sharpe by asset group:", "", fmt_table(g.rename_axis("group")), ""]
        recs = [(r["symbol"], int(y), float(v)) for rows in per.values() for r in rows for y, v in r["by_year"].items()]
        year_tables[name] = pd.DataFrame(recs, columns=["sym", "year", "pnl"]).pivot_table(
            index="year", columns="sym", values="pnl", aggfunc="sum")

    summ = pd.DataFrame(all_rows)
    out += ["## Summary by strategy and segment", "",
            fmt_table(summ.set_index(["strategy", "segment"]).round(3).reset_index().assign(
                strategy=lambda d: d["strategy"].str.slice(0, 44)).set_index("strategy")), ""]
    seg_mean = summ.groupby("segment")[["mean_sharpe", "pct_pos", "net_%/yr", "gross_%/yr", "cost_%/yr", "long_%/yr", "short_%/yr", "trades/yr", "win", "hold"]].mean()
    out += ["## Average over the distinct strategies", "", fmt_table(seg_mean.rename_axis("segment")), ""]

    if year_tables:
        yearly = pd.concat([t.mean(axis=1).rename(n) for n, t in year_tables.items()], axis=1)
        yearly["MEAN"] = yearly.mean(axis=1)
        out += ["## Mean PnL by calendar year (% of capital, avg over symbols, avg over strategies in last column)", "",
                fmt_table(yearly.rename_axis("year")), ""]

    text = "\n".join(out)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
