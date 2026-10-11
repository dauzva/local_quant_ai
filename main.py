"""local_quant_ai CLI.

Usage:
    python main.py inspect-data
    python main.py inspect-data --directory ./data
    python main.py inspect-data --symbol "@TY"
    python main.py inspect-data --file ./data/A_OHLCV_@TY_minutes_1440.dat
    python main.py list-free-models
    python main.py pipeline --generations 10   # THE WHOLE WORKFLOW, one command: generate -> validate on OSS1 ->
                                               # adjust -> validate again ... for N generations, save the best strategies,
                                               # then test them on OSS2 exactly once and print the results
    python main.py run                       # generate + validate on the whole universe (stops before OSS2)
    python main.py run --iterations 0 --minutes 120   # auto mode: until time/quota is up
    python main.py run --symbols "@TY,@ES"   # restrict the universe
    python main.py tune --minutes 30         # offline (no LLM): local evolutionary search on the leaderboard
    python main.py backtest-strategy --file path/to/strategy.py   # IS + OSS1 only
    python main.py final-test --file path/to/strategy.py --confirm  # OSS2 (test set): once, at the very end
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.utils import setup_logging, get_logger

logger = get_logger("main")


def cmd_inspect_data(args, cfg) -> None:
    # Delegate to the standalone script so behavior stays identical either way.
    script_path = Path(__file__).parent / "scripts" / "inspect_dat.py"
    sys.argv = ["inspect_dat.py"]
    if args.directory:
        sys.argv += ["--directory", args.directory]
    if args.symbol:
        sys.argv += ["--symbol", args.symbol]
    if args.file:
        sys.argv += ["--file", args.file]
    sys.argv += ["--config", args.config]

    import importlib.util
    spec = importlib.util.spec_from_file_location("inspect_dat", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def cmd_list_free_models(args, cfg) -> None:
    from src.llm import OpenRouterClient, rank_models_for_codegen
    client = OpenRouterClient(cfg)
    models = client.list_free_models(max_models=cfg.llm.get("max_free_models"))
    if not models:
        print("No free models found (check OPENROUTER_API_KEY and network access).")
        return
    ranked_ids = [m.get("id") for m in rank_models_for_codegen(models)]
    print(f"{'model_id':60s} {'context_length':>15s} {'prompt_price':>15s} {'completion_price':>18s}  codegen_rank")
    for m in models:
        pricing = m.get("pricing", {}) or {}
        mid = m.get("id", "")
        rank = ranked_ids.index(mid) + 1 if mid in ranked_ids else "-"
        print(f"{mid:60s} {str(m.get('context_length', '')):>15s} "
              f"{str(pricing.get('prompt', '')):>15s} {str(pricing.get('completion', '')):>18s}  {rank}")
    print("\ncodegen_rank: this project's preference order for strategy-code generation")
    print("(lower is better; '-' means excluded -- non-text/vision/audio/niche-tuned model).")


def _split_symbols(raw: str | None) -> list[str] | None:
    return [s.strip() for s in raw.split(",") if s.strip()] if raw else None


def cmd_run(args, cfg) -> None:
    from src.orchestrator import run_research_loop
    run_id = run_research_loop(
        cfg,
        symbols=_split_symbols(args.symbols),
        iterations=args.iterations,
        minutes=args.minutes,
    )
    print(f"Run complete: {run_id}")
    print(f"Results saved under results/{run_id}/")


def cmd_pipeline(args, cfg) -> None:
    """The full workflow in one command. Stop it whenever you are done with Ctrl-C: it then
    saves the best strategies and runs the one-shot OSS2 test, exactly as at the end of a
    normal run. `--no-final-test` skips OSS2 (the strategies are still saved)."""
    from src.orchestrator import run_research_loop
    run_id = run_research_loop(
        cfg,
        symbols=_split_symbols(args.symbols),
        iterations=args.generations,
        minutes=args.minutes,
        offline=args.offline,
        final_test=not args.no_final_test,
    )
    print(f"Pipeline complete: {run_id}")
    print(f"Best strategies (code, report, per-symbol results incl. OSS2): results/{run_id}/strategies/")
    print(f"OSS2 table: results/{run_id}/oss2_results.csv" if not args.no_final_test else "OSS2 was NOT evaluated (--no-final-test).")


def cmd_tune(args, cfg) -> None:
    """Offline mode: no LLM calls. Seed templates + local mutation search over
    the leaderboard stored in results/memory, for as long as you let it run."""
    from src.orchestrator import run_research_loop
    run_id = run_research_loop(cfg, symbols=_split_symbols(args.symbols), iterations=args.iterations,
                               minutes=args.minutes, offline=True)
    print(f"Tuning complete: {run_id}")
    print(f"Top strategies saved under results/{run_id}/strategies/")


def cmd_backtest_strategy(args, cfg) -> None:
    from src.orchestrator import backtest_single_strategy_file
    backtest_single_strategy_file(args.file, cfg, symbols=_split_symbols(args.symbols))


def cmd_final_test(args, cfg) -> None:
    from src.final_test import run_final_test
    run_final_test(args.file, cfg, symbols=_split_symbols(args.symbols), confirm=args.confirm)


def cmd_export_strategy(args, cfg) -> None:
    from src.strategy_export import export_strategy_file
    out_path = export_strategy_file(args.file, output_path=args.out)
    print(f"Exported to: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="local_quant_ai: local futures strategy research loop")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect-data", help="Inspect discovered TradeStation .dat files")
    p_inspect.add_argument("--directory", type=str, default=None)
    p_inspect.add_argument("--symbol", type=str, default=None)
    p_inspect.add_argument("--file", type=str, default=None)
    p_inspect.set_defaults(func=cmd_inspect_data)

    p_models = sub.add_parser("list-free-models", help="List free OpenRouter models")
    p_models.set_defaults(func=cmd_list_free_models)

    p_run = sub.add_parser("run", help="Run the universe strategy research loop")
    p_run.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (default: the whole universe from config.yaml)")
    p_run.add_argument("--iterations", type=int, default=None, help="0 = run until --minutes / quota / budget runs out")
    p_run.add_argument("--minutes", type=float, default=None, help="Wall-clock limit for the run")
    p_run.set_defaults(func=cmd_run)

    p_pipe = sub.add_parser("pipeline", help="Whole workflow in one command: generations of generate/validate(OSS1)/adjust, "
                                             "save the best strategies, then ONE test on OSS2 with printed results")
    p_pipe.add_argument("--generations", type=int, default=None, help="Number of generations (default: run.iterations; 0 = until --minutes / quota / Ctrl-C)")
    p_pipe.add_argument("--minutes", type=float, default=None, help="Wall-clock limit for the generation loop")
    p_pipe.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (default: the whole universe)")
    p_pipe.add_argument("--offline", action="store_true", help="No LLM: seed templates + local search only")
    p_pipe.add_argument("--no-final-test", action="store_true", help="Stop after saving the strategies; do not touch OSS2")
    p_pipe.set_defaults(func=cmd_pipeline)

    p_tune = sub.add_parser("tune", help="Offline: no LLM. Seed templates + local mutation search over the stored leaderboard")
    p_tune.add_argument("--symbols", type=str, default=None)
    p_tune.add_argument("--iterations", type=int, default=0, help="0 = until --minutes")
    p_tune.add_argument("--minutes", type=float, default=15.0, help="Wall-clock limit (default 15)")
    p_tune.set_defaults(func=cmd_tune)

    p_bt = sub.add_parser("backtest-strategy", help="Run one saved strategy .py file across the universe (IS + OSS1; never OSS2)")
    p_bt.add_argument("--file", type=str, required=True, help="Path to a strategy .py file (e.g. results/<run_id>/strategies/<name>/strat_gen_<name>.py)")
    p_bt.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (default: whole universe)")
    p_bt.set_defaults(func=cmd_backtest_strategy)

    p_ft = sub.add_parser("final-test", help="Evaluate a strategy on OSS2, the untouched TEST set. Use once, at the very end; every look is logged")
    p_ft.add_argument("--file", type=str, required=True, help="Path to a strategy .py file")
    p_ft.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (default: whole universe)")
    p_ft.add_argument("--confirm", action="store_true", help="Actually evaluate OSS2 and log the look (without it, only the ledger is shown)")
    p_ft.set_defaults(func=cmd_final_test)

    p_export = sub.add_parser("export-strategy", help="[legacy] Export an OLD JSON-format strategy to a standalone Python class. "
                                                        "Not needed for strategies generated by the current pipeline -- those are already standalone .py files.")
    p_export.add_argument("--file", type=str, required=True, help="Path to a legacy strategy JSON (e.g. metadata.json from an old run)")
    p_export.add_argument("--out", type=str, default=None, help="Output .py path (default: alongside input file)")
    p_export.set_defaults(func=cmd_export_strategy)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.get("level", "INFO"), cfg.logging.get("file"))

    args.func(args, cfg)


if __name__ == "__main__":
    main()
