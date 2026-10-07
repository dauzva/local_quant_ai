"""local_quant_ai CLI.

Usage:
    python main.py inspect-data
    python main.py inspect-data --directory ./data
    python main.py inspect-data --symbol "@TY"
    python main.py inspect-data --file ./data/A_OHLCV_@TY_minutes_1440.dat
    python main.py list-free-models
    python main.py run                       # generate + test on the whole universe
    python main.py run --iterations 0 --minutes 120   # auto mode: until time/quota is up
    python main.py run --symbols "@TY,@ES"   # restrict the universe
    python main.py tune --minutes 30         # offline (no LLM): local evolutionary search on the leaderboard
    python main.py backtest-strategy --file path/to/strategy.py
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

    p_tune = sub.add_parser("tune", help="Offline: no LLM. Seed templates + local mutation search over the stored leaderboard")
    p_tune.add_argument("--symbols", type=str, default=None)
    p_tune.add_argument("--iterations", type=int, default=0, help="0 = until --minutes")
    p_tune.add_argument("--minutes", type=float, default=15.0, help="Wall-clock limit (default 15)")
    p_tune.set_defaults(func=cmd_tune)

    p_bt = sub.add_parser("backtest-strategy", help="Run one saved strategy .py file across the universe")
    p_bt.add_argument("--file", type=str, required=True, help="Path to a strategy .py file (e.g. results/<run_id>/strategies/<name>/strat_gen_<name>.py)")
    p_bt.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (default: whole universe)")
    p_bt.set_defaults(func=cmd_backtest_strategy)

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
