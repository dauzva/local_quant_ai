"""Inspect TradeStation .dat OHLCV files.

Usage:
    python scripts/inspect_dat.py
    python scripts/inspect_dat.py --directory ./data
    python scripts/inspect_dat.py --symbol "@TY"
    python scripts/inspect_dat.py --file ./data/A_OHLCV_@TY_minutes_1440.dat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import load_config
from src.data_loader import discover_dat_files, parse_filename, read_trade_station_dat
from src.utils import setup_logging


def inspect_file(file_path: Path, cfg) -> None:
    filename_meta = parse_filename(file_path, cfg.data.filename_pattern)
    df, meta = read_trade_station_dat(file_path, filename_meta=filename_meta, cfg=cfg)

    print("=" * 70)
    print(f"FILE: {file_path}")
    print("=" * 70)

    print("\n--- Filename metadata ---")
    if filename_meta:
        print(f"parsed_symbol      : {filename_meta.parsed_symbol}")
        print(f"parsed_minutes      : {filename_meta.parsed_minutes}")
        print(f"inferred_timeframe  : {filename_meta.inferred_timeframe}")
    else:
        print("Filename did not match the configured pattern.")

    print("\n--- Binary metadata ---")
    for field in ("country", "exchange", "symbol", "description", "itype", "ispan",
                  "timezone", "session", "tick", "bpv", "resolved_symbol", "symbol_mismatch"):
        print(f"{field:20s}: {getattr(meta, field)}")

    print(f"\nrecord_count        : {meta.record_count}")
    print(f"start_datetime      : {meta.start_datetime}")
    print(f"end_datetime        : {meta.end_datetime}")

    print("\n--- First 20 rows ---")
    print(df.head(20).to_string())

    print("\n--- Last 20 rows ---")
    print(df.tail(20).to_string())

    print("\n--- Missing value counts ---")
    print(df.isna().sum().to_string())

    print("\n--- Basic OHLCV statistics ---")
    print(df.describe().to_string())

    is_daily = meta.parsed_minutes_from_filename == 1440
    print(f"\nAppears to be daily futures data: {is_daily}")
    if meta.parsed_minutes_from_filename != 1440:
        print(f"WARNING: minutes != 1440 (got {meta.parsed_minutes_from_filename})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect TradeStation .dat OHLCV files")
    parser.add_argument("--directory", type=str, default=None, help="Override data directory")
    parser.add_argument("--symbol", type=str, default=None, help="Inspect only this symbol")
    parser.add_argument("--file", type=str, default=None, help="Inspect a specific .dat file directly")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.get("level", "INFO"), cfg.logging.get("file"))

    if args.directory:
        cfg.data["directory"] = args.directory

    if args.file:
        inspect_file(Path(args.file), cfg)
        return

    file_map = discover_dat_files(cfg)
    if args.symbol:
        if args.symbol not in file_map:
            print(f"Symbol {args.symbol!r} not found. Available: {sorted(file_map.keys())}")
            return
        inspect_file(Path(file_map[args.symbol].file_path), cfg)
        return

    if not file_map:
        print(f"No .dat files discovered in {cfg.data.directory}")
        return

    for symbol, meta in file_map.items():
        inspect_file(Path(meta.file_path), cfg)


if __name__ == "__main__":
    main()
