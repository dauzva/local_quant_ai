"""Discover TradeStation .dat OHLCV files and load them into clean DataFrames.

Filename convention: A_OHLCV_{symbol}_minutes_{minutes}.dat
Example:             A_OHLCV_@TY_minutes_1440.dat  (minutes_1440 == daily bars)

The binary format itself (struct layout, epoch conversion) is NOT altered
from the reference parser supplied in the project spec.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("data_loader")

TS_MAGIC = 1111111111
TS_TYPE_FORMAT = 3


@dataclass
class FileNameMetadata:
    file_path: str
    parsed_symbol: str
    parsed_minutes: int
    inferred_timeframe: str


@dataclass
class DatMetadata:
    file_path: str
    parsed_symbol_from_filename: str
    parsed_minutes_from_filename: int
    inferred_timeframe_from_filename: str
    country: str
    exchange: str
    symbol: str
    description: str
    itype: str
    ispan: int
    timezone: str
    session: str
    tick: float
    bpv: float
    record_count: int
    start_datetime: Any
    end_datetime: Any
    resolved_symbol: str = ""
    symbol_mismatch: bool = False


def infer_timeframe_from_minutes(minutes: int) -> str:
    if minutes == 1440:
        return "1d"
    if minutes <= 0:
        return "unknown"
    return f"{minutes}m"


def parse_filename(file_path: Path, pattern: str) -> FileNameMetadata | None:
    """Parse a .dat filename against the configured regex pattern."""
    match = re.match(pattern, file_path.name)
    if not match:
        return None
    groups = match.groupdict()
    symbol = groups.get("symbol", "")
    try:
        minutes = int(groups.get("minutes", "0"))
    except ValueError:
        minutes = 0
    return FileNameMetadata(
        file_path=str(file_path),
        parsed_symbol=symbol,
        parsed_minutes=minutes,
        inferred_timeframe=infer_timeframe_from_minutes(minutes),
    )


def discover_dat_files(cfg) -> dict[str, FileNameMetadata]:
    """Build a symbol -> FileNameMetadata mapping.

    Priority:
    1. Explicit config.data.file_map entries (if present).
    2. Auto-discovery by scanning config.data.directory for files matching
       config.data.filename_pattern.

    Files whose parsed minutes != expected_minutes are skipped (with a
    warning) unless allow_other_minutes is true. If multiple files map to
    the same symbol, one is chosen deterministically (sorted filename order)
    and a warning is logged.
    """
    data_dir = Path(cfg.data.directory)
    pattern = cfg.data.filename_pattern
    expected_minutes = cfg.data.get("expected_minutes")
    allow_other_minutes = bool(cfg.data.get("allow_other_minutes", False))

    result: dict[str, FileNameMetadata] = {}

    file_map = cfg.data.get("file_map") or {}
    if file_map:
        for symbol, filename in file_map.items():
            fpath = data_dir / filename
            if not fpath.exists():
                logger.warning("file_map entry for %s points to missing file: %s", symbol, fpath)
                continue
            meta = parse_filename(fpath, pattern)
            if meta is None:
                logger.warning("file_map file %s does not match filename_pattern; using symbol from config", fpath)
                meta = FileNameMetadata(
                    file_path=str(fpath),
                    parsed_symbol=symbol,
                    parsed_minutes=expected_minutes or 1440,
                    inferred_timeframe=infer_timeframe_from_minutes(expected_minutes or 1440),
                )
            if expected_minutes is not None and meta.parsed_minutes != expected_minutes and not allow_other_minutes:
                logger.warning(
                    "Skipping %s: parsed_minutes=%s != expected_minutes=%s",
                    fpath, meta.parsed_minutes, expected_minutes,
                )
                continue
            result[symbol] = meta
        if result:
            return result
        logger.warning("file_map produced no usable entries; falling back to auto-discovery")

    if not data_dir.exists():
        logger.warning("Data directory does not exist: %s", data_dir)
        return result

    candidates: dict[str, list[FileNameMetadata]] = {}
    for fpath in sorted(data_dir.glob("*.dat")):
        meta = parse_filename(fpath, pattern)
        if meta is None:
            logger.warning("Skipping unrecognized filename: %s", fpath.name)
            continue
        if expected_minutes is not None and meta.parsed_minutes != expected_minutes and not allow_other_minutes:
            logger.warning(
                "Skipping %s: parsed_minutes=%s != expected_minutes=%s",
                fpath.name, meta.parsed_minutes, expected_minutes,
            )
            continue
        candidates.setdefault(meta.parsed_symbol, []).append(meta)

    for symbol, metas in candidates.items():
        if len(metas) > 1:
            logger.warning(
                "Multiple files found for symbol %s: %s. Using %s (deterministic: sorted first).",
                symbol, [m.file_path for m in metas], metas[0].file_path,
            )
        result[symbol] = metas[0]

    return result


def _read_str(fh) -> str:
    size = struct.unpack("i", fh.read(4))[0]
    raw = fh.read(size)
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError:
        logger.warning("ASCII decode failed for a header string field; retrying with latin-1")
        return raw.decode("latin-1")


def load_dat(file_path: Path) -> pd.DataFrame:
    """Parse a TradeStation binary .dat file into a raw OHLCV DataFrame.

    This is the canonical binary reader: header layout and the date/epoch
    conversion (`(date - 25569) * 86400` seconds since Unix epoch) must not
    be changed, since it matches TradeStation's own OLE-automation-date
    convention (epoch = 1899-12-30).
    """
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        header = f.read(8)
        if len(header) < 8:
            logger.warning("File %s is empty or truncated header; returning empty DataFrame", file_path)
            return _empty_ohlcv()

        ones, type_format = struct.unpack("ii", header)
        if ones != TS_MAGIC or type_format != TS_TYPE_FORMAT:
            raise ValueError(f"Unsupported .dat format in {file_path}")

        _tick = struct.unpack("d", f.read(8))[0]
        _bpv = struct.unpack("d", f.read(8))[0]

        _country = _read_str(f)
        _exchange = _read_str(f)
        _symbol = _read_str(f)
        _desc = _read_str(f)
        _itype = _read_str(f)
        _ispan = struct.unpack("i", f.read(4))[0]
        _tz = _read_str(f)
        _session = _read_str(f)

        dtype = np.dtype(
            [
                ("date", "f8"),
                ("open", "f8"),
                ("high", "f8"),
                ("low", "f8"),
                ("close", "f8"),
                ("volume", "f8"),
            ]
        )

        remaining = f.read()
        n_complete = len(remaining) // dtype.itemsize
        used_bytes = n_complete * dtype.itemsize
        if used_bytes != len(remaining):
            logger.warning(
                "%s has %d trailing bytes not divisible by record size; ignoring them",
                file_path, len(remaining) - used_bytes,
            )
        records = np.frombuffer(remaining[:used_bytes], dtype=dtype)

    if records.size == 0:
        logger.warning("File %s contains no OHLCV records", file_path)
        return _empty_ohlcv()

    df = pd.DataFrame.from_records(records)
    ns = ((df["date"] - 25569) * 86400).round(0).astype(np.int64) * 1_000_000_000
    df.insert(0, "datetime", pd.to_datetime(ns))
    df = df.drop(columns=["date"])
    df = df.set_index("datetime").sort_index()
    return prepare_ohlcv(df)


def _empty_ohlcv() -> pd.DataFrame:
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df.index = pd.DatetimeIndex([], name="datetime")
    return df


def prepare_ohlcv(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """Clean a raw OHLCV DataFrame: dedupe, drop bad rows, sort, cast types.

    Does NOT resample or fill missing trading days -- daily futures data is
    used as-is (no calendar-day forward filling), per project requirements.
    """
    if df.empty:
        return df

    df = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    df = df.dropna(subset=["close"])
    df = df.dropna(subset=["open", "high", "low"])

    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0.0)
    else:
        df["volume"] = 0.0

    drop_bad_rows = False
    if cfg is not None:
        try:
            drop_bad_rows = bool(cfg.data.get("drop_bad_rows", False))
        except Exception:
            drop_bad_rows = False

    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        logger.warning("%d rows have high < low", int(bad_hl.sum()))
        if drop_bad_rows:
            df = df[~bad_hl]

    if cfg is not None:
        try:
            if bool(cfg.data.get("drop_zero_volume", False)):
                df = df[df["volume"] > 0]
        except Exception:
            pass

    # Daily data: normalize timestamps to date-only midnight.
    df.index = df.index.normalize()
    df.index.name = "datetime"

    return df[["open", "high", "low", "close", "volume"]]


def read_trade_station_dat(file_path: Path, filename_meta: FileNameMetadata | None = None, cfg=None) -> tuple[pd.DataFrame, DatMetadata]:
    """Load a .dat file and return (clean OHLCV DataFrame, metadata dict-like)."""
    file_path = Path(file_path)

    with file_path.open("rb") as f:
        header = f.read(8)
        if len(header) < 8:
            raise ValueError(f"File {file_path} is empty or has a truncated header")
        ones, type_format = struct.unpack("ii", header)
        if ones != TS_MAGIC or type_format != TS_TYPE_FORMAT:
            raise ValueError(f"Unsupported .dat format in {file_path}")

        tick = struct.unpack("d", f.read(8))[0]
        bpv = struct.unpack("d", f.read(8))[0]

        country = _read_str(f)
        exchange = _read_str(f)
        symbol = _read_str(f)
        desc = _read_str(f)
        itype = _read_str(f)
        ispan = struct.unpack("i", f.read(4))[0]
        tz = _read_str(f)
        session = _read_str(f)

    df = load_dat(file_path)

    if filename_meta is None:
        pattern = cfg.data.filename_pattern if cfg is not None else r"^A_OHLCV_(?P<symbol>.+)_minutes_(?P<minutes>\d+)\.dat$"
        filename_meta = parse_filename(file_path, pattern) or FileNameMetadata(
            file_path=str(file_path), parsed_symbol=symbol, parsed_minutes=0, inferred_timeframe="unknown",
        )

    symbol_mismatch = bool(filename_meta.parsed_symbol) and filename_meta.parsed_symbol != symbol
    if symbol_mismatch:
        logger.warning(
            "Symbol mismatch for %s: filename says %r, binary metadata says %r. Using priority order.",
            file_path, filename_meta.parsed_symbol, symbol,
        )

    priority = ["filename", "binary_metadata", "config"]
    if cfg is not None:
        try:
            priority = list(cfg.data.get("symbol_priority", priority))
        except Exception:
            pass

    config_symbol = cfg.data.get("default_symbol", "") if cfg is not None else ""
    candidates = {
        "filename": filename_meta.parsed_symbol,
        "binary_metadata": symbol,
        "config": config_symbol,
    }
    resolved_symbol = ""
    for key in priority:
        if candidates.get(key):
            resolved_symbol = candidates[key]
            break

    metadata = DatMetadata(
        file_path=str(file_path),
        parsed_symbol_from_filename=filename_meta.parsed_symbol,
        parsed_minutes_from_filename=filename_meta.parsed_minutes,
        inferred_timeframe_from_filename=filename_meta.inferred_timeframe,
        country=country,
        exchange=exchange,
        symbol=symbol,
        description=desc,
        itype=itype,
        ispan=ispan,
        timezone=tz,
        session=session,
        tick=tick,
        bpv=bpv,
        record_count=len(df),
        start_datetime=df.index.min() if not df.empty else None,
        end_datetime=df.index.max() if not df.empty else None,
        resolved_symbol=resolved_symbol,
        symbol_mismatch=symbol_mismatch,
    )

    return df, metadata


def load_symbol(symbol: str, cfg) -> tuple[pd.DataFrame, DatMetadata]:
    """Convenience: discover files, then load the .dat for a given symbol."""
    file_map = discover_dat_files(cfg)
    if symbol not in file_map:
        raise FileNotFoundError(
            f"No data file found for symbol {symbol!r}. "
            f"Available symbols: {sorted(file_map.keys())}"
        )
    meta = file_map[symbol]
    return read_trade_station_dat(Path(meta.file_path), filename_meta=meta, cfg=cfg)
