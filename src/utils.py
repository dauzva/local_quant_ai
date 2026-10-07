"""Small shared utilities: logging setup, ids, safe filesystem helpers."""
from __future__ import annotations

import logging
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Configure root logging once. Safe to call multiple times (idempotent)."""
    logger = logging.getLogger("local_quant_ai")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"local_quant_ai.{name}")


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{prefix}_{stamp}_{short}"


def safe_strategy_dirname(strategy_name: str) -> str:
    """Turn an arbitrary strategy name into a filesystem-safe directory name."""
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]+", "_", strategy_name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "strategy"


def strategy_filename(strategy_name: str) -> str:
    """File name for a saved strategy: strat_gen_<strategy_name>.py"""
    return f"strat_gen_{safe_strategy_dirname(strategy_name)}.py"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def strip_code_fences(text: str) -> str:
    """Strip markdown ```json ... ``` or ``` ... ``` fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
