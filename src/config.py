"""Config loading: config.yaml + environment variables (.env supported)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class AttrDict(dict):
    """Dict that also supports attribute access, recursively.

    Kept intentionally simple (no pydantic) since config.yaml is trusted
    local input, not external/untrusted data.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(config_path: str | Path = "config.yaml") -> AttrDict:
    """Load config.yaml, load .env if present, and return an AttrDict.

    Environment variables referenced by config (e.g. llm.api_key_env) are
    resolved lazily by the code that needs them (see src/llm.py), not here,
    so the API key itself is never stored inside the config object.
    """
    load_dotenv(override=False)

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = _wrap(raw)

    # A couple of light defaults / normalizations so downstream code is simple.
    cfg.setdefault("run", AttrDict())
    cfg.run.setdefault("iterations", 5)
    cfg.run.setdefault("strategies_per_iteration", 5)
    cfg.run.setdefault("random_seed", 42)

    return cfg


def get_api_key(cfg: AttrDict) -> str | None:
    """Read the OpenRouter API key from the environment variable named in config."""
    env_var = cfg.llm.get("api_key_env", "OPENROUTER_API_KEY")
    return os.environ.get(env_var)
