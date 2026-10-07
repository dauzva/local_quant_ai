"""Pydantic schema for LLM-generated strategies, plus safe-DSL validation.

The LLM must only ever produce data (JSON) that is validated against these
models. Nothing here uses eval() or exec() on LLM output.
"""
from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from src.indicators import ALLOWED_INDICATORS

ALLOWED_OPERATORS = {">", "<", ">=", "<=", "cross_above", "cross_below"}
BASE_COLUMNS = {"open", "high", "low", "close", "volume"}

# Sane bounds per indicator parameter, used to reject absurd/overfit values.
INDICATOR_PARAM_BOUNDS: dict[str, dict[str, tuple[int, int]]] = {
    "sma": {"period": (2, 300)},
    "ema": {"period": (2, 300)},
    "rsi": {"period": (2, 100)},
    "macd": {"fast": (2, 100), "slow": (2, 200), "signal": (2, 100)},
    "atr": {"period": (2, 100)},
    "bollinger_bands": {"period": (2, 200), "num_std": (1, 4)},
    "donchian": {"period": (2, 300)},
    "roc": {"period": (1, 200)},
    "stochastic": {"period": (2, 100), "smooth_k": (1, 20), "smooth_d": (1, 20)},
    "adx": {"period": (2, 100)},
}


class IndicatorSpec(BaseModel):
    name: str
    column: str | None = "close"
    params: dict[str, float | int] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def name_allowed(cls, v: str) -> str:
        if v not in ALLOWED_INDICATORS:
            raise ValueError(f"Indicator '{v}' is not in the allowed indicator set: {sorted(ALLOWED_INDICATORS)}")
        return v

    @field_validator("column")
    @classmethod
    def column_allowed(cls, v: str | None) -> str | None:
        if v is not None and v not in BASE_COLUMNS:
            raise ValueError(f"Indicator column '{v}' must be one of {sorted(BASE_COLUMNS)}")
        return v

    @model_validator(mode="after")
    def params_within_bounds(self) -> "IndicatorSpec":
        bounds = INDICATOR_PARAM_BOUNDS.get(self.name, {})
        for key, value in self.params.items():
            if key not in bounds:
                raise ValueError(f"Unknown parameter '{key}' for indicator '{self.name}'")
            lo, hi = bounds[key]
            if not (lo <= value <= hi):
                raise ValueError(f"Parameter '{key}'={value} out of bounds [{lo}, {hi}] for indicator '{self.name}'")
        return self


class Condition(BaseModel):
    left: str
    op: str
    right: Union[float, int, str]

    @field_validator("op")
    @classmethod
    def op_allowed(cls, v: str) -> str:
        if v not in ALLOWED_OPERATORS:
            raise ValueError(f"Operator '{v}' not allowed. Must be one of {sorted(ALLOWED_OPERATORS)}")
        return v


class ConditionGroup(BaseModel):
    logic: Literal["and", "or"]
    conditions: list[Condition] = Field(min_length=1, max_length=8)


class StrategySchema(BaseModel):
    strategy_name: str
    symbol: str
    timeframe: str = "1d"
    rationale: str = ""
    indicators: list[IndicatorSpec] = Field(min_length=1, max_length=4)
    entry: ConditionGroup  # LONG entry rule (kept as `entry` for backward compatibility)
    exit: ConditionGroup   # LONG exit rule (kept as `exit` for backward compatibility)
    short_entry: ConditionGroup | None = None  # optional SHORT entry rule
    short_exit: ConditionGroup | None = None   # optional SHORT exit rule
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    position_sizing: str = "risk_based_contracts"
    risk_per_trade: float = 0.01

    @field_validator("stop_loss_pct")
    @classmethod
    def stop_reasonable(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 < v <= 0.5):
            raise ValueError("stop_loss_pct must be in (0, 0.5]")
        return v

    @field_validator("take_profit_pct")
    @classmethod
    def tp_reasonable(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 < v <= 2.0):
            raise ValueError("take_profit_pct must be in (0, 2.0]")
        return v

    @field_validator("risk_per_trade")
    @classmethod
    def risk_reasonable(cls, v: float) -> float:
        if not (0.0 < v <= 0.1):
            raise ValueError("risk_per_trade must be in (0, 0.1]")
        return v

    @model_validator(mode="after")
    def short_legs_paired(self) -> "StrategySchema":
        # Either both short_entry and short_exit are present, or neither is --
        # a short entry with no defined exit (or vice versa) is not a usable rule.
        if (self.short_entry is None) != (self.short_exit is None):
            raise ValueError("short_entry and short_exit must both be provided, or both omitted")
        return self

    @property
    def has_short_logic(self) -> bool:
        return self.short_entry is not None and self.short_exit is not None


def _indicator_output_columns(spec: IndicatorSpec) -> list[str]:
    """Predict the deterministic output column name(s) for an indicator spec,
    without actually computing it, so entry/exit conditions can be validated
    before running any backtest.
    """
    name = spec.name
    p = spec.params
    col = spec.column or "close"
    if name == "sma":
        return [f"sma_{col}_{int(p.get('period', 20))}"]
    if name == "ema":
        return [f"ema_{col}_{int(p.get('period', 20))}"]
    if name == "rsi":
        return [f"rsi_{col}_{int(p.get('period', 14))}"]
    if name == "roc":
        return [f"roc_{col}_{int(p.get('period', 10))}"]
    if name == "macd":
        fast, slow, sig = int(p.get("fast", 12)), int(p.get("slow", 26)), int(p.get("signal", 9))
        prefix = f"macd_{fast}_{slow}_{sig}"
        return [f"{prefix}_line", f"{prefix}_signal", f"{prefix}_hist", prefix]
    if name == "atr":
        return [f"atr_{int(p.get('period', 14))}"]
    if name == "bollinger_bands":
        period = int(p.get("period", 20))
        num_std = p.get("num_std", 2.0)
        tag = int(num_std) if float(num_std).is_integer() else num_std
        prefix = f"bb_{period}_{tag}"
        return [f"{prefix}_middle", f"{prefix}_upper", f"{prefix}_lower"]
    if name == "donchian":
        period = int(p.get("period", 20))
        return [f"donchian_{period}_upper", f"donchian_{period}_lower", f"donchian_{period}_middle"]
    if name == "stochastic":
        period, sk, sd = int(p.get("period", 14)), int(p.get("smooth_k", 3)), int(p.get("smooth_d", 3))
        prefix = f"stoch_{period}_{sk}_{sd}"
        return [f"{prefix}_k", f"{prefix}_d"]
    if name == "adx":
        period = int(p.get("period", 14))
        return [f"adx_{period}", f"plus_di_{period}", f"minus_di_{period}"]
    return []


def validate_strategy_columns(strategy: StrategySchema) -> list[str]:
    """Check that every condition column referenced by entry/exit either is a
    base OHLCV column or is produced by one of the declared indicators.
    Returns a list of problems (empty list == valid).
    """
    problems: list[str] = []
    available = set(BASE_COLUMNS)
    for spec in strategy.indicators:
        available.update(_indicator_output_columns(spec))

    groups = [strategy.entry, strategy.exit]
    if strategy.short_entry is not None:
        groups.append(strategy.short_entry)
    if strategy.short_exit is not None:
        groups.append(strategy.short_exit)

    for group in groups:
        for cond in group.conditions:
            if cond.left not in available:
                problems.append(f"Unknown column referenced on left side: '{cond.left}'")
            if isinstance(cond.right, str) and cond.right not in available:
                # right side could still be a plain number encoded as string; reject only if clearly not numeric
                try:
                    float(cond.right)
                except ValueError:
                    problems.append(f"Unknown column referenced on right side: '{cond.right}'")
    return problems


def validate_strategy_dict(data: dict) -> tuple[StrategySchema | None, list[str]]:
    """Validate a raw dict (typically parsed LLM JSON) end to end.
    Returns (StrategySchema or None, list_of_error_strings).
    """
    try:
        strategy = StrategySchema.model_validate(data)
    except Exception as exc:  # pydantic ValidationError or similar
        return None, [str(exc)]

    problems = validate_strategy_columns(strategy)
    if problems:
        return None, problems
    return strategy, []
