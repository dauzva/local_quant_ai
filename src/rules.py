"""Convert validated strategy entry/exit rules into boolean pandas Series.

No eval() or exec() is ever used. Only a small fixed set of operators is
supported, applied via plain pandas comparisons.
"""
from __future__ import annotations

import pandas as pd

from src.schema import Condition, ConditionGroup, StrategySchema


def _resolve_side(df: pd.DataFrame, value) -> pd.Series:
    """Resolve the right-hand side of a condition to a Series (broadcast if numeric)."""
    if isinstance(value, (int, float)):
        return pd.Series(float(value), index=df.index)
    if isinstance(value, str):
        try:
            return pd.Series(float(value), index=df.index)
        except ValueError:
            pass
        if value not in df.columns:
            raise ValueError(f"Column '{value}' not found while evaluating rule")
        return df[value]
    raise ValueError(f"Unsupported condition value: {value!r}")


def _eval_condition(df: pd.DataFrame, cond: Condition) -> pd.Series:
    if cond.left not in df.columns:
        raise ValueError(f"Column '{cond.left}' not found while evaluating rule")
    left = df[cond.left]
    right = _resolve_side(df, cond.right)

    if cond.op == ">":
        return left > right
    if cond.op == "<":
        return left < right
    if cond.op == ">=":
        return left >= right
    if cond.op == "<=":
        return left <= right
    if cond.op == "cross_above":
        prev_left = left.shift(1)
        prev_right = right.shift(1)
        return (prev_left <= prev_right) & (left > right)
    if cond.op == "cross_below":
        prev_left = left.shift(1)
        prev_right = right.shift(1)
        return (prev_left >= prev_right) & (left < right)
    raise ValueError(f"Unsupported operator: {cond.op}")


def evaluate_condition_group(df: pd.DataFrame, group: ConditionGroup) -> pd.Series:
    results = [_eval_condition(df, c) for c in group.conditions]
    combined = results[0]
    for r in results[1:]:
        combined = (combined & r) if group.logic == "and" else (combined | r)
    return combined.fillna(False)


def compute_signals(df: pd.DataFrame, strategy: StrategySchema) -> pd.DataFrame:
    """Return df with four extra boolean columns: entry_signal / exit_signal
    (long) and short_entry_signal / short_exit_signal (short).

    Signals are computed strictly from data available at (or before) the
    current bar's close -- no shifting forward, no lookahead. Execution
    (next bar open) is handled separately by the backtester.

    short_entry_signal / short_exit_signal are all-False when the strategy
    has no short logic (strategy.short_entry / strategy.short_exit are
    None) -- this keeps the backtester's column access uniform regardless
    of whether a given strategy trades short at all.
    """
    out = df.copy()
    out["entry_signal"] = evaluate_condition_group(df, strategy.entry)
    out["exit_signal"] = evaluate_condition_group(df, strategy.exit)

    if strategy.short_entry is not None:
        out["short_entry_signal"] = evaluate_condition_group(df, strategy.short_entry)
    else:
        out["short_entry_signal"] = False

    if strategy.short_exit is not None:
        out["short_exit_signal"] = evaluate_condition_group(df, strategy.short_exit)
    else:
        out["short_exit_signal"] = False

    return out
