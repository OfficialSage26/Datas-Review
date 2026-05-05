"""Shared utilities for the fraud detection project."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATAS_DIR = PROJECT_ROOT / "Datas"
GUIDES_DIR = PROJECT_ROOT / "Guides"
MODELS_DIR = PROJECT_ROOT / "models"


def normalize_column_name(name: Any) -> str:
    """Convert arbitrary column names into stable snake_case names."""
    value = str(name).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized column names."""
    renamed = {column: normalize_column_name(column) for column in df.columns}
    return df.rename(columns=renamed).copy()


def clamp(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = low
    if math.isnan(numeric) or math.isinf(numeric):
        numeric = low
    return max(low, min(high, numeric))


def safe_divide(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return default
    if den == 0 or math.isnan(den) or math.isinf(den):
        return default
    value = num / den
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def risk_level_from_score(score: Any) -> str:
    score_value = clamp(score)
    if score_value <= 24:
        return "Low"
    if score_value <= 49:
        return "Medium"
    if score_value <= 74:
        return "High"
    return "Critical"


def decision_from_score(score: Any, row: dict[str, Any] | pd.Series | None = None) -> str:
    score_value = clamp(score)
    row_dict = {} if row is None else dict(row)
    repeated = bool(row_dict.get("repeated_suspicious_behavior", False))
    manipulated_after_approval = bool(row_dict.get("manipulated_after_approval", False))
    if score_value >= 90 and (repeated or manipulated_after_approval):
        return "Escalate"
    if score_value >= 75:
        return "Reject"
    if score_value >= 50:
        return "Reject and Ask for Analytics in Ticket"
    if score_value >= 25:
        return "Needs Manual Review"
    return "Approve"


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return False
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "flag", "flagged"}


def coerce_numeric_columns(
    df: pd.DataFrame,
    known_numeric: Iterable[str] | None = None,
    known_boolean: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Safely coerce numeric-looking columns without damaging text columns."""
    result = df.copy()
    known_numeric = set(known_numeric or [])
    known_boolean = set(known_boolean or [])

    for column in result.columns:
        if column in known_boolean or column.endswith("_flag"):
            result[column] = result[column].map(as_bool)
            continue

        series = result[column]
        should_try = column in known_numeric or series.dtype == object
        if not should_try:
            continue

        converted = pd.to_numeric(series, errors="coerce")
        non_missing_original = series.notna().sum()
        non_missing_converted = converted.notna().sum()
        if column in known_numeric or non_missing_original == 0:
            result[column] = converted
        elif non_missing_converted / max(non_missing_original, 1) >= 0.9:
            result[column] = converted
    return result


def ensure_columns(df: pd.DataFrame, columns: Iterable[str], default: Any = 0) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = default
    return result


def first_present(row: dict[str, Any] | pd.Series, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return default


def clean_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default
