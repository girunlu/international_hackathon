from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

_QUANTITY_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _normalize_item_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, str):
        candidate = value.strip()
    else:
        if pd.isna(value):
            return ""
        candidate = str(value).strip()
    if not candidate:
        return ""
    return candidate.lower()


def _format_display_item_name(value: Any) -> str:
    normalized = _normalize_item_name(value)
    if not normalized:
        return ""
    return " ".join(word.capitalize() for word in normalized.split())


def _coerce_quantity(*values: Any) -> float:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, float)):
            result = float(value)
            if math.isfinite(result):
                return result
            continue
        if isinstance(value, str):
            match = _QUANTITY_PATTERN.search(value)
            if match:
                try:
                    result = float(match.group(0))
                except ValueError:
                    continue
                if math.isfinite(result):
                    return result
    return 0.0


def _shift_month_start(date_value: date, offset: int) -> date:
    base = date_value.replace(day=1)
    month_index = base.month - 1 + offset
    year = base.year + month_index // 12
    month = month_index % 12 + 1
    return datetime(year, month, 1).date()


def _get_previous_week_and_year(reference: datetime | None = None) -> tuple[int, int]:
    ref = reference or datetime.now(timezone.utc)
    previous = ref - timedelta(days=7)
    iso = previous.isocalendar()
    return iso.week, iso.year


def _get_previous_week_bounds(
    reference: datetime | None = None,
) -> tuple[date, date]:
    ref_date = (reference or datetime.now(timezone.utc)).date()
    start_current_week = ref_date - timedelta(days=ref_date.weekday())
    start_previous_week = start_current_week - timedelta(days=7)
    end_previous_week = start_current_week - timedelta(days=1)
    return start_previous_week, end_previous_week


def _format_quantity_label(quantity: float, unit: str | None = None) -> str:
    if not math.isfinite(quantity):
        quantity = 0.0
    normalized = float(quantity)
    if math.isclose(normalized, round(normalized)):
        numeric = str(int(round(normalized)))
    else:
        numeric = f"{normalized:.2f}".rstrip("0").rstrip(".")
    return f"{numeric} {unit}".strip() if unit else numeric


__all__ = [
    "_normalize_item_name",
    "_format_display_item_name",
    "_coerce_quantity",
    "_shift_month_start",
    "_get_previous_week_and_year",
    "_get_previous_week_bounds",
    "_format_quantity_label",
]

