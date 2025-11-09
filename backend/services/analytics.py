from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

try:
    from ..core.config import SAVED_ITEMS_COLUMNS
    from ..core.helpers import (
        _format_display_item_name,
        _normalize_item_name,
        _shift_month_start,
    )
    from ..core.storage import (
        read_csv_as_dataframe,
        read_json_payload,
        write_json_payload,
    )
    from .pricing import _build_unit_price_map
except ImportError:  # pragma: no cover - allow direct script execution
    from core.config import SAVED_ITEMS_COLUMNS
    from core.helpers import (
        _format_display_item_name,
        _normalize_item_name,
        _shift_month_start,
    )
    from core.storage import (
        read_csv_as_dataframe,
        read_json_payload,
        write_json_payload,
    )
    from services.pricing import _build_unit_price_map


def _calculate_analytics_series(
    time_range: str, item_name: str | None
) -> dict[str, Any]:
    today = datetime.now(timezone.utc).date()
    normalized_range = "monthly" if time_range == "monthly" else "weekly"

    if normalized_range == "monthly":
        period_count = 6
        current_month_start = today.replace(day=1)
        bucket_definitions: list[tuple[date, date, str]] = []
        for offset in range(period_count):
            month_start = _shift_month_start(current_month_start, -offset)
            month_end_day = calendar.monthrange(month_start.year, month_start.month)[1]
            month_end = month_start.replace(day=month_end_day)
            label = f"{month_start.year}-{month_start.month:02d}"
            bucket_definitions.append((month_start, month_end, label))
        bucket_definitions.reverse()
    else:
        period_count = 5
        bucket_definitions = []
        for offset in range(period_count):
            end_date = today - timedelta(days=offset * 7)
            start_period = end_date - timedelta(days=6)
            label = end_date.isoformat()
            bucket_definitions.append((start_period, end_date, label))
        bucket_definitions.reverse()

    if not bucket_definitions:
        bucket_definitions = [(today, today, today.isoformat())]

    range_start = bucket_definitions[0][0]
    unit_price_map = _build_unit_price_map()
    normalized_filter: str | None = None
    display_filter: str | None = None
    if item_name:
        candidate = _normalize_item_name(item_name)
        if candidate and candidate not in {"general", "all"}:
            normalized_filter = candidate
            display_filter = _format_display_item_name(candidate)

    all_items: dict[str, str] = {}

    purchases_df = read_csv_as_dataframe(
        "purchases", columns=["date_purchased", "item_name", "item_amount"]
    )
    if not purchases_df.empty:
        purchases_df = purchases_df.dropna(subset=["date_purchased", "item_name"])
        purchases_df = purchases_df.copy()
        purchases_df["item_name"] = purchases_df["item_name"].map(_normalize_item_name)
        purchases_df["normalized_name"] = purchases_df["item_name"]
        for name in purchases_df["normalized_name"]:
            if isinstance(name, str) and name:
                all_items.setdefault(name, _format_display_item_name(name))
        purchases_df["date_purchased"] = pd.to_datetime(
            purchases_df["date_purchased"], errors="coerce"
        )
        purchases_df = purchases_df.dropna(subset=["date_purchased"])
        purchases_df["date_only"] = purchases_df["date_purchased"].dt.date
        purchases_df = purchases_df[
            (purchases_df["date_only"] >= range_start)
            & (purchases_df["date_only"] <= today)
        ]
        purchases_df["item_amount"] = pd.to_numeric(
            purchases_df["item_amount"], errors="coerce"
        ).fillna(0.0)
        purchases_df["unit_price"] = purchases_df["normalized_name"].map(
            unit_price_map
        ).fillna(0.0)
        purchases_df["item_total_price"] = (
            purchases_df["item_amount"] * purchases_df["unit_price"]
        )
        if normalized_filter:
            purchases_df = purchases_df[
                purchases_df["normalized_name"] == normalized_filter
            ]
    else:
        purchases_df = pd.DataFrame(
            columns=["date_only", "item_total_price", "normalized_name", "item_name"]
        )

    saved_df = read_csv_as_dataframe("saved_items", columns=SAVED_ITEMS_COLUMNS)
    if not saved_df.empty:
        saved_df = saved_df.dropna(subset=["date_saved", "item_name"])
        saved_df = saved_df.copy()
        saved_df["item_name"] = saved_df["item_name"].map(_normalize_item_name)
        saved_df["normalized_name"] = saved_df["item_name"]
        for name in saved_df["normalized_name"]:
            if isinstance(name, str) and name:
                all_items.setdefault(name, _format_display_item_name(name))
        saved_df["date_saved"] = pd.to_datetime(
            saved_df["date_saved"], errors="coerce"
        )
        saved_df = saved_df.dropna(subset=["date_saved"])
        saved_df["date_only"] = saved_df["date_saved"].dt.date
        saved_df = saved_df[
            (saved_df["date_only"] >= range_start)
            & (saved_df["date_only"] <= today)
        ]
        saved_df["saved_money"] = pd.to_numeric(
            saved_df["saved_money"], errors="coerce"
        ).fillna(0.0)
        if normalized_filter:
            saved_df = saved_df[saved_df["normalized_name"] == normalized_filter]
    else:
        saved_df = pd.DataFrame(columns=["date_only", "saved_money", "normalized_name"])

    waste_df = read_csv_as_dataframe(
        "waste_log", columns=["date_wasted", "item_name", "waste_amount"]
    )
    if not waste_df.empty:
        waste_df = waste_df.dropna(subset=["date_wasted", "item_name"])
        waste_df = waste_df.copy()
        waste_df["item_name"] = waste_df["item_name"].map(_normalize_item_name)
        waste_df["normalized_name"] = waste_df["item_name"]
        for name in waste_df["normalized_name"]:
            if isinstance(name, str) and name:
                all_items.setdefault(name, _format_display_item_name(name))
        waste_df["date_wasted"] = pd.to_datetime(
            waste_df["date_wasted"], errors="coerce"
        )
        waste_df = waste_df.dropna(subset=["date_wasted"])
        waste_df["date_only"] = waste_df["date_wasted"].dt.date
        waste_df = waste_df[
            (waste_df["date_only"] >= range_start)
            & (waste_df["date_only"] <= today)
        ]
        waste_df["waste_amount"] = pd.to_numeric(
            waste_df["waste_amount"], errors="coerce"
        ).fillna(0.0)
        waste_df["unit_price"] = waste_df["normalized_name"].map(unit_price_map).fillna(
            0.0
        )
        waste_df["wasted_price"] = (
            waste_df["waste_amount"] * waste_df["unit_price"]
        )
        if normalized_filter:
            waste_df = waste_df[waste_df["normalized_name"] == normalized_filter]
    else:
        waste_df = pd.DataFrame(columns=["date_only", "wasted_price", "normalized_name"])

    spent_by_date: dict[date, float] = {}
    if not purchases_df.empty:
        spent_by_date = (
            purchases_df.groupby("date_only")["item_total_price"].sum().to_dict()
        )

    saved_by_date: dict[date, float] = {}
    if not saved_df.empty:
        saved_by_date = saved_df.groupby("date_only")["saved_money"].sum().to_dict()

    wasted_by_date: dict[date, float] = {}
    if not waste_df.empty:
        wasted_by_date = waste_df.groupby("date_only")["wasted_price"].sum().to_dict()

    series = []
    total_spent = 0.0
    total_saved = 0.0
    total_wasted = 0.0

    for start_period, end_period, label in bucket_definitions:
        spent_value = float(
            sum(
                value
                for date_key, value in spent_by_date.items()
                if start_period <= date_key <= end_period
            )
        )
        saved_value = float(
            sum(
                value
                for date_key, value in saved_by_date.items()
                if start_period <= date_key <= end_period
            )
        )
        wasted_value = float(
            sum(
                value
                for date_key, value in wasted_by_date.items()
                if start_period <= date_key <= end_period
            )
        )
        total_spent += spent_value
        total_saved += saved_value
        total_wasted += wasted_value
        series.append(
            {
                "date": label,
                "rangeStart": start_period.isoformat(),
                "rangeEnd": end_period.isoformat(),
                "spent": round(spent_value, 2),
                "saved": round(saved_value, 2),
                "wasted": round(wasted_value, 2),
            }
        )

    waste_percentage = 0.0
    if total_spent > 0:
        waste_percentage = (total_wasted / total_spent) * 100.0

    available_items = ["general"] + [
        all_items[key] for key in sorted(all_items)
    ]

    return {
        "timeRange": time_range,
        "itemName": display_filter or "general",
        "series": series,
        "totals": {
            "spent": round(total_spent, 2),
            "saved": round(total_saved, 2),
            "wasted": round(total_wasted, 2),
            "wastePercentage": round(waste_percentage, 2),
        },
        "availableItems": available_items,
    }


def update_analytics_summary_snapshot(
    reference: datetime | None = None,
) -> dict[str, Any]:
    today = (reference or datetime.now(timezone.utc)).date()
    target_year = today.year
    target_month = today.month

    unit_price_map = _build_unit_price_map()

    existing_summary_payload = read_json_payload("analytics_summary", {})
    weekly_suggestion: Any = None
    weekly_suggestion_generated_at: Any = None
    if isinstance(existing_summary_payload, dict):
        weekly_suggestion = existing_summary_payload.get("weeklySuggestion")
        weekly_suggestion_generated_at = existing_summary_payload.get(
            "weeklySuggestionGeneratedAt"
        )

    purchases_df = read_csv_as_dataframe(
        "purchases", columns=["date_purchased", "item_name", "item_amount"]
    )
    purchases_df = purchases_df.dropna(subset=["date_purchased", "item_name"])
    if not purchases_df.empty:
        purchases_df = purchases_df.copy()
        purchases_df["date_purchased"] = pd.to_datetime(
            purchases_df["date_purchased"], errors="coerce"
        )
        purchases_df = purchases_df.dropna(subset=["date_purchased"])
    month_purchases = pd.DataFrame(columns=purchases_df.columns)
    if not purchases_df.empty:
        purchases_df["year"] = purchases_df["date_purchased"].dt.year
        purchases_df["month"] = purchases_df["date_purchased"].dt.month
        month_purchases = purchases_df[
            (purchases_df["year"] == target_year)
            & (purchases_df["month"] == target_month)
        ].copy()

    total_spent = 0.0
    top_wasted: list[dict[str, Any]] = []
    if not month_purchases.empty:
        month_purchases["item_amount"] = pd.to_numeric(
            month_purchases["item_amount"], errors="coerce"
        ).fillna(0.0)
        month_purchases["item_name"] = month_purchases["item_name"].map(
            _normalize_item_name
        )
        month_purchases["normalized_name"] = month_purchases["item_name"]
        month_purchases["unit_price"] = month_purchases["normalized_name"].map(
            unit_price_map
        ).fillna(0.0)
        month_purchases["item_total_price"] = (
            month_purchases["item_amount"] * month_purchases["unit_price"]
        )
        total_spent = float(month_purchases["item_total_price"].sum())

        purchase_counts = (
            month_purchases.groupby("normalized_name").size().sort_values(ascending=False)
        )
        purchase_totals = month_purchases.groupby("normalized_name")[
            "item_total_price"
        ].sum()
        display_names = month_purchases.groupby("normalized_name")["item_name"].last()

        top_wasted = []
        for normalized_name in purchase_counts.head(3).index:
            top_wasted.append(
                {
                    "name": _format_display_item_name(
                        display_names.get(normalized_name, normalized_name)
                    ),
                    "count": int(purchase_counts.get(normalized_name, 0)),
                    "amount": float(purchase_totals.get(normalized_name, 0.0)),
                }
            )

    saved_df = read_csv_as_dataframe("saved_items", columns=SAVED_ITEMS_COLUMNS)
    saved_df = saved_df.dropna(subset=["item_name", "date_saved"])
    month_saved = pd.DataFrame(columns=saved_df.columns)
    if not saved_df.empty:
        saved_df = saved_df.copy()
        saved_df["date_saved"] = pd.to_datetime(
            saved_df["date_saved"], errors="coerce"
        )
        saved_df = saved_df.dropna(subset=["date_saved"])
        saved_df["year"] = saved_df["date_saved"].dt.year
        saved_df["month"] = saved_df["date_saved"].dt.month
        month_saved = saved_df[
            (saved_df["year"] == target_year) & (saved_df["month"] == target_month)
        ].copy()

    total_saved = 0.0
    top_saved: list[dict[str, Any]] = []
    if not month_saved.empty:
        month_saved["item_name"] = month_saved["item_name"].map(_normalize_item_name)
        month_saved["saved_money"] = pd.to_numeric(
            month_saved["saved_money"], errors="coerce"
        ).fillna(0.0)
        month_saved["save_amount"] = pd.to_numeric(
            month_saved["save_amount"], errors="coerce"
        ).fillna(0.0)

        total_saved = float(month_saved["saved_money"].sum())

        saved_amounts = (
            month_saved.groupby("item_name")["saved_money"]
            .sum()
            .sort_values(ascending=False)
        )
        saved_counts = month_saved.groupby("item_name").size()

        for item_name in saved_amounts.head(3).index:
            top_saved.append(
                {
                    "name": _format_display_item_name(item_name),
                    "amount": float(saved_amounts.get(item_name, 0.0)),
                    "count": int(saved_counts.get(item_name, 0)),
                }
            )

    waste_df = read_csv_as_dataframe(
        "waste_log", columns=["date_wasted", "item_name", "waste_amount"]
    )
    waste_df = waste_df.dropna(subset=["date_wasted", "item_name"])
    month_waste = pd.DataFrame(columns=waste_df.columns)
    if not waste_df.empty:
        waste_df = waste_df.copy()
        waste_df["date_wasted"] = pd.to_datetime(
            waste_df["date_wasted"], errors="coerce"
        )
        waste_df = waste_df.dropna(subset=["date_wasted"])
        waste_df["year"] = waste_df["date_wasted"].dt.year
        waste_df["month"] = waste_df["date_wasted"].dt.month
        month_waste = waste_df[
            (waste_df["year"] == target_year) & (waste_df["month"] == target_month)
        ].copy()

    total_wasted = 0.0
    if not month_waste.empty:
        month_waste["item_name"] = month_waste["item_name"].map(_normalize_item_name)
        month_waste["normalized_name"] = month_waste["item_name"]
        month_waste["waste_amount"] = pd.to_numeric(
            month_waste["waste_amount"], errors="coerce"
        ).fillna(0.0)
        month_waste["unit_price"] = month_waste["normalized_name"].map(
            unit_price_map
        ).fillna(0.0)
        month_waste["wasted_price"] = (
            month_waste["waste_amount"] * month_waste["unit_price"]
        )
        total_wasted = float(month_waste["wasted_price"].sum())

    waste_percentage = 0.0
    if total_spent > 0:
        waste_percentage = (total_wasted / total_spent) * 100.0

    summary = {
        "wastePercentage": round(waste_percentage, 2),
        "totalSpent": round(total_spent, 2),
        "totalSaved": round(total_saved, 2),
        "totalWasted": round(total_wasted, 2),
        "topWasted": top_wasted,
        "topSaved": top_saved,
    }
    if weekly_suggestion is not None:
        summary["weeklySuggestion"] = weekly_suggestion
    if weekly_suggestion_generated_at is not None:
        summary["weeklySuggestionGeneratedAt"] = weekly_suggestion_generated_at

    write_json_payload("analytics_summary", summary)
    return summary


__all__ = ["_calculate_analytics_series", "update_analytics_summary_snapshot"]

