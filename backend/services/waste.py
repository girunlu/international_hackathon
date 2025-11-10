from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

try:
    from .analytics import update_analytics_summary_snapshot
    from ..core import (
        _coerce_quantity,
        _format_display_item_name,
        _format_quantity_label,
        _get_previous_week_and_year,
        _get_previous_week_bounds,
        _normalize_item_name,
        read_csv_as_dataframe,
        write_dataframe_to_csv,
    )
    from .snapshots import (
        update_waste_snapshot,
    )
except ImportError:  # pragma: no cover - allow direct script execution
    from services.analytics import update_analytics_summary_snapshot
    from core import (
        _coerce_quantity,
        _format_display_item_name,
        _format_quantity_label,
        _get_previous_week_and_year,
        _get_previous_week_bounds,
        _normalize_item_name,
        read_csv_as_dataframe,
        write_dataframe_to_csv,
    )
    from services.snapshots import (
        update_waste_snapshot,
    )


def check_waste_history(payload: dict[str, Any]) -> dict[str, Any]:
    raw_name = payload.get("itemName")
    quantity_numeric = _coerce_quantity(
        payload.get("quantity_numeric"), payload.get("quantity")
    )
    unit_value = payload.get("unit")

    item_name = _normalize_item_name(raw_name) if isinstance(raw_name, str) else ""
    display_item_name = _format_display_item_name(item_name)
    if not item_name:
        return {
            "hasWaste": False,
            "itemName": display_item_name if display_item_name else raw_name,
            "wastedAmount": None,
            "wastedQuantityNumeric": None,
            "suggestedAmount": None,
            "suggestedQuantityNumeric": None,
            "suggestedUnit": unit_value if isinstance(unit_value, str) else None,
        }

    normalized_item = item_name
    unit: str | None
    if isinstance(unit_value, str):
        unit = unit_value.strip() or None
    else:
        unit = None

    start_prev_week, end_prev_week = _get_previous_week_bounds()

    waste_df = read_csv_as_dataframe(
        "waste_log", columns=["date_wasted", "item_name", "waste_amount"]
    )
    if waste_df.empty:
        return {
            "hasWaste": False,
            "itemName": display_item_name,
            "wastedAmount": None,
            "wastedQuantityNumeric": None,
            "suggestedAmount": None,
            "suggestedQuantityNumeric": None,
            "suggestedUnit": unit,
        }

    waste_df = waste_df.dropna(subset=["date_wasted", "item_name"])
    if waste_df.empty:
        return {
            "hasWaste": False,
            "itemName": display_item_name,
            "wastedAmount": None,
            "wastedQuantityNumeric": None,
            "suggestedAmount": None,
            "suggestedQuantityNumeric": None,
            "suggestedUnit": unit,
        }

    waste_df["date_wasted"] = pd.to_datetime(
        waste_df["date_wasted"], errors="coerce"
    ).dt.date
    waste_df = waste_df.dropna(subset=["date_wasted"])

    waste_df["item_name"] = waste_df["item_name"].astype(str).str.strip()
    waste_df["waste_amount"] = pd.to_numeric(
        waste_df["waste_amount"], errors="coerce"
    ).fillna(0.0)

    relevant_waste = waste_df[
        (waste_df["item_name"].str.lower() == normalized_item)
        & (waste_df["date_wasted"] >= start_prev_week)
        & (waste_df["date_wasted"] <= end_prev_week)
    ]

    total_wasted = (
        float(relevant_waste["waste_amount"].sum())
        if not relevant_waste.empty
        else 0.0
    )
    if total_wasted <= 0:
        return {
            "hasWaste": False,
            "itemName": display_item_name,
            "wastedAmount": None,
            "wastedQuantityNumeric": None,
            "suggestedAmount": None,
            "suggestedQuantityNumeric": None,
            "suggestedUnit": unit,
        }

    purchases_df = read_csv_as_dataframe(
        "purchases", columns=["date_purchased", "item_name", "item_amount"]
    )
    purchases_df = purchases_df.dropna(subset=["date_purchased", "item_name"])
    purchased_amount = 0.0
    if not purchases_df.empty:
        purchases_df["date_purchased"] = pd.to_datetime(
            purchases_df["date_purchased"], errors="coerce"
        ).dt.date
        purchases_df = purchases_df.dropna(subset=["date_purchased"])
        purchases_df["item_name"] = purchases_df["item_name"].astype(str).str.strip()
        purchases_df["item_amount"] = pd.to_numeric(
            purchases_df["item_amount"], errors="coerce"
        ).fillna(0.0)

        relevant_purchases = purchases_df[
            (purchases_df["item_name"].str.lower() == normalized_item)
            & (purchases_df["date_purchased"] >= start_prev_week)
            & (purchases_df["date_purchased"] <= end_prev_week)
        ]

        if not relevant_purchases.empty:
            purchased_amount = float(relevant_purchases["item_amount"].sum())

    baseline_quantity = purchased_amount if purchased_amount > 0 else quantity_numeric
    net_consumed = max(baseline_quantity - total_wasted, 0.0)

    suggested_quantity = quantity_numeric
    if total_wasted > 0:
        if quantity_numeric > net_consumed:
            suggested_quantity = net_consumed
        elif purchased_amount > 0 and purchased_amount > net_consumed:
            suggested_quantity = net_consumed

    if quantity_numeric > 0 and suggested_quantity >= quantity_numeric:
        suggested_quantity = max(quantity_numeric * 0.5, 0.0)

    suggested_amount_label: str | None = None
    suggested_quantity_numeric: float | None = None
    if suggested_quantity > 0 and (
        quantity_numeric <= 0 or suggested_quantity < quantity_numeric
    ):
        suggested_quantity_numeric = suggested_quantity
        suggested_amount_label = _format_quantity_label(suggested_quantity, unit)

    wasted_amount_label = _format_quantity_label(total_wasted, unit)

    return {
        "hasWaste": True,
        "itemName": display_item_name,
        "wastedAmount": wasted_amount_label,
        "wastedQuantityNumeric": total_wasted,
        "suggestedAmount": suggested_amount_label,
        "suggestedQuantityNumeric": suggested_quantity_numeric,
        "suggestedUnit": unit,
    }


def log_waste(payload: dict[str, Any]) -> dict[str, Any]:
    raw_item_name = payload.get("itemName")
    item_name = _normalize_item_name(raw_item_name)
    quantity_label = payload.get("quantity")
    quantity_numeric = payload.get("quantity_numeric")

    try:
        quantity_numeric = float(quantity_numeric)
    except (TypeError, ValueError):
        quantity_numeric = 0.0

    if not item_name or quantity_numeric <= 0:
        return {"success": False, "error": "Invalid waste entry"}

    waste_df = read_csv_as_dataframe(
        "waste_log", columns=["date_wasted", "item_name", "waste_amount"]
    )
    new_row = pd.DataFrame(
        [
            {
                "date_wasted": datetime.now(timezone.utc).date().isoformat(),
                "item_name": item_name,
                "waste_amount": quantity_numeric,
            }
        ]
    )
    waste_df = pd.concat([waste_df, new_row], ignore_index=True)
    write_dataframe_to_csv(waste_df, "waste_log")
    update_waste_snapshot()
    update_analytics_summary_snapshot()
    return {"success": True}


def analyze_waste(payload: dict[str, Any]) -> dict[str, Any]:
    item_name = payload.get("itemName")

    purchases_df = read_csv_as_dataframe(
        "purchases",
        columns=["date_purchased", "item_name", "item_amount"],
    )
    waste_df = read_csv_as_dataframe(
        "waste_log",
        columns=["date_wasted", "item_name", "waste_amount"],
    )

    summary = {
        "shouldSuggestReduction": False,
        "suggestedQuantity": None,
        "suggestedQuantityNumeric": None,
        "lastWeekPurchased": 0.0,
        "lastWeekWasted": 0.0,
    }

    if not item_name:
        return summary

    def _prepare(df: pd.DataFrame, date_column: str, value_column: str) -> pd.DataFrame:
        df = df.dropna(subset=[date_column, "item_name"])
        if df.empty:
            return df
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=[date_column])
        if df.empty:
            return df
        df["item_name"] = df["item_name"].astype(str).str.strip()
        df = df[df["item_name"] != ""]
        df[value_column] = pd.to_numeric(df[value_column], errors="coerce").fillna(0.0)
        df["week"] = df[date_column].dt.isocalendar().week
        df["year"] = df[date_column].dt.isocalendar().year
        return df

    purchases_df = _prepare(purchases_df, "date_purchased", "item_amount")
    waste_df = _prepare(waste_df, "date_wasted", "waste_amount")

    if purchases_df.empty and waste_df.empty:
        return summary
    prev_week, prev_year = _get_previous_week_and_year()

    item_lower = str(item_name).strip().lower()

    last_week_purchases = purchases_df[
        (purchases_df["week"] == prev_week)
        & (purchases_df["year"] == prev_year)
        & (purchases_df["item_name"].str.lower() == item_lower)
    ]

    last_week_waste = waste_df[
        (waste_df["week"] == prev_week)
        & (waste_df["year"] == prev_year)
        & (waste_df["item_name"].str.lower() == item_lower)
    ]

    purchased_amount = (
        last_week_purchases["item_amount"].sum()
        if not last_week_purchases.empty
        else 0.0
    )
    wasted_amount = (
        last_week_waste["waste_amount"].sum()
        if not last_week_waste.empty
        else 0.0
    )

    suggested_quantity = max(purchased_amount - wasted_amount, 0.0)

    summary["lastWeekPurchased"] = round(purchased_amount, 4)
    summary["lastWeekWasted"] = round(wasted_amount, 4)
    if purchased_amount <= 0 or wasted_amount <= 0:
        return summary

    summary["suggestedQuantityNumeric"] = round(suggested_quantity, 4)
    summary["suggestedQuantity"] = f"{suggested_quantity:.2f}"
    summary["shouldSuggestReduction"] = suggested_quantity < purchased_amount

    requested_quantity = payload.get("requestedQuantity")
    if requested_quantity is not None:
        try:
            requested_numeric = float(requested_quantity)
            if requested_numeric <= suggested_quantity:
                summary["shouldSuggestReduction"] = False
        except (TypeError, ValueError):
            pass

    return summary


__all__ = ["check_waste_history", "log_waste", "analyze_waste"]

