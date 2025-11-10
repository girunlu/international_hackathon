from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

try:
    from ..core import (
        SAVED_ITEMS_COLUMNS,
        SHOPPING_LIST_COLUMNS,
        _coerce_quantity,
        _format_display_item_name,
        _normalize_item_name,
        read_csv_as_dataframe,
        write_dataframe_to_csv,
    )
    from .analytics import update_analytics_summary_snapshot
    from .pricing import _build_unit_price_map
    from .snapshots import (
        update_saved_items_snapshot,
        update_shopping_history_snapshot,
        update_shopping_list_snapshot,
    )
except ImportError:  # pragma: no cover - allow direct script execution
    from core import (
        SAVED_ITEMS_COLUMNS,
        SHOPPING_LIST_COLUMNS,
        _coerce_quantity,
        _format_display_item_name,
        _normalize_item_name,
        read_csv_as_dataframe,
        write_dataframe_to_csv,
    )
    from services.analytics import update_analytics_summary_snapshot
    from services.pricing import _build_unit_price_map
    from services.snapshots import (
        update_saved_items_snapshot,
        update_shopping_history_snapshot,
        update_shopping_list_snapshot,
    )


def add_shopping_item(payload: dict[str, Any]) -> dict[str, Any]:
    raw_item_name = payload.get("itemName", "")
    item_name = _normalize_item_name(raw_item_name)
    quantity_label = payload.get("quantity")
    quantity_numeric = payload.get("quantity_numeric")
    unit_value = payload.get("unit")

    numeric_quantity = _coerce_quantity(quantity_numeric, quantity_label)

    unit: str | None
    if isinstance(unit_value, str):
        unit = unit_value.strip() or None
    else:
        unit = None

    if not item_name or numeric_quantity <= 0.0:
        return {"success": False, "error": "Invalid shopping list item"}

    shopping_df = read_csv_as_dataframe("shopping_list", columns=SHOPPING_LIST_COLUMNS)
    existing_count = len(shopping_df.index)

    today = datetime.now(timezone.utc).date().isoformat()
    new_row = pd.DataFrame(
        [
            {
                "date_added": today,
                "item_name": item_name,
                "item_amount": numeric_quantity,
                "item_unit": unit,
                "saved_original_amount": pd.NA,
                "saved_final_amount": pd.NA,
                "saved_amount_difference": pd.NA,
                "saved_unit_price": pd.NA,
            }
        ]
    )

    shopping_df = pd.concat([shopping_df, new_row], ignore_index=True)
    write_dataframe_to_csv(shopping_df, "shopping_list")
    records = update_shopping_list_snapshot()

    created_item = next(
        (record for record in records if record.get("id") == existing_count + 1),
        records[-1] if records else None,
    )

    return {"success": created_item is not None, "item": created_item}


def remove_shopping_item(payload: dict[str, Any]) -> dict[str, Any]:
    item_id = payload.get("id")

    shopping_df = read_csv_as_dataframe("shopping_list", columns=SHOPPING_LIST_COLUMNS)

    if shopping_df.empty:
        update_shopping_list_snapshot()
        return {"success": False, "error": "List is already empty"}

    try:
        index = int(item_id) - 1
    except (TypeError, ValueError):
        update_shopping_list_snapshot()
        return {"success": False, "error": "Invalid item id"}

    if index < 0 or index >= len(shopping_df):
        update_shopping_list_snapshot()
        return {"success": False, "error": "Item not found"}

    shopping_df = shopping_df.drop(shopping_df.index[index]).reset_index(drop=True)
    write_dataframe_to_csv(shopping_df, "shopping_list")
    update_shopping_list_snapshot()
    return {"success": True}


def mark_items_bought(payload: dict[str, Any]) -> dict[str, Any]:
    items = payload.get("items") or []

    if not isinstance(items, Iterable):
        return {"success": False, "error": "items must be a list", "removedItemIds": []}

    shopping_df = read_csv_as_dataframe("shopping_list", columns=SHOPPING_LIST_COLUMNS)
    purchases_df = read_csv_as_dataframe(
        "purchases", columns=["date_purchased", "item_name", "item_amount"]
    )

    price_map = _build_unit_price_map()

    removed_item_ids: list[int] = []
    saved_entries: list[dict[str, Any]] = []
    today = datetime.now(timezone.utc).date().isoformat()

    for item in items:
        raw_name = item.get("item_name", "")
        name = _normalize_item_name(raw_name)
        quantity_numeric = item.get("quantity_numeric")

        try:
            quantity = float(quantity_numeric)
        except (TypeError, ValueError):
            quantity = 0.0

        if not name or quantity <= 0:
            continue

        new_purchase = pd.DataFrame(
            [
                {
                    "date_purchased": today,
                    "item_name": name,
                    "item_amount": quantity,
                }
            ]
        )
        purchases_df = pd.concat([purchases_df, new_purchase], ignore_index=True)

        matches = shopping_df[
            shopping_df["item_name"].astype(str).str.strip().str.lower() == name.lower()
        ].index
        if not matches.empty:
            removed_index = matches[0]
            row = shopping_df.loc[removed_index]

            saved_difference_raw = row.get("saved_amount_difference")
            try:
                saved_difference = float(saved_difference_raw)
            except (TypeError, ValueError):
                saved_difference = 0.0

            saved_difference = (
                float(saved_difference) if math.isfinite(saved_difference) else 0.0
            )
            if saved_difference > 0:
                saved_unit_price_raw = row.get("saved_unit_price")
                try:
                    saved_unit_price = float(saved_unit_price_raw)
                except (TypeError, ValueError):
                    saved_unit_price = 0.0
                if not math.isfinite(saved_unit_price) or saved_unit_price <= 0:
                    saved_unit_price = float(price_map.get(name.lower(), 0.0))
                saved_money = saved_difference * saved_unit_price
                saved_entries.append(
                    {
                        "date_saved": today,
                        "item_name": row.get("item_name") or name,
                        "save_amount": saved_difference,
                        "saved_money": (
                            saved_money if math.isfinite(saved_money) else 0.0
                        ),
                    }
                )

            removed_item_ids.append(int(removed_index) + 1)
            shopping_df = shopping_df.drop(index=removed_index).reset_index(drop=True)

    write_dataframe_to_csv(shopping_df, "shopping_list")
    write_dataframe_to_csv(purchases_df, "purchases")
    update_shopping_list_snapshot()
    update_shopping_history_snapshot()

    if saved_entries:
        saved_df = read_csv_as_dataframe("saved_items", columns=SAVED_ITEMS_COLUMNS)
        saved_df = pd.concat([saved_df, pd.DataFrame(saved_entries)], ignore_index=True)
        write_dataframe_to_csv(saved_df, "saved_items")
        update_saved_items_snapshot()
        update_analytics_summary_snapshot()
    else:
        update_analytics_summary_snapshot()

    return {"success": True, "removedItemIds": removed_item_ids}


def record_saved_item(payload: dict[str, Any]) -> dict[str, Any]:
    raw_item_name = payload.get("itemName", "")
    item_name = _normalize_item_name(raw_item_name)
    final_quantity = payload.get("finalQuantity")
    saved_quantity = payload.get("savedQuantity")
    quantity_numeric = _coerce_quantity(final_quantity, saved_quantity)

    unit_value = (
        payload.get("unit")
        or payload.get("finalUnit")
        or payload.get("savedUnit")
        or payload.get("originalUnit")
    )
    unit: str | None
    if isinstance(unit_value, str):
        unit = unit_value.strip() or None
    else:
        unit = None

    if not item_name or quantity_numeric <= 0:
        return {"success": False, "error": "Invalid item"}

    original_quantity = _coerce_quantity(
        payload.get("originalQuantity"),
        payload.get("original_quantity"),
        payload.get("requestedQuantity"),
        payload.get("requested_quantity"),
    )
    if original_quantity <= 0:
        original_quantity = quantity_numeric

    quantity_numeric = float(quantity_numeric)
    saved_amount = max(float(original_quantity) - quantity_numeric, 0.0)

    unit_price_map = _build_unit_price_map()
    unit_price = float(unit_price_map.get(item_name.lower(), 0.0))

    shopping_df = read_csv_as_dataframe("shopping_list", columns=SHOPPING_LIST_COLUMNS)
    existing_count = len(shopping_df.index)
    today = datetime.now(timezone.utc).date().isoformat()
    new_row = pd.DataFrame(
        [
            {
                "date_added": today,
                "item_name": item_name,
                "item_amount": quantity_numeric,
                "item_unit": unit,
                "saved_original_amount": (
                    float(original_quantity) if saved_amount > 0 else pd.NA
                ),
                "saved_final_amount": (
                    quantity_numeric if saved_amount > 0 else pd.NA
                ),
                "saved_amount_difference": (
                    saved_amount if saved_amount > 0 else pd.NA
                ),
                "saved_unit_price": (unit_price if saved_amount > 0 else pd.NA),
            }
        ]
    )

    shopping_df = pd.concat([shopping_df, new_row], ignore_index=True)
    write_dataframe_to_csv(shopping_df, "shopping_list")
    records = update_shopping_list_snapshot()
    created_item = next(
        (record for record in records if record.get("id") == existing_count + 1),
        records[-1] if records else None,
    )

    return {"success": created_item is not None, "item": created_item}


__all__ = [
    "add_shopping_item",
    "remove_shopping_item",
    "mark_items_bought",
    "record_saved_item",
]

