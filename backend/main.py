from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from .ai_models import ModelAssetMissing, get_classifier, get_weekly_agent
except ImportError:  # pragma: no cover - support running as a module script
    from ai_models import ModelAssetMissing, get_classifier, get_weekly_agent

try:
    from .core import (
        SAVED_ITEMS_COLUMNS,
        SHOPPING_LIST_COLUMNS,
        logger,
        _coerce_quantity,
        _format_display_item_name,
        _format_quantity_label,
        _get_previous_week_and_year,
        _get_previous_week_bounds,
        _normalize_item_name,
        read_csv_as_dataframe,
        read_json_payload,
        write_dataframe_to_csv,
        write_json_payload,
    )
    from .services import (
        _build_unit_price_map,
        _calculate_analytics_series,
        _write_frequent_items_snapshot,
        _write_weekly_recommendation_snapshot,
        update_analytics_summary_snapshot,
        update_saved_items_snapshot,
        update_shopping_history_snapshot,
        update_shopping_list_snapshot,
        update_waste_snapshot,
    )
except ImportError:  # pragma: no cover - allow running as standalone script
    from core import (
        SAVED_ITEMS_COLUMNS,
        SHOPPING_LIST_COLUMNS,
        logger,
        _coerce_quantity,
        _format_display_item_name,
        _format_quantity_label,
        _get_previous_week_and_year,
        _get_previous_week_bounds,
        _normalize_item_name,
        read_csv_as_dataframe,
        read_json_payload,
        write_dataframe_to_csv,
        write_json_payload,
    )
    from services import (
        _build_unit_price_map,
        _calculate_analytics_series,
        _write_frequent_items_snapshot,
        _write_weekly_recommendation_snapshot,
        update_analytics_summary_snapshot,
        update_saved_items_snapshot,
        update_shopping_history_snapshot,
        update_shopping_list_snapshot,
        update_waste_snapshot,
    )
# ---------------------------------------------------------------------------
# FastAPI application (fill in endpoint logic as you implement features)
# ---------------------------------------------------------------------------

app = FastAPI(title="Food Waste Backend Skeleton")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/detect-item")
async def detect_item(payload: dict[str, Any]) -> dict[str, Any]:
    """Classify the supplied base64 image into a known vegetable name (if confident)."""
    image_data = payload.get("image")
    if not isinstance(image_data, str) or not image_data.strip():
        return {
            "item_name": None,
            "confidence": 0.0,
            "probabilities": {},
            "error": "Image payload missing or empty.",
        }

    try:
        classifier = get_classifier()
        prediction = classifier.predict_from_base64(image_data)
    except ModelAssetMissing as exc:
        logger.warning("Vegetable classifier assets missing: %s", exc)
        return {"item_name": None, "confidence": 0.0, "probabilities": {}, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to run vegetable classifier")
        return {
            "item_name": None,
            "confidence": 0.0,
            "probabilities": {},
            "error": "Unable to classify image at this time.",
        }

    resolved_name = prediction.label.lower() if prediction.label else None
    return {
        "item_name": resolved_name,
        "confidence": round(prediction.confidence, 4),
        "probabilities": prediction.probabilities,
    }


@app.post("/check-waste-history")
async def check_waste_history(payload: dict[str, Any]) -> dict[str, Any]:
    """Look up last week's waste to suggest a reduced quantity for the requested item."""
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

    total_wasted = float(relevant_waste["waste_amount"].sum()) if not relevant_waste.empty else 0.0
    if total_wasted <= 0:
        return {
            "hasWaste": False,
            "itemName": display_item_name,
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

    # ensure we provide a meaningful suggestion that's lower than requested
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


@app.post("/add-shopping-item")
async def add_shopping_item(payload: dict[str, Any]) -> dict[str, Any]:
    """Append a shopping item to the CSV store and refresh the static snapshot."""
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


@app.post("/get-shopping-list")
async def get_shopping_list() -> Any:
    """
    Return the latest shopping list snapshot (refreshing it from CSV first).
    """
    records = update_shopping_list_snapshot()
    return records


@app.post("/remove-shopping-item")
async def remove_shopping_item(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Remove an item from shopping_list.csv and refresh the static snapshot.
    """
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


@app.post("/mark-items-bought")
async def mark_items_bought(payload: dict[str, Any]) -> dict[str, Any]:
    """Move confirmed items into purchases CSV and refresh dependent snapshots."""
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

            saved_difference = float(saved_difference) if math.isfinite(saved_difference) else 0.0
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
                        "saved_money": saved_money if math.isfinite(saved_money) else 0.0,
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


@app.post("/log-waste")
async def log_waste(payload: dict[str, Any]) -> dict[str, Any]:
    """Record a waste entry, update CSV storage, and refresh derived analytics."""
    raw_item_name = payload.get("itemName")
    item_name = _normalize_item_name(raw_item_name)
    quantity_label = payload.get("quantity")
    quantity_numeric = payload.get("quantity_numeric")
    unit = payload.get("unit")
    price = payload.get("price")

    try:
        quantity_numeric = float(quantity_numeric)
    except (TypeError, ValueError):
        quantity_numeric = 0.0

    try:
        price = float(price)
    except (TypeError, ValueError):
        price = 0.0

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


@app.post("/record-saved-item")
async def record_saved_item(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Persist the final shopping item choice (after suggestion confirmation) and
    update shopping list CSV/JSON snapshots.
    """
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


@app.post("/get-shopping-history")
async def get_shopping_history() -> Any:
    snapshot = update_shopping_history_snapshot()
    return snapshot.get("shoppingTrips", [])


@app.post("/analyze-waste")
async def analyze_waste(payload: dict[str, Any]) -> Any:
    """
    Analyze last week's purchases and waste for the requested item to provide
    a suggestion for reduced quantity.
    """
    filter_value = payload.get("filter")
    time_range = payload.get("timeRange")
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
        last_week_purchases["item_amount"].sum() if not last_week_purchases.empty else 0.0
    )
    wasted_amount = last_week_waste["waste_amount"].sum() if not last_week_waste.empty else 0.0

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


@app.post("/weekly-recommendations")
async def weekly_recommendations() -> Any:
    """Generate or retrieve the current weekly food-waste suggestion snapshot."""
    existing_payload = read_json_payload(
        "weekly_recommendations", {"recommendation": None, "generated_at": None}
    )

    try:
        agent = get_weekly_agent()
        result = agent.generate()
    except ModelAssetMissing as exc:
        logger.warning("Weekly suggestion assets missing: %s", exc)
        error_payload = dict(existing_payload)
        error_payload["error"] = str(exc)
        return error_payload
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to generate weekly recommendation")
        error_payload = dict(existing_payload)
        error_payload["error"] = "Unable to generate weekly recommendation right now."
        return error_payload

    snapshot_payload = {
        "recommendation": result.get("recommendation"),
        "generated_at": result.get("generated_at"),
        "forecast": result.get("forecast"),
        "recent_weeks": result.get("recent_weeks"),
    }
    _write_weekly_recommendation_snapshot(snapshot_payload)

    return {
        "recommendation": snapshot_payload["recommendation"],
        "generated_at": snapshot_payload["generated_at"],
        "forecast": snapshot_payload["forecast"],
    }


@app.post("/get-frequent-items")
async def get_frequent_items() -> Any:
    """
    Compute the five most frequently purchased items from the previous week and
    update the `frequent_items.json` snapshot.
    """
    purchases_df = read_csv_as_dataframe(
        "purchases",
        columns=["date_purchased", "item_name", "item_amount"],
    )

    if purchases_df.empty:
        frequent_items: list[str] = []
        _write_frequent_items_snapshot(frequent_items)
        return {"items": frequent_items}

    purchases_df = purchases_df.dropna(subset=["date_purchased", "item_name"])
    if purchases_df.empty:
        frequent_items = []
        _write_frequent_items_snapshot(frequent_items)
        return {"items": frequent_items}

    purchases_df["date_purchased"] = pd.to_datetime(purchases_df["date_purchased"], errors="coerce")
    purchases_df = purchases_df.dropna(subset=["date_purchased"])
    if purchases_df.empty:
        frequent_items = []
        _write_frequent_items_snapshot(frequent_items)
        return {"items": frequent_items}

    purchases_df["item_name"] = purchases_df["item_name"].map(_normalize_item_name)
    purchases_df["week_of_year"] = purchases_df["date_purchased"].dt.isocalendar().week
    purchases_df["year"] = purchases_df["date_purchased"].dt.isocalendar().year

    prev_week, prev_year = _get_previous_week_and_year()
    filtered = purchases_df[
        (purchases_df["week_of_year"] == prev_week)
        & (purchases_df["year"] == prev_year)
    ]

    if filtered.empty:
        frequent_items = []
        _write_frequent_items_snapshot(frequent_items)
        return {"items": frequent_items}

    item_counts = (
        filtered["item_name"]
        .astype(str)
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(5)
    )

    frequent_items = [
        _format_display_item_name(name) for name in item_counts.index.tolist()
    ]

    _write_frequent_items_snapshot(frequent_items)

    return {"items": frequent_items}

@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/monthly-analytics-summary")
async def monthly_analytics_summary() -> Any:
    return update_analytics_summary_snapshot()


@app.post("/analytics")
async def analytics(payload: dict[str, Any]) -> Any:
    time_range = str(payload.get("timeRange", "weekly")).strip().lower()
    if time_range not in {"weekly", "monthly"}:
        time_range = "weekly"

    raw_item = payload.get("itemName") or payload.get("item")
    item_name: str | None = None
    if isinstance(raw_item, str):
        cleaned = raw_item.strip()
        if cleaned and cleaned.lower() not in {"general", "all"}:
            item_name = cleaned

    return _calculate_analytics_series(time_range, item_name)