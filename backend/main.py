from __future__ import annotations

from typing import Any

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
        _format_display_item_name,
        _get_previous_week_and_year,
        _normalize_item_name,
        read_csv_as_dataframe,
        read_json_payload,
    )
    from .services import (
        _calculate_analytics_series,
        _write_frequent_items_snapshot,
        _write_weekly_recommendation_snapshot,
        add_shopping_item as service_add_shopping_item,
        analyze_waste as service_analyze_waste,
        check_waste_history as service_check_waste_history,
        log_waste as service_log_waste,
        mark_items_bought as service_mark_items_bought,
        remove_shopping_item as service_remove_shopping_item,
        record_saved_item as service_record_saved_item,
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
        _format_display_item_name,
        _get_previous_week_and_year,
        _normalize_item_name,
        read_csv_as_dataframe,
        read_json_payload,
    )
    from services import (
        _calculate_analytics_series,
        _write_frequent_items_snapshot,
        _write_weekly_recommendation_snapshot,
        add_shopping_item as service_add_shopping_item,
        analyze_waste as service_analyze_waste,
        check_waste_history as service_check_waste_history,
        log_waste as service_log_waste,
        mark_items_bought as service_mark_items_bought,
        remove_shopping_item as service_remove_shopping_item,
        record_saved_item as service_record_saved_item,
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
    return service_check_waste_history(payload)


@app.post("/add-shopping-item")
async def add_shopping_item(payload: dict[str, Any]) -> dict[str, Any]:
    """Append a shopping item to the CSV store and refresh the static snapshot."""
    return service_add_shopping_item(payload)


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
    return service_remove_shopping_item(payload)


@app.post("/mark-items-bought")
async def mark_items_bought(payload: dict[str, Any]) -> dict[str, Any]:
    """Move confirmed items into purchases CSV and refresh dependent snapshots."""
    return service_mark_items_bought(payload)


@app.post("/log-waste")
async def log_waste(payload: dict[str, Any]) -> dict[str, Any]:
    """Record a waste entry, update CSV storage, and refresh derived analytics."""
    return service_log_waste(payload)


@app.post("/record-saved-item")
async def record_saved_item(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Persist the final shopping item choice (after suggestion confirmation) and
    update shopping list CSV/JSON snapshots.
    """
    return service_record_saved_item(payload)


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
    return service_analyze_waste(payload)


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