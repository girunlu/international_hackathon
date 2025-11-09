from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

import pandas as pd

try:
    from ..core.config import SAVED_ITEMS_COLUMNS, SHOPPING_LIST_COLUMNS
    from ..core.helpers import _format_display_item_name, _normalize_item_name
    from ..core.storage import (
        read_csv_as_dataframe,
        read_json_payload,
        write_dataframe_to_csv,
        write_json_payload,
    )
    from .pricing import _build_unit_price_map
except ImportError:  # pragma: no cover - allow direct script execution
    from core.config import SAVED_ITEMS_COLUMNS, SHOPPING_LIST_COLUMNS
    from core.helpers import _format_display_item_name, _normalize_item_name
    from core.storage import (
        read_csv_as_dataframe,
        read_json_payload,
        write_dataframe_to_csv,
        write_json_payload,
    )
    from services.pricing import _build_unit_price_map


def _write_frequent_items_snapshot(items: Sequence[str]) -> None:
    write_json_payload("frequent_items", {"items": list(items)})


def _write_weekly_recommendation_snapshot(payload: dict[str, Any]) -> None:
    write_json_payload("weekly_recommendations", payload)

    summary_payload = read_json_payload("analytics_summary", {})
    if not isinstance(summary_payload, dict):
        summary_payload = {}

    summary_payload["weeklySuggestion"] = payload.get("recommendation")
    summary_payload["weeklySuggestionGeneratedAt"] = payload.get("generated_at")

    write_json_payload("analytics_summary", summary_payload)


def update_shopping_history_snapshot() -> dict[str, Any]:
    purchases_df = read_csv_as_dataframe(
        "purchases", columns=["date_purchased", "item_name", "item_amount"]
    )
    prices_df = read_csv_as_dataframe(
        "prices", columns=["item_name", "unit_item_price"]
    )
    waste_df = read_csv_as_dataframe(
        "waste_log", columns=["date_wasted", "item_name", "waste_amount"]
    )

    if purchases_df.empty:
        snapshot = {"shoppingTrips": []}
        write_json_payload("shopping_history", snapshot)
        return snapshot

    purchases_df = purchases_df.dropna(subset=["date_purchased", "item_name"])
    purchases_df["date_purchased"] = pd.to_datetime(
        purchases_df["date_purchased"], errors="coerce"
    )
    purchases_df = purchases_df.dropna(subset=["date_purchased"])
    if purchases_df.empty:
        snapshot = {"shoppingTrips": []}
        write_json_payload("shopping_history", snapshot)
        return snapshot

    purchases_df["item_amount"] = pd.to_numeric(
        purchases_df["item_amount"], errors="coerce"
    ).fillna(0.0)
    purchases_df["item_name"] = purchases_df["item_name"].map(_normalize_item_name)
    purchases_df["normalized_name"] = purchases_df["item_name"]
    purchases_df["week"] = purchases_df["date_purchased"].dt.isocalendar().week.astype(
        int
    )
    purchases_df["year"] = purchases_df["date_purchased"].dt.isocalendar().year.astype(
        int
    )

    price_map: dict[str, float] = {}
    if not prices_df.empty:
        prices_df = prices_df.dropna(subset=["item_name", "unit_item_price"])
        prices_df["item_name"] = prices_df["item_name"].astype(str).str.strip()
        prices_df["unit_item_price"] = pd.to_numeric(
            prices_df["unit_item_price"], errors="coerce"
        ).fillna(0.0)
        prices_df["normalized_item"] = prices_df["item_name"].str.lower()
        price_map = (
            prices_df.set_index("normalized_item")["unit_item_price"]
            .astype(float)
            .to_dict()
        )

    purchases_df["unit_price"] = purchases_df["normalized_name"].map(price_map).fillna(
        0.0
    )

    unit_price_lookup: dict[str, float] = dict(price_map)
    if not purchases_df.empty:
        fallback_prices = (
            purchases_df[purchases_df["unit_price"] > 0]
            .groupby("normalized_name")["unit_price"]
            .first()
            .to_dict()
        )
        for name, price in fallback_prices.items():
            unit_price_lookup.setdefault(name, float(price))

    missing_unit_mask = purchases_df["unit_price"] <= 0
    if missing_unit_mask.any():
        purchases_df.loc[missing_unit_mask, "unit_price"] = purchases_df.loc[
            missing_unit_mask, "normalized_name"
        ].map(unit_price_lookup).fillna(0.0)

    purchases_df["item_total_price"] = (
        purchases_df["item_amount"] * purchases_df["unit_price"]
    )

    weekly_item_total_price_map: dict[tuple[str, int, int], float] = {}
    weekly_overall_total_price_map: dict[tuple[int, int], float] = {}
    if not purchases_df.empty:
        item_totals_series = purchases_df.groupby(
            ["normalized_name", "year", "week"]
        )["item_total_price"].sum()
        weekly_item_total_price_map = {
            (name, int(year), int(week)): float(total)
            for (name, year, week), total in item_totals_series.items()
        }

        overall_totals_series = purchases_df.groupby(["year", "week"])[
            "item_total_price"
        ].sum()
        weekly_overall_total_price_map = {
            (int(year), int(week)): float(total)
            for (year, week), total in overall_totals_series.items()
        }

    waste_totals_map: dict[tuple[str, int, int], float] = {}
    weekly_item_waste_price_map: dict[tuple[str, int, int], float] = {}
    weekly_overall_waste_price_map: dict[tuple[int, int], float] = {}
    if not waste_df.empty:
        waste_df = waste_df.dropna(subset=["date_wasted", "item_name"])
        waste_df = waste_df.copy()
        waste_df["date_wasted"] = pd.to_datetime(
            waste_df["date_wasted"], errors="coerce"
        )
        waste_df = waste_df.dropna(subset=["date_wasted"])
        waste_df["item_name"] = waste_df["item_name"].map(_normalize_item_name)
        waste_df["normalized_name"] = waste_df["item_name"]
        waste_df["waste_amount"] = pd.to_numeric(
            waste_df["waste_amount"], errors="coerce"
        ).fillna(0.0)
        waste_df["week"] = waste_df["date_wasted"].dt.isocalendar().week.astype(int)
        waste_df["year"] = waste_df["date_wasted"].dt.isocalendar().year.astype(int)
        waste_totals_series = waste_df.groupby(
            ["normalized_name", "year", "week"]
        )["waste_amount"].sum()

        for (name, year, week), amount in waste_totals_series.items():
            key = (str(name), int(year), int(week))
            amount_float = float(amount)
            waste_totals_map[key] = amount_float
            unit_price = float(unit_price_lookup.get(str(name), 0.0))
            wasted_price = amount_float * unit_price
            weekly_item_waste_price_map[key] = wasted_price
            weekly_overall_waste_price_map[(int(year), int(week))] = (
                weekly_overall_waste_price_map.get((int(year), int(week)), 0.0)
                + wasted_price
            )

    grouped = purchases_df.groupby(
        purchases_df["date_purchased"].dt.date, sort=False
    )

    raw_trips: list[dict[str, Any]] = []
    for trip_date, group in grouped:
        items: list[dict[str, Any]] = []
        total_price = 0.0
        trip_week = int(trip_date.isocalendar().week)
        trip_year = int(trip_date.isocalendar().year)

        aggregated = (
            group.groupby("normalized_name", sort=False)
            .agg(
                item_name=("item_name", "last"),
                total_amount=("item_amount", "sum"),
                total_price=("item_total_price", "sum"),
            )
            .reset_index()
        )

        for _, row in aggregated.iterrows():
            name = str(row["item_name"])
            amount = float(row["total_amount"])
            normalized_name = str(row["normalized_name"])
            item_total_price = float(row["total_price"])

            weekly_item_total_price = float(
                weekly_item_total_price_map.get(
                    (normalized_name, trip_year, trip_week), 0.0
                )
            )
            weekly_item_wasted_price = float(
                weekly_item_waste_price_map.get(
                    (normalized_name, trip_year, trip_week), 0.0
                )
            )
            weekly_overall_total_price = float(
                weekly_overall_total_price_map.get((trip_year, trip_week), 0.0)
            )
            weekly_overall_wasted_price = float(
                weekly_overall_waste_price_map.get((trip_year, trip_week), 0.0)
            )

            items.append(
                {
                    "name": _format_display_item_name(name),
                    "amount": amount,
                    "totalPrice": item_total_price,
                    "wastedPrice": weekly_item_wasted_price,
                    "weeklyTotalPrice": weekly_overall_total_price,
                    "weeklyWastedPrice": weekly_overall_wasted_price,
                    "itemWeeklyTotalPrice": weekly_item_total_price,
                    "itemWeeklyWastedPrice": weekly_item_wasted_price,
                }
            )
            total_price += item_total_price

        raw_trips.append(
            {
                "date": trip_date.isoformat(),
                "_date": trip_date,
                "items": items,
                "total": total_price,
            }
        )

    raw_trips.sort(key=lambda trip: trip["_date"], reverse=True)

    trips: list[dict[str, Any]] = []
    for trip_id, entry in enumerate(raw_trips, start=1):
        entry.pop("_date", None)
        entry["id"] = trip_id
        trips.append(entry)

    snapshot = {"shoppingTrips": trips}
    write_json_payload("shopping_history", snapshot)
    return snapshot


def update_shopping_list_snapshot() -> list[dict[str, Any]]:
    shopping_df = read_csv_as_dataframe("shopping_list", columns=SHOPPING_LIST_COLUMNS)
    if shopping_df.empty:
        records: list[dict[str, Any]] = []
        write_json_payload("shopping_list", records)
        return records

    shopping_df = shopping_df.copy()
    normalized_names = shopping_df["item_name"].map(_normalize_item_name)
    if not normalized_names.equals(shopping_df["item_name"]):
        shopping_df["item_name"] = normalized_names
        write_dataframe_to_csv(shopping_df, "shopping_list")
    else:
        shopping_df["item_name"] = normalized_names
    shopping_df["date_added"] = shopping_df["date_added"].fillna(
        datetime.now(timezone.utc).date().isoformat()
    )
    shopping_df["item_amount"] = pd.to_numeric(
        shopping_df["item_amount"], errors="coerce"
    ).fillna(0.0)
    if "item_unit" in shopping_df.columns:
        shopping_df["item_unit"] = shopping_df["item_unit"].astype(str).str.strip()
        shopping_df.loc[shopping_df["item_unit"] == "", "item_unit"] = pd.NA
    else:
        shopping_df["item_unit"] = pd.NA

    records = []
    for idx, row in shopping_df.reset_index(drop=True).iterrows():
        raw_unit = row["item_unit"]
        pretty_unit = (
            str(raw_unit).strip() if isinstance(raw_unit, str) and raw_unit.strip() else None
        )
        records.append(
            {
                "id": idx + 1,
                "item_name": _format_display_item_name(row["item_name"]),
                "quantity": f"{row['item_amount']:.2f}",
                "quantity_numeric": float(row["item_amount"]),
                "quantity_unit": pretty_unit,
                "date_added": str(row["date_added"]),
            }
        )

    write_json_payload("shopping_list", records)
    return records


def update_waste_snapshot() -> list[dict[str, Any]]:
    waste_df = read_csv_as_dataframe(
        "waste_log", columns=["date_wasted", "item_name", "waste_amount"]
    )
    if waste_df.empty:
        records: list[dict[str, Any]] = []
        write_json_payload("waste_log", records)
        return records

    waste_df = waste_df.copy()
    normalized_names = waste_df["item_name"].map(_normalize_item_name)
    if not normalized_names.equals(waste_df["item_name"]):
        waste_df["item_name"] = normalized_names
        write_dataframe_to_csv(waste_df, "waste_log")
    else:
        waste_df["item_name"] = normalized_names
    waste_df["date_wasted"] = waste_df["date_wasted"].fillna(
        datetime.now(timezone.utc).date().isoformat()
    )
    waste_df["waste_amount"] = pd.to_numeric(
        waste_df["waste_amount"], errors="coerce"
    ).fillna(0.0)

    records = []
    for idx, row in waste_df.reset_index(drop=True).iterrows():
        records.append(
            {
                "id": idx + 1,
                "item_name": _format_display_item_name(row["item_name"]),
                "waste_amount": float(row["waste_amount"]),
                "date_wasted": str(row["date_wasted"]),
            }
        )

    write_json_payload("waste_log", records)
    return records


def update_saved_items_snapshot() -> list[dict[str, Any]]:
    saved_df = read_csv_as_dataframe("saved_items", columns=SAVED_ITEMS_COLUMNS)
    if saved_df.empty:
        records: list[dict[str, Any]] = []
        write_json_payload("saved_items", records)
        return records

    saved_df = saved_df.copy()
    normalized_names = saved_df["item_name"].map(_normalize_item_name)
    if not normalized_names.equals(saved_df["item_name"]):
        saved_df["item_name"] = normalized_names
        write_dataframe_to_csv(saved_df, "saved_items")
    else:
        saved_df["item_name"] = normalized_names
    saved_df["date_saved"] = saved_df["date_saved"].fillna(
        datetime.now(timezone.utc).date().isoformat()
    )
    saved_df["save_amount"] = pd.to_numeric(
        saved_df["save_amount"], errors="coerce"
    ).fillna(0.0)
    saved_df["saved_money"] = pd.to_numeric(
        saved_df["saved_money"], errors="coerce"
    ).fillna(0.0)

    records: list[dict[str, Any]] = []
    for idx, row in saved_df.reset_index(drop=True).iterrows():
        records.append(
            {
                "id": idx + 1,
                "date_saved": str(row["date_saved"]),
                "item_name": _format_display_item_name(row["item_name"]),
                "save_amount": float(row["save_amount"]),
                "saved_money": float(row["saved_money"]),
            }
        )

    write_json_payload("saved_items", records)
    return records


__all__ = [
    "_write_frequent_items_snapshot",
    "_write_weekly_recommendation_snapshot",
    "update_shopping_history_snapshot",
    "update_shopping_list_snapshot",
    "update_waste_snapshot",
    "update_saved_items_snapshot",
]

