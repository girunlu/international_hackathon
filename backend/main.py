from __future__ import annotations

import calendar
import json
import logging
import math
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from .ai_models import ModelAssetMissing, get_classifier, get_weekly_agent
except ImportError:  # pragma: no cover - support running as a module script
    from ai_models import ModelAssetMissing, get_classifier, get_weekly_agent

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "database"
STATIC_DIR = BASE_DIR.parent / "frontend" / "static_files"

DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
# Known dataset names → file names (fallback to raw name if not in map)
CSV_FILES = {
    "shopping_list": "shopping_list.csv",
    "purchases": "purchases.csv",
    "waste_log": "waste_log.csv",
    "saved_items": "saved_items.csv",
    "prices": "prices.csv",
}

JSON_FILES = {
    "shopping_list": "shopping_list.json",
    "purchases": "purchases.json",
    "waste_log": "waste_log.json",
    "saved_items": "saved_items.json",
    "frequent_items": "frequent_items.json",
    "shopping_history": "shopping_history.json",
    "analytics_summary": "analytics_summary.json",
    "weekly_recommendations": "weekly_recommendations.json",
}

SHOPPING_LIST_COLUMNS = [
    "date_added",
    "item_name",
    "item_amount",
    "item_unit",
    "saved_original_amount",
    "saved_final_amount",
    "saved_amount_difference",
    "saved_unit_price",
]

SAVED_ITEMS_COLUMNS = [
    "date_saved",
    "item_name",
    "save_amount",
    "saved_money",
]

_QUANTITY_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


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


def _shift_month_start(date_value: datetime.date, offset: int) -> datetime.date:
    """
    Return the first day of the month that is `offset` months away from
    the month containing `date_value`. Negative offsets move backwards.
    """
    base = date_value.replace(day=1)
    month_index = base.month - 1 + offset
    year = base.year + month_index // 12
    month = month_index % 12 + 1
    return datetime(year, month, 1).date()


def _build_unit_price_map() -> dict[str, float]:
    prices_df = read_csv_as_dataframe(
        "prices", columns=["item_name", "unit_item_price"]
    )
    if prices_df.empty:
        base_map: dict[str, float] = {}
    else:
        prices_df = prices_df.dropna(subset=["item_name", "unit_item_price"])
        if prices_df.empty:
            base_map = {}
        else:
            prices_df = prices_df.copy()
            prices_df["item_name"] = prices_df["item_name"].astype(str).str.strip()
            prices_df["normalized_item"] = prices_df["item_name"].str.lower()
            prices_df["unit_item_price"] = pd.to_numeric(
                prices_df["unit_item_price"], errors="coerce"
            ).fillna(0.0)

            base_map = (
                prices_df.set_index("normalized_item")["unit_item_price"]
                .astype(float)
                .to_dict()
            )

    return base_map


def _resolve_csv_path(name_or_filename: str) -> Path:
    filename = CSV_FILES.get(name_or_filename, name_or_filename)
    return DATA_DIR / filename


def _resolve_json_path(name_or_filename: str) -> Path:
    filename = JSON_FILES.get(name_or_filename, name_or_filename)
    return STATIC_DIR / filename


def _get_previous_week_and_year(reference: datetime | None = None) -> tuple[int, int]:
    ref = reference or datetime.now(timezone.utc)
    previous = ref - timedelta(days=7)
    iso = previous.isocalendar()
    return iso.week, iso.year


def _get_previous_week_bounds(reference: datetime | None = None) -> tuple[datetime.date, datetime.date]:
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


# ---------------------------------------------------------------------------
# Core conversion helpers (CSV ⇄ DataFrame ⇄ JSON)
# ---------------------------------------------------------------------------

def read_csv_as_dataframe(name: str, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Load a CSV file from backend/database into a DataFrame. Missing columns are
    created so downstream code always works with the expected schema.
    """
    path = _resolve_csv_path(name)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()

    if columns:
        for column in columns:
            if column not in df.columns:
                df[column] = pd.NA
        df = df.reindex(columns=columns)

    return df


def write_dataframe_to_csv(df: pd.DataFrame, name: str) -> None:
    """
    Persist a DataFrame back to backend/database. The caller controls ordering
    and schema; this helper simply writes the file to disk.
    """
    path = _resolve_csv_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def dataframe_to_json(
    df: pd.DataFrame,
    name: str,
    *,
    orient: str = "records",
    root_key: str | None = None,
) -> None:
    """
    Serialise a DataFrame into frontend/static_files/<json_name>.json.
    By default, dumps records array; optional root_key wraps the payload.
    """
    path = _resolve_json_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Any
    if orient == "records":
        payload = df.where(pd.notnull(df), None).to_dict(orient="records")
        if root_key is not None:
            payload = {root_key: payload}
    else:
        payload = df.to_dict(orient=orient)

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def json_to_dataframe(
    name: str,
    *,
    columns: Sequence[str] | None = None,
    records_key: str | None = None,
) -> pd.DataFrame:
    """
    Load frontend/static_files/<json_name>.json and convert it into a DataFrame.
    Use records_key when the JSON wraps the array inside an object.
    """
    path = _resolve_json_path(name)
    if not path.exists():
        return pd.DataFrame(columns=columns or [])

    raw = json.loads(path.read_text(encoding="utf-8"))
    if records_key:
        raw = raw.get(records_key, []) if isinstance(raw, dict) else []

    df = pd.DataFrame(raw)
    if columns:
        for column in columns:
            if column not in df.columns:
                df[column] = pd.NA
        df = df.reindex(columns=columns)
    return df


def write_json_payload(name: str, payload: Any) -> None:
    path = _resolve_json_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json_payload(name: str, default: Any) -> Any:
    path = _resolve_json_path(name)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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
    purchases_df["item_name"] = purchases_df["item_name"].astype(str).str.strip()
    purchases_df["normalized_name"] = purchases_df["item_name"].str.lower()
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
        waste_df["item_name"] = waste_df["item_name"].astype(str).str.strip()
        waste_df["normalized_name"] = waste_df["item_name"].str.lower()
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
                    "name": name,
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
    shopping_df["date_added"] = shopping_df["date_added"].fillna(
        datetime.now(timezone.utc).date().isoformat()
    )
    shopping_df["item_amount"] = pd.to_numeric(
        shopping_df["item_amount"], errors="coerce"
    ).fillna(0.0)
    shopping_df["item_name"] = shopping_df["item_name"].astype(str).str.strip()
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
                "item_name": row["item_name"],
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
    waste_df["date_wasted"] = waste_df["date_wasted"].fillna(
        datetime.now(timezone.utc).date().isoformat()
    )
    waste_df["item_name"] = waste_df["item_name"].astype(str).str.strip()
    waste_df["waste_amount"] = pd.to_numeric(
        waste_df["waste_amount"], errors="coerce"
    ).fillna(0.0)

    records = []
    for idx, row in waste_df.reset_index(drop=True).iterrows():
        records.append(
            {
                "id": idx + 1,
                "item_name": row["item_name"],
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
    saved_df["date_saved"] = saved_df["date_saved"].fillna(
        datetime.now(timezone.utc).date().isoformat()
    )
    saved_df["item_name"] = saved_df["item_name"].astype(str).str.strip()
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
                "item_name": row["item_name"],
                "save_amount": float(row["save_amount"]),
                "saved_money": float(row["saved_money"]),
            }
        )

    write_json_payload("saved_items", records)
    return records


def _calculate_analytics_series(
    time_range: str, item_name: str | None
) -> dict[str, Any]:
    today = datetime.now(timezone.utc).date()
    normalized_range = "monthly" if time_range == "monthly" else "weekly"

    if normalized_range == "monthly":
        period_count = 6
        current_month_start = today.replace(day=1)
        bucket_definitions: list[tuple[datetime.date, datetime.date, str]] = []
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
    if item_name:
        normalized_filter = item_name.strip().lower() or None

    all_items: set[str] = set()

    purchases_df = read_csv_as_dataframe(
        "purchases", columns=["date_purchased", "item_name", "item_amount"]
    )
    if not purchases_df.empty:
        purchases_df = purchases_df.dropna(subset=["date_purchased", "item_name"])
        purchases_df = purchases_df.copy()
        purchases_df["item_name"] = purchases_df["item_name"].astype(str).str.strip()
        all_items.update(
            name for name in purchases_df["item_name"] if isinstance(name, str) and name
        )
        purchases_df["normalized_name"] = purchases_df["item_name"].str.lower()
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
        saved_df["item_name"] = saved_df["item_name"].astype(str).str.strip()
        all_items.update(
            name for name in saved_df["item_name"] if isinstance(name, str) and name
        )
        saved_df["normalized_name"] = saved_df["item_name"].str.lower()
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
        waste_df["item_name"] = waste_df["item_name"].astype(str).str.strip()
        all_items.update(
            name for name in waste_df["item_name"] if isinstance(name, str) and name
        )
        waste_df["normalized_name"] = waste_df["item_name"].str.lower()
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

    spent_by_date: dict[datetime.date, float] = {}
    if not purchases_df.empty:
        spent_by_date = (
            purchases_df.groupby("date_only")["item_total_price"].sum().to_dict()
        )

    saved_by_date: dict[datetime.date, float] = {}
    if not saved_df.empty:
        saved_by_date = (
            saved_df.groupby("date_only")["saved_money"].sum().to_dict()
        )

    wasted_by_date: dict[datetime.date, float] = {}
    if not waste_df.empty:
        wasted_by_date = (
            waste_df.groupby("date_only")["wasted_price"].sum().to_dict()
        )

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

    available_items = ["general"] + sorted(all_items, key=lambda value: value.lower())

    return {
        "timeRange": time_range,
        "itemName": item_name or "general",
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
        month_purchases["item_name"] = month_purchases["item_name"].astype(str).str.strip()
        month_purchases["normalized_name"] = month_purchases["item_name"].str.lower()
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
                    "name": str(display_names.get(normalized_name, normalized_name)),
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
        month_saved["item_name"] = month_saved["item_name"].astype(str).str.strip()
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
                    "name": str(item_name),
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
        month_waste["item_name"] = month_waste["item_name"].astype(str).str.strip()
        month_waste["normalized_name"] = month_waste["item_name"].str.lower()
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
    """
    TODO: implement waste history lookup using pandas DataFrames.
    """
    raw_name = payload.get("itemName")
    quantity_numeric = _coerce_quantity(
        payload.get("quantity_numeric"), payload.get("quantity")
    )
    unit_value = payload.get("unit")

    item_name = str(raw_name).strip() if isinstance(raw_name, str) else ""
    if not item_name:
        return {
            "hasWaste": False,
            "itemName": raw_name,
            "wastedAmount": None,
            "wastedQuantityNumeric": None,
            "suggestedAmount": None,
            "suggestedQuantityNumeric": None,
            "suggestedUnit": unit_value if isinstance(unit_value, str) else None,
        }

    normalized_item = item_name.lower()
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
            "itemName": item_name,
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
            "itemName": item_name,
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
            "itemName": item_name,
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
    suggested_quantity = max(baseline_quantity - total_wasted, 0.0)
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
        "itemName": item_name,
        "wastedAmount": wasted_amount_label,
        "wastedQuantityNumeric": total_wasted,
        "suggestedAmount": suggested_amount_label,
        "suggestedQuantityNumeric": suggested_quantity_numeric,
        "suggestedUnit": unit,
    }


@app.post("/add-shopping-item")
async def add_shopping_item(payload: dict[str, Any]) -> dict[str, Any]:
    """
    TODO: update shopping_list.csv and the matching static JSON snapshot here.
    """
    item_name = str(payload.get("itemName", "")).strip()
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
    """
    TODO: move items from shopping_list.csv to purchases.csv and update snapshots.
    """
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
        name = str(item.get("item_name", "")).strip()
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
    """
    TODO: append to waste_log.csv and refresh waste-related JSON files.
    """
    item_name = payload.get("itemName")
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
    item_name = str(payload.get("itemName", "")).strip()
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
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(5)
    )

    frequent_items = item_counts.index.tolist()

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