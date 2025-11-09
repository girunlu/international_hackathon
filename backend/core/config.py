from __future__ import annotations

import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "database"
STATIC_DIR = BASE_DIR.parent / "frontend" / "static_files"

DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("food_waste_backend")

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

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "STATIC_DIR",
    "logger",
    "CSV_FILES",
    "JSON_FILES",
    "SHOPPING_LIST_COLUMNS",
    "SAVED_ITEMS_COLUMNS",
]

