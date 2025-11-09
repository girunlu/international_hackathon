from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

try:
    from .config import CSV_FILES, DATA_DIR, JSON_FILES, STATIC_DIR
except ImportError:  # pragma: no cover - allow direct script execution
    from core.config import CSV_FILES, DATA_DIR, JSON_FILES, STATIC_DIR


def _resolve_csv_path(name_or_filename: str) -> Path:
    filename = CSV_FILES.get(name_or_filename, name_or_filename)
    return DATA_DIR / filename


def _resolve_json_path(name_or_filename: str) -> Path:
    filename = JSON_FILES.get(name_or_filename, name_or_filename)
    return STATIC_DIR / filename


def read_csv_as_dataframe(
    name: str, *, columns: Sequence[str] | None = None
) -> pd.DataFrame:
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


__all__ = [
    "_resolve_csv_path",
    "_resolve_json_path",
    "read_csv_as_dataframe",
    "write_dataframe_to_csv",
    "dataframe_to_json",
    "json_to_dataframe",
    "write_json_payload",
    "read_json_payload",
]

