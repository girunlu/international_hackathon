from __future__ import annotations

import pandas as pd

try:
    from ..core.storage import read_csv_as_dataframe
except ImportError:  # pragma: no cover - allow direct script execution
    from core.storage import read_csv_as_dataframe


def _build_unit_price_map() -> dict[str, float]:
    prices_df = read_csv_as_dataframe("prices", columns=["item_name", "unit_item_price"])
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


__all__ = ["_build_unit_price_map"]

