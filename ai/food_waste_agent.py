#!/usr/bin/env python
"""
Weekly food waste coaching agent.

This script builds a LangChain-powered assistant that:
1. Summarises the last two weeks of purchasing + waste data from `time_series/weekly_spend_waste_net.csv`.
2. Runs the weekly LSTM forecaster to anticipate next week's tomato/potato/carrot demand.
3. Checks upcoming seasonal events based on the current date.
4. Sends the contextual payload to an OpenAI-backed chat model and prints a calm, friendly suggestion
   focused on reducing food waste.

Requirements (already satisfied if you followed the training notebooks):
    pip install torch torchvision pandas numpy pillow langchain langchain-openai python-dotenv

Make sure `OPENAI_API_KEY` is available in your environment (see `.env` usage in the notebooks).

Example:
    python food_waste_agent.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


DATA_ROOT = Path(__file__).resolve().parent
FORECASTER_CHECKPOINT = DATA_ROOT / "weekly_lstm_forecaster.pth"
FORECASTER_SCALER = DATA_ROOT / "weekly_lstm_scaler.npy"
WEEKLY_DATA_CSV = DATA_ROOT / "time_series" / "weekly_spend_waste_net.csv"


def _ensure_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {description} at '{path}', but it does not exist.")


class MultiOutputLSTM(nn.Module):
    """Same architecture used in `veg_weekly_lstm.ipynb`."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, output_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        return self.head(out)


class WeeklyForecaster:
    """Loads the trained weekly LSTM and scaler to infer next week's net requirements."""

    def __init__(
        self,
        checkpoint_path: Path = FORECASTER_CHECKPOINT,
        scaler_path: Path = FORECASTER_SCALER,
        device: Optional[str] = None,
    ) -> None:
        _ensure_file(checkpoint_path, "LSTM checkpoint")
        _ensure_file(scaler_path, "LSTM scaler params")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        scaler_params = np.load(scaler_path, allow_pickle=True).item()

        self.mean = scaler_params["mean"].astype(np.float32)
        self.scale = scaler_params["scale"].astype(np.float32)
        self.window_size = checkpoint["window_size"]
        self.target_columns = checkpoint["target_columns"]

        self.model = MultiOutputLSTM(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            output_dim=checkpoint["output_dim"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def forecast(self, last_window: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(last_window, dtype=np.float32)
        expected_shape = (self.window_size, len(self.target_columns))
        if arr.shape != expected_shape:
            raise ValueError(f"Expected last_window with shape {expected_shape}, received {arr.shape}")

        scaled = (arr - self.mean) / self.scale
        tensor = torch.from_numpy(scaled).unsqueeze(0).to(self.device)
        pred_scaled = self.model(tensor).cpu().numpy()
        preds = pred_scaled * self.scale + self.mean
        return {col: float(val) for col, val in zip(self.target_columns, preds[0])}


@dataclass
class WeeklyDataSummary:
    recent_weeks: List[Dict[str, object]]
    last_week: Dict[str, object]
    previous_week: Dict[str, object]
    two_week_net_sum: Dict[str, float]
    two_week_waste_sum: Dict[str, float]
    two_week_net_avg: Dict[str, float]
    baseline_net_avg: Dict[str, float]
    waste_percentages: Dict[str, float]
    forecast_next_week: Optional[Dict[str, float]] = None
    date_generated: dt.date = dt.date.today()

    def to_payload(self) -> Dict[str, object]:
        return {
            "date_generated": self.date_generated.isoformat(),
            "recent_weeks": self.recent_weeks,
            "last_week": self.last_week,
            "previous_week": self.previous_week,
            "two_week_net_sum": self.two_week_net_sum,
            "two_week_waste_sum": self.two_week_waste_sum,
            "two_week_net_avg": self.two_week_net_avg,
            "baseline_net_avg": self.baseline_net_avg,
            "waste_percentages": self.waste_percentages,
            "forecast_next_week": self.forecast_next_week,
        }


def load_weekly_summary(
    csv_path: Path = WEEKLY_DATA_CSV,
    forecaster: Optional[WeeklyForecaster] = None,
) -> WeeklyDataSummary:
    _ensure_file(csv_path, "weekly data CSV")

    df = pd.read_csv(csv_path, parse_dates=["Week"])
    df = df.sort_values("Week").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Need at least two weeks of data to build a summary.")

    recent = df.tail(2)
    history = df.iloc[:-2] if len(df) > 2 else df.tail(2)

    veg_cols = ["Tomato_net", "Potato_net", "Carrot_net"]
    waste_cols = ["Tomato_waste", "Potato_waste", "Carrot_waste"]
    base_names = [col.replace("_net", "") for col in veg_cols]

    two_week_net_sum_series = recent[veg_cols].sum()
    two_week_waste_sum_series = recent[waste_cols].sum()
    two_week_net_sum = {name: float(two_week_net_sum_series[f"{name}_net"]) for name in base_names}
    two_week_waste_sum = {name: float(two_week_waste_sum_series[f"{name}_waste"]) for name in base_names}
    two_week_net_avg = {name: total / 2.0 for name, total in two_week_net_sum.items()}

    baseline_means = history[veg_cols].mean()
    baseline_net_avg = {name: float(baseline_means[f"{name}_net"]) for name in base_names}

    waste_percentages = {
        name: two_week_waste_sum[name] / max(two_week_net_sum[name], 1e-6) for name in base_names
    }

    def _week_payload(row: pd.Series) -> Dict[str, object]:
        net = {name: float(row[f"{name}_net"]) for name in base_names}
        waste = {name: float(row[f"{name}_waste"]) for name in base_names}
        total_net = sum(net.values())
        total_waste = sum(waste.values())
        waste_ratio = total_waste / max(total_net, 1e-6)
        return {
            "week_ending": row["Week"].date().isoformat(),
            "net": net,
            "waste": waste,
            "total_net": total_net,
            "total_waste": total_waste,
            "waste_ratio": waste_ratio,
        }

    recent_weeks = [_week_payload(row) for _, row in recent.iterrows()]
    last_week_payload = _week_payload(recent.iloc[-1])
    previous_week_payload = _week_payload(recent.iloc[0])

    forecast_payload = None
    if forecaster is not None:
        window = recent[veg_cols].to_numpy(dtype=np.float32)
        forecast_payload = forecaster.forecast(window)

    summary = WeeklyDataSummary(
        recent_weeks=recent_weeks,
        last_week=last_week_payload,
        previous_week=previous_week_payload,
        two_week_net_sum=two_week_net_sum,
        two_week_waste_sum=two_week_waste_sum,
        two_week_net_avg=two_week_net_avg,
        baseline_net_avg=baseline_net_avg,
        waste_percentages=waste_percentages,
        forecast_next_week=forecast_payload,
        date_generated=recent["Week"].max().date(),
    )
    return summary


def seasonal_context(reference_date: dt.date) -> List[str]:
    notes: List[str] = []
    if reference_date.month == 12 or reference_date.month == 11:
        notes.append("Festive gatherings like Christmas can increase leftovers—plan portion sizes carefully.")
    elif reference_date.month == 10 and reference_date.day >= 25:
        notes.append("Halloween leftovers (soups, roasts) can be repurposed into hearty stews.")
    elif reference_date.month == 1 and reference_date.day <= 15:
        notes.append("New Year routines are forming—use planned leftovers to stay on track.")
    elif reference_date.month == 2 and reference_date.day >= 10:
        notes.append("Late-winter comfort dishes help use extra produce before spring menus arrive.")
    return notes


def build_agent_response(
    summary: WeeklyDataSummary,
    extra_context: List[str],
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.35,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Load it via your environment or a .env file.")

    llm = ChatOpenAI(model=model_name, temperature=temperature, max_retries=3)
    payload = summary.to_payload()
    payload["seasonal_context"] = extra_context

    if summary.forecast_next_week is not None:
        payload["forecast_vs_last_week"] = {
            veg: summary.forecast_next_week.get(veg, 0.0) - summary.last_week["net"].get(veg, 0.0)
            for veg in summary.two_week_net_sum.keys()
        }

    payload["last_week_vs_baseline"] = {
        veg: summary.last_week["net"].get(veg, 0.0) - summary.baseline_net_avg.get(veg, 0.0)
        for veg in summary.baseline_net_avg.keys()
    }
    payload["waste_change_week_over_week"] = {
        veg: summary.last_week["waste"].get(veg, 0.0) - summary.previous_week["waste"].get(veg, 0.0)
        for veg in summary.last_week["waste"].keys()
    }

    system_message = SystemMessage(
        content=(
            "You are a calm, indirect household food-planning guide. "
            "Offer warm, encouraging suggestions that help reduce food waste while celebrating small wins. "
            "Compare last week, the week before, baseline averages, and the upcoming forecast to find balanced advice. "
            "Keep the response under 30 words, limited to at most two sentences, and provide one or two practical, friendly ideas."
        )
    )

    human_message = HumanMessage(
        content=(
            "Use the following JSON data to craft a weekly food-waste tip.\n"
            "Reference the vegetables explicitly when helpful and recommend sustainable uses.\n"
            "JSON payload:\n"
            f"{json.dumps(payload, indent=2)}"
        )
    )

    response = llm.invoke([system_message, human_message])
    return response.content.strip()


def main() -> None:
    # Prepare forecaster and data summary
    forecaster = WeeklyForecaster()

    summary = load_weekly_summary(forecaster=forecaster)

    # Seasonal / situational context
    context_notes = seasonal_context(summary.date_generated)

    # Generate suggestion via LangChain
    suggestion = build_agent_response(
        summary=summary,
        extra_context=context_notes,
    )

    print("=" * 72)
    print("Weekly Food Waste Insight")
    print("=" * 72)
    print(suggestion)
    print("\nData snapshot:")
    print(json.dumps(summary.to_payload(), indent=2))
    if context_notes:
        print("\nSeasonal context hints:")
        for note in context_notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()


