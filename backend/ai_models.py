from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_DIR = PROJECT_ROOT / "ai"


class ModelAssetMissing(RuntimeError):
    """Raised when an expected model asset is not present on disk."""


def _resolve_asset(path: Path, description: str) -> Path:
    if not path.exists():
        raise ModelAssetMissing(f"Expected {description} at '{path}' but the file is missing.")
    return path


def _normalize_base64(image_data: str) -> bytes:
    """
    Accept base64 payloads with or without a data URL prefix and return raw bytes.
    """
    if not isinstance(image_data, str) or not image_data.strip():
        raise ValueError("Image payload is empty.")

    data = image_data.strip()
    if "," in data and data.split(",", 1)[0].startswith("data:"):
        data = data.split(",", 1)[1]

    try:
        return base64.b64decode(data, validate=True)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise ValueError("Invalid base64 image payload.") from exc


@dataclass(frozen=True)
class VegetablePrediction:
    label: Optional[str]
    confidence: float
    probabilities: Dict[str, float]


class VegetableClassifier:
    """
    Lightweight wrapper around the 4-class ResNet-18 that distinguishes tomato, potato,
    carrot, and a pooled "Other" class.
    """

    IMAGE_SIZE = 224
    MODEL_FILENAME = "tomato_potato_carrot_vs_other_resnet18.pth"

    def __init__(self, *, model_path: Optional[Path] = None, device: Optional[str] = None) -> None:
        checkpoint_path = _resolve_asset(
            model_path or AI_DIR / self.MODEL_FILENAME, "vegetable classifier checkpoint"
        )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        class_names = checkpoint.get("class_names")
        if not isinstance(class_names, Sequence) or not class_names:
            raise ValueError("Classifier checkpoint is missing `class_names` metadata.")
        self.class_names = [str(name) for name in class_names]

        # Build model architecture (matching the training notebook)
        base_model = models.resnet18(weights=None)
        in_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, len(self.class_names)),
        )
        base_model.load_state_dict(checkpoint["model_state_dict"])
        base_model.to(self.device)
        base_model.eval()

        self.model = base_model
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def predict_from_bytes(self, image_bytes: bytes, *, threshold: float = 0.55) -> VegetablePrediction:
        with Image.open(io.BytesIO(image_bytes)) as img:
            tensor = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_index = int(np.argmax(probabilities))
        top_label = self.class_names[top_index]
        top_confidence = float(probabilities[top_index])
        resolved_label: Optional[str] = top_label if top_label.lower() != "other" else None
        if top_confidence < threshold:
            resolved_label = None

        return VegetablePrediction(
            label=resolved_label,
            confidence=top_confidence,
            probabilities={cls: float(prob) for cls, prob in zip(self.class_names, probabilities)},
        )

    def predict_from_base64(self, image_data: str, *, threshold: float = 0.55) -> VegetablePrediction:
        image_bytes = _normalize_base64(image_data)
        return self.predict_from_bytes(image_bytes, threshold=threshold)


class MultiOutputLSTM(nn.Module):
    """Architecture matching the notebook used to train the weekly forecaster."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        return self.head(out)


class WeeklyForecaster:
    """
    Loads the trained multi-output LSTM to predict the next week's net tomato/potato/carrot values.
    """

    CHECKPOINT_FILENAME = "weekly_lstm_forecaster.pth"
    SCALER_FILENAME = "weekly_lstm_scaler.npy"

    def __init__(
        self,
        *,
        checkpoint_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        checkpoint_file = _resolve_asset(
            checkpoint_path or AI_DIR / self.CHECKPOINT_FILENAME, "weekly LSTM checkpoint"
        )
        scaler_file = _resolve_asset(
            scaler_path or AI_DIR / self.SCALER_FILENAME, "weekly LSTM scaler parameters"
        )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        scaler_params = np.load(scaler_file, allow_pickle=True).item()

        self.mean = np.asarray(scaler_params["mean"], dtype=np.float32)
        self.scale = np.asarray(scaler_params["scale"], dtype=np.float32)
        self.window_size = int(checkpoint["window_size"])
        self.target_columns = list(checkpoint["target_columns"])

        self.model = MultiOutputLSTM(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_layers=int(checkpoint["num_layers"]),
            dropout=float(checkpoint["dropout"]),
            output_dim=int(checkpoint["output_dim"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def forecast(self, last_window: np.ndarray) -> Dict[str, float]:
        expected_shape = (self.window_size, len(self.target_columns))
        arr = np.asarray(last_window, dtype=np.float32)
        if arr.shape != expected_shape:
            raise ValueError(f"Expected last_window with shape {expected_shape}, received {arr.shape}")

        scaled = (arr - self.mean) / self.scale
        tensor = torch.from_numpy(scaled).unsqueeze(0).to(self.device)
        pred_scaled = self.model(tensor).cpu().numpy()
        predictions = pred_scaled * self.scale + self.mean

        return {
            column: float(value)
            for column, value in zip(self.target_columns, predictions[0])
        }


@dataclass
class WeeklyDataSummary:
    recent_weeks: List[Dict[str, Any]]
    last_week: Dict[str, Any]
    previous_week: Dict[str, Any]
    two_week_net_sum: Dict[str, float]
    two_week_waste_sum: Dict[str, float]
    baseline_net_avg: Dict[str, float]
    waste_percentages: Dict[str, float]
    forecast_next_week: Optional[Dict[str, float]]
    date_generated: date

    def get_primary_focus(self) -> str:
        if not self.waste_percentages:
            return "tomato"
        return max(self.waste_percentages.items(), key=lambda item: item[1])[0].lower()


def _seasonal_context(reference_date: date) -> List[str]:
    notes: List[str] = []
    if reference_date.month in {11, 12}:
        notes.append("Festive gatherings can boost leftovers—share portions or freeze extras.")
    elif reference_date.month == 10 and reference_date.day >= 25:
        notes.append("With cozy autumn meals ahead, plan soups to use produce promptly.")
    elif reference_date.month == 2 and reference_date.day >= 10:
        notes.append("Late-winter comfort dishes are perfect for using hearty veg.")
    elif reference_date.month == 1 and reference_date.day <= 15:
        notes.append("Fresh routines are forming—batch cook with what you already have.")
    return notes


class WeeklySuggestionAgent:
    """
    Generates short-form weekly guidance using recent purchase vs waste history and the LSTM forecast.
    """

    DATA_FILENAME = "weekly_spend_waste_net.csv"
    TARGET_VEGETABLES = ("Tomato", "Potato", "Carrot")
    FRIENDLY_WEIGHT = 0.82
    OPEN_WEIGHT = 0.90
    NEUTRAL_WEIGHT = 0.65
    ENGAGING_WEIGHT = 0.75

    def __init__(
        self,
        *,
        data_path: Optional[Path] = None,
        forecaster: Optional[WeeklyForecaster] = None,
    ) -> None:
        self.data_path = _resolve_asset(
            data_path or AI_DIR / "time_series" / self.DATA_FILENAME,
            "weekly spend/waste/net dataset",
        )
        self.forecaster = forecaster

    def _load_summary(self) -> WeeklyDataSummary:
        df = pd.read_csv(self.data_path, parse_dates=["Week"])
        df = df.sort_values("Week").reset_index(drop=True)
        if len(df) < 2:
            raise ValueError("Need at least two weeks of data to produce a recommendation.")

        recent = df.tail(max(2, getattr(self.forecaster, "window_size", 2)))
        veg_cols = [f"{veg}_net" for veg in self.TARGET_VEGETABLES]
        waste_cols = [f"{veg}_waste" for veg in self.TARGET_VEGETABLES]

        two_week_recent = recent.tail(2)
        history = df.iloc[:-2] if len(df) > 2 else two_week_recent

        two_week_net_sum_series = two_week_recent[veg_cols].sum()
        two_week_waste_sum_series = two_week_recent[waste_cols].sum()

        two_week_net_sum = {
            veg: float(two_week_net_sum_series[f"{veg}_net"]) for veg in self.TARGET_VEGETABLES
        }
        two_week_waste_sum = {
            veg: float(two_week_waste_sum_series[f"{veg}_waste"]) for veg in self.TARGET_VEGETABLES
        }

        baseline_means = history[veg_cols].mean()
        baseline_net_avg = {
            veg: float(baseline_means[f"{veg}_net"]) for veg in self.TARGET_VEGETABLES
        }

        waste_percentages = {
            veg: two_week_waste_sum[veg] / max(two_week_net_sum[veg], 1e-6)
            for veg in self.TARGET_VEGETABLES
        }

        def _week_payload(row: pd.Series) -> Dict[str, Any]:
            net = {veg: float(row[f"{veg}_net"]) for veg in self.TARGET_VEGETABLES}
            waste = {veg: float(row[f"{veg}_waste"]) for veg in self.TARGET_VEGETABLES}
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

        recent_payload = [_week_payload(row) for _, row in df.tail(2).iterrows()]
        last_week_payload = recent_payload[-1]
        previous_week_payload = recent_payload[0]

        forecast_payload: Optional[Dict[str, float]] = None
        if self.forecaster is not None:
            window = recent[veg_cols].tail(self.forecaster.window_size).to_numpy(dtype=np.float32)
            if window.shape == (self.forecaster.window_size, len(self.forecaster.target_columns)):
                forecast_raw = self.forecaster.forecast(window)
                forecast_payload = {
                    veg: float(forecast_raw.get(f"{veg}_net", forecast_raw.get(veg, 0.0)))
                    for veg in self.TARGET_VEGETABLES
                }

        return WeeklyDataSummary(
            recent_weeks=recent_payload,
            last_week=last_week_payload,
            previous_week=previous_week_payload,
            two_week_net_sum=two_week_net_sum,
            two_week_waste_sum=two_week_waste_sum,
            baseline_net_avg=baseline_net_avg,
            waste_percentages=waste_percentages,
            forecast_next_week=forecast_payload,
            date_generated=df["Week"].max().date(),
        )

    def _build_message(self, summary: WeeklyDataSummary) -> str:
        def _select_phrase(options: Sequence[str], weight: float) -> str:
            if not options:
                return ""
            index = min(int(max(weight, 0.0) * len(options)), len(options) - 1)
            return options[index]

        def _word_count(text: str) -> int:
            return len(text.replace("—", " ").split())

        veg_focus = summary.get_primary_focus()
        display_name = veg_focus.capitalize()

        waste_pct = summary.waste_percentages.get(display_name, 0.0)
        forecast_rise = False
        if summary.forecast_next_week:
            forecast_val = summary.forecast_next_week.get(display_name, summary.last_week["net"][display_name])
            forecast_rise = forecast_val > summary.last_week["net"][display_name] + 0.6

        friendly_open_weight = (self.FRIENDLY_WEIGHT + self.OPEN_WEIGHT) / 2.0
        if waste_pct >= 0.2:
            options = [
                f"{display_name} leftovers linger; give them a gentle turn before adding more to the basket.",
                f"{display_name} portions wait quietly—let them lead tonight’s relaxed plate.",
                f"{display_name} resting in the fridge deserve a soft spotlight before buying fresh ones.",
                f"Open the fridge kindly to those {display_name}; enjoy them before adding to the list.",
            ]
        elif forecast_rise:
            options = [
                f"{display_name} cravings may rise—keep meals centered on what is already at home.",
                f"{display_name} might be in demand; plan a gentle plate with what’s waiting.",
                f"Expect a pull toward {display_name}; invite what you have into an easy supper.",
                f"Those {display_name} could disappear fast—share what’s ready before topping up.",
            ]
        else:
            options = [
                f"Keep the basket light; let the {veg_focus} on hand shape this week’s meals.",
                f"Stir the {veg_focus} you already have into tonight’s calm plan before restocking.",
                f"Let familiar {display_name} lead the table so cupboards stay settled.",
                f"Lean on the {display_name} resting at home to keep shopping soft and open.",
            ]
        first_sentence = _select_phrase(options, friendly_open_weight)

        closing_options = [
            "Pause before topping up; savor what’s waiting.",
            "Take a gentle breath before adding extras to the basket.",
            "Keep servings easy and finish what’s chilled first.",
            "Share what’s resting at home before drafting the next list.",
        ]
        closing_weight = (self.NEUTRAL_WEIGHT + self.ENGAGING_WEIGHT) / 2.0
        second_sentence = _select_phrase(closing_options, closing_weight)

        message = f"{first_sentence} {second_sentence}".strip()
        if _word_count(message) > 25:
            fallback_first = f"Finish the {veg_focus} on hand before buying extras."
            fallback_second = "Pause before topping up; savor what’s waiting."
            message = f"{fallback_first} {fallback_second}"
        return message.strip()

    def generate(self) -> Dict[str, Any]:
        summary = self._load_summary()
        recommendation = self._build_message(summary)
        return {
            "generated_at": summary.date_generated.isoformat(),
            "recommendation": recommendation,
            "recent_weeks": summary.recent_weeks,
            "forecast": summary.forecast_next_week,
        }


@lru_cache(maxsize=1)
def get_classifier() -> VegetableClassifier:
    return VegetableClassifier()


@lru_cache(maxsize=1)
def get_weekly_forecaster() -> WeeklyForecaster:
    return WeeklyForecaster()


@lru_cache(maxsize=1)
def get_weekly_agent() -> WeeklySuggestionAgent:
    forecaster: Optional[WeeklyForecaster]
    try:
        forecaster = get_weekly_forecaster()
    except ModelAssetMissing:
        forecaster = None
    return WeeklySuggestionAgent(forecaster=forecaster)


