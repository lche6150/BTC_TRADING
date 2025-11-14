"""Utility to load the cached GRU model and produce probability forecasts."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from tensorflow.keras.models import load_model


class GRUSignalModel:
    """Wraps the cached GRU classifier for fast inference."""

    def __init__(self, model_dir: Path | str = Path("models/gru_signal")) -> None:
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "model.keras"
        self.scaler_path = self.model_dir / "scaler.pkl"
        self.meta_path = self.model_dir / "meta.json"

        if not self.model_path.exists() or not self.scaler_path.exists():
            raise FileNotFoundError(
                "Cached GRU model not found. Please run trading_backtest.py once to train and cache the model."
            )

        self.model = load_model(self.model_path)
        with self.scaler_path.open("rb") as f:
            self.scaler = pickle.load(f)

        self.available_features: list[str]
        self.sequence_length: int
        self.best_threshold: float

        if self.meta_path.exists():
            with self.meta_path.open("r") as f:
                meta = json.load(f)
            self.available_features = meta.get("available_features", [])
            self.sequence_length = int(meta.get("sequence_length", 35))
            self.best_threshold = float(meta.get("best_threshold", 0.5))
        else:
            # Fallback defaults; should rarely occur.
            self.available_features = []
            self.sequence_length = 35
            self.best_threshold = 0.5

    def predict_dataframe(self, feature_df) -> Optional[Dict[str, float]]:
        """
        Produce probability outputs using the latest rows of the provided feature dataframe.

        feature_df should already contain the engineered columns in the same order as training.
        Returns a dict with 'prob_up', 'prob_down', 'confidence' or None if insufficient history.
        """

        if not self.available_features:
            # If metadata missing, use intersection of columns.
            self.available_features = [c for c in feature_df.columns if c not in {"datetime"}]

        subset = feature_df[self.available_features].dropna()
        if len(subset) < self.sequence_length:
            return None

        window = subset.iloc[-self.sequence_length:]
        scaled = self.scaler.transform(window)
        X = scaled.reshape(1, self.sequence_length, len(self.available_features))
        prob_up = float(self.model.predict(X, verbose=0).flatten()[0])
        prob_down = 1.0 - prob_up
        confidence = max(prob_up, prob_down)
        return {
            "prob_up": prob_up,
            "prob_down": prob_down,
            "confidence": confidence,
        }

    def predict_series(self, feature_df, limit: int = 500) -> Optional[Dict[str, Dict[str, float]]]:
        """Return probabilities for the most recent rows (up to limit)."""
        if not self.available_features:
            self.available_features = [c for c in feature_df.columns if c not in {"datetime"}]

        subset = feature_df[self.available_features].dropna()
        if len(subset) < self.sequence_length:
            return None

        subset = subset.iloc[-(self.sequence_length + limit):]
        values = subset.values
        sequences = []
        indices = []
        for idx in range(self.sequence_length, len(values)):
            seq = values[idx - self.sequence_length: idx]
            sequences.append(seq)
            indices.append(subset.index[idx])

        if not sequences:
            return None

        scaled = self.scaler.transform(np.vstack(sequences))
        scaled = scaled.reshape(len(sequences), self.sequence_length, len(self.available_features))
        probs = self.model.predict(scaled, verbose=0).flatten()
        results = {}
        for idx_val, prob_up in zip(indices, probs):
            prob_down = 1.0 - prob_up
            results[idx_val] = {
                "prob_up": float(prob_up),
                "prob_down": float(prob_down),
                "confidence": float(max(prob_up, prob_down)),
            }
        return results
