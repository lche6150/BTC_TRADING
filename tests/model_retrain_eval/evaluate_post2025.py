#!/usr/bin/env python3
"""
Evaluate the GRU model trained on pre-2025 data against post-2025 samples.

Usage:
    PYTHONPATH=. python tests/model_retrain_eval/evaluate_post2025.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.data.database import TradingDatabase
from src.data.features import EnhancedFeatureEngine
from src.models.stop_aligned_labeler import generate_stop_aligned_labels

MODEL_FEATURES = [
    "rsi_14", "rsi_14_1h", "rsi_14_4h",
    "bb_position", "bb_position_1h", "bb_position_4h",
    "macd_signal", "macd_signal_1h",
    "volume_sma_ratio", "volume_sma_ratio_1h",
    "price_position", "hl_ratio", "momentum_10", "momentum_20",
    "volatility", "volatility_1h", "atr_normalized",
    "williams_r", "stoch_k", "cci", "adx", "roc",
]

POST_DB = Path("data/trading_data_post2025.db")
ARTIFACT_DIR = Path("tests/model_retrain_eval/artifacts")
RESULTS_PATH = Path("tests/model_retrain_eval/post2025_metrics.json")


def load_artifacts():
    model_path = ARTIFACT_DIR / "gru_pre2025.keras"
    scaler_path = ARTIFACT_DIR / "scaler_pre2025.pkl"
    meta_path = ARTIFACT_DIR / "meta_pre2025.json"

    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run train_pre2025.py first.")

    model = load_model(model_path)
    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)
    with meta_path.open("r") as f:
        meta = json.load(f)

    return model, scaler, meta


def load_data() -> pd.DataFrame:
    db = TradingDatabase(db_path=str(POST_DB))
    df = db.get_ohlcv_data("BTCUSDT", "15m")
    if df.empty:
        raise RuntimeError("Post-2025 database is empty.")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def prepare_features(df: pd.DataFrame, feature_engine: EnhancedFeatureEngine) -> pd.DataFrame:
    features = feature_engine.create_all_features(df, include_multiframe=True, min_periods=100)
    features = features.reset_index(drop=True)
    return features


def evaluate(model, scaler, meta, df: pd.DataFrame) -> dict[str, float]:
    feature_engine = EnhancedFeatureEngine(performance_mode=True)
    features = prepare_features(df, feature_engine)
    available_features = meta["available_features"]
    sequence_length = meta["sequence_length"]
    features = features.dropna(subset=available_features)

    future_periods = 20

    label_series = generate_stop_aligned_labels(features, future_periods=future_periods)
    mask = label_series != -1
    features = features.loc[mask]
    label_values = label_series.loc[mask].astype(int).values

    X_raw = features[available_features]
    X_scaled = scaler.transform(X_raw)

    sequences = []
    sequence_labels = []
    for i in range(sequence_length, len(X_scaled) + 1):
        sequences.append(X_scaled[i-sequence_length:i])
        sequence_labels.append(label_values[i-1])

    if not sequences:
        raise RuntimeError("No sequences available for evaluation.")

    X_eval = np.array(sequences)
    y_eval = np.array(sequence_labels).astype(int)

    prob_up = model.predict(X_eval, verbose=0).flatten()
    preds = (prob_up >= 0.5).astype(int)

    accuracy = (preds == y_eval).mean()
    precision = np.sum((preds == 1) & (y_eval == 1)) / max(np.sum(preds == 1), 1)
    recall = np.sum((preds == 1) & (y_eval == 1)) / max(np.sum(y_eval == 1), 1)

    return {
        "samples": int(len(y_eval)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
    }


def main() -> None:
    if not POST_DB.exists():
        raise FileNotFoundError(f"Post-2025 database not found: {POST_DB}")

    model, scaler, meta = load_artifacts()
    df = load_data()
    metrics = evaluate(model, scaler, meta, df)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Evaluation complete. Metrics saved to", RESULTS_PATH)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
