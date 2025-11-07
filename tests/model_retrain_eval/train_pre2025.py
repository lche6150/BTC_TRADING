#!/usr/bin/env python3
"""
Train the GRU model using data prior to 2025-01-01 and persist artifacts.

Usage:
    PYTHONPATH=. python tests/model_retrain_eval/train_pre2025.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from scripts.gap_test_real_data import create_gru_model, create_sequences, load_optimized_config
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

PRE_DB = Path("data/trading_data_pre2025.db")
ARTIFACT_DIR = Path("tests/model_retrain_eval/artifacts")


def load_data() -> pd.DataFrame:
    db = TradingDatabase(db_path=str(PRE_DB))
    df = db.get_ohlcv_data("BTCUSDT", "15m")
    if df.empty:
        raise RuntimeError("Pre-2025 database is empty.")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def create_training_sets(df: pd.DataFrame):
    feature_engine = EnhancedFeatureEngine(performance_mode=True)
    features = feature_engine.create_all_features(df, include_multiframe=True, min_periods=100)
    features = features.reset_index(drop=True)
    available_features = [f for f in MODEL_FEATURES if f in features.columns]
    if not available_features:
        raise RuntimeError("No overlapping features for training.")

    config = load_optimized_config()
    sequence_length = config["sequence_length"]
    future_periods = 20

    labels = generate_stop_aligned_labels(features, future_periods=future_periods)
    targets = labels[labels != -1]

    X_raw = features.loc[targets.index, available_features]
    valid_mask = ~X_raw.isnull().any(axis=1)
    X_clean = X_raw[valid_mask]
    y_clean = targets.loc[valid_mask]

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_clean),
        columns=available_features,
        index=X_clean.index,
    )

    X_seq, y_seq = create_sequences(X_scaled, y_clean, sequence_length)
    if len(X_seq) == 0:
        raise RuntimeError("Not enough sequences to train the model.")

    return X_seq, y_seq, scaler, available_features, sequence_length


def train_model(X_seq, y_seq, sequence_length, feature_count):
    config = load_optimized_config()
    model = create_gru_model(config, feature_count)

    total_sequences = len(X_seq)
    val_size = max(1, int(total_sequences * 0.2)) if total_sequences > 1 else 0
    train_size = total_sequences - val_size
    if train_size <= 0:
        train_size = total_sequences
        val_size = 0

    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    validation_data = None
    if val_size > 0:
        X_val = X_seq[train_size:]
        y_val = y_seq[train_size:]
        validation_data = (X_val, y_val)

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=config["patience"],
            restore_best_weights=True,
            verbose=0,
            mode="max",
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.7,
            patience=max(1, config["patience"] // 2),
            min_lr=1e-6,
            verbose=0,
            mode="max",
        ),
    ]

    fit_kwargs = dict(
        x=X_train,
        y=y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )
    if validation_data:
        fit_kwargs["validation_data"] = validation_data
    else:
        fit_kwargs["validation_split"] = 0.0

    model.fit(**fit_kwargs)
    return model


def save_artifacts(model, scaler, features, sequence_length, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "gru_pre2025.keras"
    scaler_path = output_dir / "scaler_pre2025.pkl"
    meta_path = output_dir / "meta_pre2025.json"

    model.save(model_path)
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)
    meta = {
        "available_features": features,
        "sequence_length": sequence_length,
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"Artifacts saved to {output_dir}")


def main() -> None:
    if not PRE_DB.exists():
        raise FileNotFoundError(f"Pre-2025 database not found: {PRE_DB}")

    df = load_data()
    X_seq, y_seq, scaler, features, sequence_length = create_training_sets(df)
    model = train_model(X_seq, y_seq, sequence_length, len(features))
    save_artifacts(model, scaler, features, sequence_length, ARTIFACT_DIR)
    print("✅ Training complete.")


if __name__ == "__main__":
    main()
