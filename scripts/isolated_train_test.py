#!/usr/bin/env python3
"""
Train the GRU model on an isolated database and evaluate it on a separate database.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
import shutil

import numpy as np
import pandas as pd

from scripts.gap_test_real_data import load_optimized_config, create_gru_model, create_sequences
from src.data.database import TradingDatabase
from src.data.features import EnhancedFeatureEngine
from trading_backtest import (
    MODEL_FEATURES,
    run_trading_simulation_with_tracking,
    create_trading_log,
    create_figure1_style_chart,
)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)
    engine = EnhancedFeatureEngine(performance_mode=True)
    return engine.create_all_features(df, include_multiframe=True, min_periods=100)


def train_gru(train_features: pd.DataFrame) -> Tuple[object, object, list[str], int, float]:
    available_features = [f for f in MODEL_FEATURES if f in train_features.columns]
    if not available_features:
        raise ValueError("No overlapping features in training data.")

    config = load_optimized_config()
    sequence_length = config["sequence_length"]
    future_periods = 6
    threshold = 0.001
    choices = [0, 1]

    future_returns = train_features["close"].pct_change(periods=future_periods).shift(-future_periods)
    labels = np.select(
        [future_returns <= -threshold, future_returns >= threshold],
        choices,
        default=-1,
    )
    mask = labels != -1
    targets = pd.Series(labels, index=train_features.index)[mask]

    X_raw = train_features.loc[targets.index, available_features]
    valid_mask = ~X_raw.isnull().any(axis=1)
    X_clean = X_raw[valid_mask]
    y_clean = targets.loc[valid_mask]

    if len(X_clean) <= sequence_length:
        raise ValueError("Insufficient training rows after cleaning.")

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_clean),
        columns=available_features,
        index=X_clean.index,
    )

    X_seq, y_seq = create_sequences(X_scaled, y_clean, sequence_length)
    model = create_gru_model(config, X_seq.shape[2])

    val_fraction = 0.2
    val_size = int(len(X_seq) * val_fraction)
    if 0 < val_size < len(X_seq):
        val_start = len(X_seq) - val_size
        X_train, y_train = X_seq[:val_start], y_seq[:val_start]
        X_val, y_val = X_seq[val_start:], y_seq[val_start:]
        validation_data = (X_val, y_val)
    else:
        X_train, y_train = X_seq, y_seq
        validation_data = None

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
        verbose=0,
        shuffle=False,
    )
    if validation_data:
        fit_kwargs["validation_data"] = validation_data

    model.fit(**fit_kwargs)

    best_threshold = 0.5
    if validation_data:
        val_probs = model.predict(validation_data[0], verbose=0).flatten()
        from sklearn.metrics import accuracy_score

        best_acc = -1.0
        for thresh in np.arange(0.4, 0.7, 0.05):
            preds = (val_probs > thresh).astype(int)
            acc = accuracy_score(validation_data[1], preds)
            if acc > best_acc:
                best_acc = acc
                best_threshold = float(thresh)

    return model, scaler, available_features, sequence_length, best_threshold


def predict_probabilities(
    model,
    scaler,
    features: pd.DataFrame,
    available_features: list[str],
    sequence_length: int,
):
    X_raw = features[available_features]
    valid_mask = ~X_raw.isnull().any(axis=1)
    X_clean = X_raw[valid_mask]
    X_scaled = pd.DataFrame(
        scaler.transform(X_clean),
        columns=available_features,
        index=X_clean.index,
    )
    dummy_targets = pd.Series(np.zeros(len(X_scaled)), index=X_scaled.index)
    X_seq, _ = create_sequences(X_scaled, dummy_targets, sequence_length)
    if len(X_seq) == 0:
        raise ValueError("Insufficient rows for prediction after cleaning.")
    prob_up = model.predict(X_seq, verbose=0).flatten()
    indices = X_scaled.index[sequence_length:]
    return prob_up, indices


def build_signal_dataframe(
    test_features: pd.DataFrame,
    raw_df: pd.DataFrame,
    prob_up: np.ndarray,
    indices: pd.Index,
    threshold: float,
) -> pd.DataFrame:
    merged = test_features.loc[indices].copy()
    merged["prob_up"] = prob_up
    merged["prob_down"] = 1.0 - prob_up
    merged["confidence"] = merged[["prob_up", "prob_down"]].max(axis=1)
    merged["prediction"] = (merged["prob_up"] >= threshold).astype(int)
    merged["direction"] = merged["prediction"].map({1: "UP", 0: "DOWN"})

    raw_lookup = raw_df.set_index("datetime")
    if "datetime" in merged.columns:
        raw_subset = raw_lookup.loc[merged["datetime"].values]
        for col in ["volume", "volume_ma", "volatility", "price_change_4h", "trend_1h", "close", "open"]:
            if col in raw_subset.columns and col not in merged.columns:
                merged[col] = raw_subset[col].values

    def strength(conf: float) -> str:
        if conf >= 0.80:
            return "Strong"
        if conf >= 0.68:
            return "Medium"
        return "Weak"

    merged["signal_strength"] = merged["confidence"].apply(strength)
    merged.reset_index(inplace=True, drop=True)
    return merged


def load_dataframe(db_path: Path, symbol: str, interval: str) -> pd.DataFrame:
    db = TradingDatabase(str(db_path))
    df = db.get_ohlcv_data(symbol, interval)
    if df.empty:
        raise ValueError(f"No rows found in {db_path}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["returns"] = df["close"].pct_change()
    df["volume_ma"] = df["volume"].rolling(24).mean()
    df["volatility"] = df["returns"].rolling(24).std()
    df["price_change_4h"] = df["close"].pct_change(16)
    df["sma_short"] = df["close"].rolling(12).mean()
    df["sma_long"] = df["close"].rolling(48).mean()
    df["trend_1h"] = np.where(
        df["sma_short"] > df["sma_long"], "BULLISH",
        np.where(df["sma_short"] < df["sma_long"], "BEARISH", "NEUTRAL"),
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on isolated DB and evaluate on another DB.")
    parser.add_argument("--train-db", type=Path, required=True, help="Path to isolated training database.")
    parser.add_argument("--test-db", type=Path, required=True, help="Path to isolated testing database.")
    parser.add_argument("--config", type=Path, default=Path("best_config.json"), help="Trading system configuration.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol.")
    parser.add_argument("--interval", default="15m", help="Interval type.")
    parser.add_argument("--output", type=Path, default=Path("results/isolated_evaluation"), help="Output directory.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("models/gru_isolated"),
                        help="Directory to persist trained GRU artifacts (model, scaler, meta).")
    parser.add_argument("--sync-to-signal", action="store_true",
                        help="Copy saved artifacts into models/gru_signal for paper trader usage.")
    parser.add_argument("--replay-feed", type=Path, default=None,
                        help="Optional path for exporting the engineered feature feed used for replay.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_dataframe(args.train_db, args.symbol, args.interval)
    test_raw = load_dataframe(args.test_db, args.symbol, args.interval)

    print(f"📊 Training rows: {len(train_raw):,}")
    print(f"📊 Testing rows:  {len(test_raw):,}")

    train_features = prepare_features(train_raw)
    model, scaler, available_features, sequence_length, threshold = train_gru(train_features)
    test_features = prepare_features(test_raw)

    missing = [f for f in available_features if f not in test_features.columns]
    if missing:
        raise ValueError(f"Test data missing required features: {missing}")

    # Persist artifacts for reuse by backtest/paper trader
    artifact_dir = args.artifact_dir
    model_path = artifact_dir / "model.keras"
    scaler_path = artifact_dir / "scaler.pkl"
    meta_path = artifact_dir / "meta.json"

    model.save(model_path)
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "available_features": available_features,
        "sequence_length": sequence_length,
        "best_threshold": float(threshold),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "train_database": str(args.train_db),
        "train_rows": len(train_raw),
        "train_start": train_raw["datetime"].iloc[0].isoformat(),
        "train_end": train_raw["datetime"].iloc[-1].isoformat(),
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"💾 Saved artifacts to {artifact_dir}")

    if args.sync_to_signal:
        signal_dir = Path("models/gru_signal")
        signal_dir.mkdir(parents=True, exist_ok=True)
        for name in ("model.keras", "scaler.pkl", "meta.json"):
            shutil.copy2(artifact_dir / name, signal_dir / name)
        print(f"🔄 Synced artifacts to {signal_dir}")

    # Merge essential market columns for downstream replay/paper trader comparisons
    market_cols = ["datetime", "close", "volume", "volume_ma", "volatility", "price_change_4h", "trend_1h"]
    available_market = [c for c in market_cols if c in test_raw.columns]
    if available_market:
        enriched_test_features = test_features.merge(
            test_raw[available_market],
            on="datetime",
            how="left",
            suffixes=("", "_raw"),
        )
    else:
        enriched_test_features = test_features.copy()

    replay_feed_path = args.replay_feed or (args.output / "replay_feed.parquet")
    enriched_test_features.sort_values("datetime").reset_index(drop=True).to_parquet(replay_feed_path, index=False)
    print(f"📦 Replay feed saved to {replay_feed_path}")

    prob_up, indices = predict_probabilities(model, scaler, test_features, available_features, sequence_length)
    signals_df = build_signal_dataframe(test_features, test_raw, prob_up, indices, threshold)

    with open(args.config) as f:
        system_config = json.load(f)

    trading_system, trades_log, all_signals = run_trading_simulation_with_tracking(signals_df, system_config=system_config)
    metrics: Dict[str, float] = trading_system.get_performance_summary()

    open_trades = [t for t in trades_log if t["action"] == "OPEN"]
    close_trades = [t for t in trades_log if t["action"] == "CLOSE"]
    trade_pairs = {}
    for trade in open_trades:
        trade_pairs[trade["position_id"]] = {"open": trade, "closes": []}
    for trade in close_trades:
        trade_pairs.setdefault(trade["position_id"], {"open": None, "closes": []})["closes"].append(trade)

    log_path = args.output / "trading_log.txt"
    start_label = signals_df["datetime"].iloc[0].strftime("%b %d, %Y")
    end_label = signals_df["datetime"].iloc[-1].strftime("%b %d, %Y")
    create_trading_log(trade_pairs, signals_df, start_label, end_label, save_path=log_path)

    figure_path = args.output / "Figure_1.png"
    create_figure1_style_chart(signals_df, trades_log, all_signals, f"{start_label} - {end_label}", save_path=figure_path)

    metrics_path = args.output / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=float)

    signal_log_path = args.output / "signal_log.jsonl"
    with signal_log_path.open("w") as f:
        for sig in all_signals:
            serialisable = sig.copy()
            serialisable["timestamp"] = pd.to_datetime(serialisable["timestamp"]).isoformat()
            for key, value in list(serialisable.items()):
                if isinstance(value, (np.floating, np.float32, np.float64)):
                    serialisable[key] = float(value)
                elif isinstance(value, (np.integer, np.int32, np.int64)):
                    serialisable[key] = int(value)
            f.write(json.dumps(serialisable) + "\n")

    print("✅ Evaluation complete")
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
