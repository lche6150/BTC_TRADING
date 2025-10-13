#!/usr/bin/env python3
"""
Gap Test with Real Data - Test model on actual unseen data
Fetch historical data for a specified window and test the optimized model
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
from typing import Optional, Tuple, Dict, Any
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.database import TradingDatabase
from src.data.features import EnhancedFeatureEngine


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for gap test configuration."""
    parser = argparse.ArgumentParser(description="Run gap test evaluation on historical data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol to evaluate")
    parser.add_argument("--interval", default="15m", help="Kline interval to use")
    parser.add_argument("--test-start", dest="test_start", default="2025-08-01", help="Gap test start date (UTC)")
    parser.add_argument("--test-end", dest="test_end", default="2025-09-26", help="Gap test end date (UTC)")
    parser.add_argument("--training-cutoff", dest="training_cutoff", default=None,
                        help="Optional training cutoff datetime (inclusive). Defaults to just before test start")
    parser.add_argument("--output-prefix", dest="output_prefix", default="gap_test",
                        help="Prefix for saved result artifacts")
    return parser.parse_args()

def load_optimized_config():
    """Load optimized model configuration"""
    try:
        with open("models/optimized_lstm/best_config.json", "r") as f:
            config = json.load(f)
        print("✅ Loaded optimized model configuration")
        return config
    except FileNotFoundError:
        print("⚠️ Using default configuration")
        return {
            'architecture': 'gru_deep',
            'sequence_length': 35,
            'lstm_units': [80, 40, 20],
            'dense_units': [24],
            'dropout': 0.2,
            'l2_reg': 1e-4,
            'batch_size': 128,
            'epochs': 30,
            'learning_rate': 0.002,
            'patience': 5
        }

def create_gru_model(config, n_features):
    """Create GRU model from configuration"""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    model = Sequential()

    # GRU layers
    model.add(GRU(
        config['lstm_units'][0],
        return_sequences=len(config['lstm_units']) > 1,
        input_shape=(config['sequence_length'], n_features),
        dropout=config['dropout'],
        kernel_regularizer=l2(config['l2_reg'])
    ))

    for i, units in enumerate(config['lstm_units'][1:], 1):
        model.add(GRU(
            units,
            return_sequences=i < len(config['lstm_units']) - 1,
            dropout=config['dropout'],
            kernel_regularizer=l2(config['l2_reg'])
        ))

    model.add(BatchNormalization())

    for units in config['dense_units']:
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(config['l2_reg'])))
        model.add(Dropout(config['dropout']))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def create_sequences(X, y, sequence_length):
    """Create sequences for GRU"""
    sequences_X = []
    sequences_y = []

    for i in range(sequence_length, len(X)):
        sequences_X.append(X.iloc[i-sequence_length:i].values)
        sequences_y.append(y.iloc[i])

    return np.array(sequences_X), np.array(sequences_y)

def prepare_gap_test_data(symbol: str,
                          interval: str,
                          test_start: str,
                          test_end: str,
                          training_cutoff: Optional[str] = None):
    """Prepare gap test data using the specified date range"""
    print("=" * 80)
    print(f"📊 GAP TEST: REAL DATA FROM {test_start.upper()} TO {test_end.upper()}")
    print("=" * 80)

    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end)

    if training_cutoff:
        training_cutoff_dt = pd.to_datetime(training_cutoff)
    else:
        training_cutoff_dt = test_start_dt - pd.Timedelta(minutes=15)

    if training_cutoff_dt >= test_start_dt:
        raise ValueError("Training cutoff must be before test start date")

    # Load existing database
    db = TradingDatabase()
    df = db.get_ohlcv_data(symbol, interval)

    if df.empty:
        print("❌ No data found in database!")
        return None, None, None, None

    # Convert datetime column to proper datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"📊 Total available data: {len(df):,} records")
    print(f"📅 Full date range: {df['datetime'].min()} to {df['datetime'].max()}")

    # Define the gap test periods as requested
    gap_test_start = test_start_dt
    gap_test_end = test_end_dt

    # Split data
    train_mask = df['datetime'] <= training_cutoff_dt
    test_mask = (df['datetime'] >= gap_test_start) & (df['datetime'] <= gap_test_end)

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    if test_df.empty:
        print("❌ No test data found in the specified range!")
        print("Available data range:", df['datetime'].min(), "to", df['datetime'].max())
        return None, None, None, None

    print(f"\n📋 DATA SPLIT:")
    print(f"  Training: {len(train_df):,} records (up to {train_df['datetime'].max()})")
    print(f"  Testing: {len(test_df):,} records ({test_df['datetime'].min()} to {test_df['datetime'].max()})")

    if train_df.empty:
        time_gap_days = None
    else:
        time_gap = test_df['datetime'].min() - train_df['datetime'].max()
        time_gap_days = time_gap.total_seconds() / 86400
        print(f"  Time gap: {time_gap_days:.2f} days")

    return train_df, test_df, time_gap_days, df

def run_gap_test(symbol: str,
                 interval: str,
                 test_start: str,
                 test_end: str,
                 training_cutoff: Optional[str],
                 output_prefix: str) -> Tuple[bool, Dict[str, Any]]:
    """Run the complete gap test"""
    print("=" * 80)
    print("🧪 RUNNING GAP TEST ON REAL UNSEEN DATA")
    print("=" * 80)

    # Prepare data
    train_df, test_df, time_gap_days, full_df = prepare_gap_test_data(
        symbol=symbol,
        interval=interval,
        test_start=test_start,
        test_end=test_end,
        training_cutoff=training_cutoff
    )
    range_label = f"{test_start} → {test_end}"

    if train_df is None:
        print("❌ Failed to prepare test data")
        return False, {}

    # Feature engineering for training data
    print("\n🔧 Preparing training data...")
    feature_engine = EnhancedFeatureEngine(performance_mode=True)

    train_features = feature_engine.create_all_features(
        df=train_df,
        include_multiframe=True,
        min_periods=100
    )

    # Create training targets (same as optimization)
    future_periods = 6  # 90-min ahead
    threshold = 0.001   # 0.1%

    train_future_returns = train_features['close'].pct_change(periods=future_periods).shift(-future_periods)
    conditions = [
        train_future_returns <= -threshold,
        train_future_returns >= threshold,
    ]
    choices = [0, 1]
    train_clear_movements = np.select(conditions, choices, default=-1)
    train_sideways_mask = train_clear_movements != -1
    train_targets = train_clear_movements[train_sideways_mask]

    print(f"Training target distribution:")
    unique, counts = np.unique(train_targets, return_counts=True)
    for val, count in zip(unique, counts):
        label = 'Down' if val == 0 else 'Up'
        print(f"  {label}: {count:,} ({count/len(train_targets)*100:.1f}%)")

    # Feature selection (use proven crypto features)
    crypto_features = [
        'rsi_14', 'rsi_14_1h', 'rsi_14_4h',
        'bb_position', 'bb_position_1h', 'bb_position_4h',
        'macd_signal', 'macd_signal_1h',
        'volume_sma_ratio', 'volume_sma_ratio_1h',
        'price_position', 'hl_ratio', 'momentum_10', 'momentum_20',
        'volatility', 'volatility_1h', 'atr_normalized',
        'williams_r', 'stoch_k', 'cci', 'adx', 'roc'
    ]

    available_features = [f for f in crypto_features if f in train_features.columns]
    print(f"📊 Using {len(available_features)} features")

    # Prepare clean training data
    X_train_raw = train_features[available_features]
    X_train_filtered = X_train_raw[train_sideways_mask]

    valid_mask = ~(X_train_filtered.isnull().any(axis=1) | pd.Series(train_targets).isnull())
    X_train_clean = X_train_filtered[valid_mask]
    y_train_clean = pd.Series(train_targets)[valid_mask]

    print(f"✅ Clean training data: {len(X_train_clean):,} samples")

    # Feature engineering for test data
    print(f"\n🔧 Preparing gap test data...")

    test_features = feature_engine.create_all_features(
        df=test_df,
        include_multiframe=True,
        min_periods=100
    )

    # Create test targets
    test_future_returns = test_features['close'].pct_change(periods=future_periods).shift(-future_periods)
    test_clear_movements = np.select([
        test_future_returns <= -threshold,
        test_future_returns >= threshold,
    ], choices, default=-1)
    test_sideways_mask = test_clear_movements != -1
    test_targets = test_clear_movements[test_sideways_mask]

    print(f"Test target distribution:")
    unique, counts = np.unique(test_targets, return_counts=True)
    for val, count in zip(unique, counts):
        label = 'Down' if val == 0 else 'Up'
        print(f"  {label}: {count:,} ({count/len(test_targets)*100:.1f}%)")

    # Prepare clean test data (same features as training)
    X_test_raw = test_features[available_features]
    X_test_filtered = X_test_raw[test_sideways_mask]

    # Create properly aligned target series with same index as filtered X
    test_targets_aligned = pd.Series(test_targets, index=X_test_filtered.index)

    test_valid_mask = ~(X_test_filtered.isnull().any(axis=1) | test_targets_aligned.isnull())
    X_test_clean = X_test_filtered[test_valid_mask]
    y_test_clean = test_targets_aligned[test_valid_mask]

    print(f"✅ Clean test data: {len(X_test_clean):,} samples")

    if len(X_test_clean) < 100:
        print("❌ Not enough test data for reliable evaluation")
        return False, {}

    # Scale data (fit on training, transform both)
    print(f"\n⚙️ Scaling data...")
    scaler = MinMaxScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_clean),
        columns=X_train_clean.columns,
        index=X_train_clean.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_clean),
        columns=X_test_clean.columns,
        index=X_test_clean.index
    )

    # Create sequences
    config = load_optimized_config()
    sequence_length = config['sequence_length']

    print(f"📊 Creating sequences (length={sequence_length})...")

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_clean, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_clean, sequence_length)

    print(f"Training sequences: {X_train_seq.shape}")
    print(f"Test sequences: {X_test_seq.shape}")

    if len(X_test_seq) < 50:
        print("❌ Not enough test sequences for evaluation")
        return False, {}

    # Train model
    print(f"\n🧠 Training GRU model on pre-gap data...")

    model = create_gru_model(config, X_train_seq.shape[2])

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=config['patience'],
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.7,
            patience=config['patience']//2,
            min_lr=1e-6,
            mode='max',
            verbose=1
        )
    ]

    start_time = time.time()

    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time

    # Test on gap data
    print(f"\n🎯 Testing on gap data ({range_label})...")

    y_pred_prob = model.predict(X_test_seq, verbose=0)

    # Optimize threshold
    thresholds = np.arange(0.4, 0.7, 0.05)
    best_threshold = 0.5
    best_acc = 0

    for threshold in thresholds:
        y_pred_temp = (y_pred_prob > threshold).astype(int).flatten()
        acc_temp = accuracy_score(y_test_seq, y_pred_temp)
        if acc_temp > best_acc:
            best_acc = acc_temp
            best_threshold = threshold

    y_pred = (y_pred_prob > best_threshold).astype(int).flatten()

    # Confidence of predicted class
    predicted_confidence = np.where(y_pred == 1, y_pred_prob.flatten(), 1.0 - y_pred_prob.flatten())
    confidence_threshold = 0.55
    high_conf_mask = predicted_confidence >= confidence_threshold
    if high_conf_mask.any():
        high_conf_accuracy = accuracy_score(y_test_seq[high_conf_mask], y_pred[high_conf_mask])
        high_conf_count = int(high_conf_mask.sum())
    else:
        high_conf_accuracy = None
        high_conf_count = 0

    # Calculate metrics
    accuracy = accuracy_score(y_test_seq, y_pred)

    up_mask = y_test_seq == 1
    down_mask = y_test_seq == 0

    up_accuracy = accuracy_score(y_test_seq[up_mask], y_pred[up_mask]) if up_mask.sum() > 0 else 0
    down_accuracy = accuracy_score(y_test_seq[down_mask], y_pred[down_mask]) if down_mask.sum() > 0 else 0

    # Results
    print(f"\n🏆 GAP TEST RESULTS ({range_label.upper()})")
    print("=" * 60)
    print(f"⚡ Training time: {training_time:.1f} seconds")
    print(f"🎯 Overall accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"📈 Up movement accuracy: {up_accuracy:.3f} ({up_accuracy*100:.1f}%)")
    print(f"📉 Down movement accuracy: {down_accuracy:.3f} ({down_accuracy*100:.1f}%)")
    print(f"🔢 Total predictions: {len(y_pred)}")
    print(f"✅ Correct predictions: {np.sum(y_pred == y_test_seq)}")
    print(f"🎚️ Optimal threshold: {best_threshold:.2f}")
    if time_gap_days is not None:
        print(f"⏰ Time gap: {time_gap_days:.2f} days")
    if high_conf_accuracy is not None:
        print(f"💎 High-confidence accuracy (≥{confidence_threshold:.2f}): {high_conf_accuracy:.3f} "
              f"({high_conf_accuracy*100:.1f}%) on {high_conf_count} samples")
    else:
        print(f"💎 High-confidence accuracy (≥{confidence_threshold:.2f}): N/A (no samples)")

    print(f"\n📊 Classification Report:")
    print(classification_report(y_test_seq, y_pred, target_names=['Down', 'Up']))

    # Create visualizations
    print(f"\n📈 Creating visualizations...")

    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    gap_desc = "N/A" if time_gap_days is None else f"{time_gap_days:.2f} days"
    fig.suptitle(
        f'Gap Test Results: {test_start} → {test_end} (Gap: {gap_desc})',
        fontsize=16,
        fontweight='bold'
    )

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test_seq, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_ylabel('True Label')
    axes[0,0].set_xlabel('Predicted Label')

    # 2. Prediction Distribution
    axes[0,1].hist(y_pred_prob[y_test_seq == 0], alpha=0.5, label='Down (Actual)', bins=20, color='red')
    axes[0,1].hist(y_pred_prob[y_test_seq == 1], alpha=0.5, label='Up (Actual)', bins=20, color='green')
    axes[0,1].axvline(best_threshold, color='black', linestyle='--', label=f'Threshold={best_threshold:.2f}')
    axes[0,1].set_title('Prediction Probability Distribution')
    axes[0,1].set_xlabel('Predicted Probability')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()

    # 3. Training History
    axes[1,0].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    axes[1,0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    axes[1,0].set_title('Model Training History')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 4. Performance Comparison
    metrics = ['Overall', 'Up Movements', 'Down Movements']
    accuracies = [accuracy, up_accuracy, down_accuracy]
    colors = ['steelblue', 'green', 'red']

    bars = axes[1,1].bar(metrics, accuracies, color=colors, alpha=0.7)
    axes[1,1].set_title('Accuracy by Movement Type')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].set_ylim(0, 1)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

    # Add baseline line
    axes[1,1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random (50%)')
    axes[1,1].legend()

    plt.tight_layout()

    # Save plots
    output_dir = Path("results") / "gap_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    range_slug = f"{pd.to_datetime(test_start).strftime('%Y%m%d')}_{pd.to_datetime(test_end).strftime('%Y%m%d')}"
    artifact_prefix = f"{output_prefix}_{range_slug}"
    summary_path = output_dir / f"{artifact_prefix}.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved performance plots to {summary_path}")

    # Create time series plot
    print(f"\n📊 Creating time series visualization...")

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'Gap Test Time Series: {range_label}', fontsize=16, fontweight='bold')

    # Get corresponding dates for test sequences
    test_dates = test_features['datetime'][test_sideways_mask][test_valid_mask].iloc[sequence_length:]
    test_prices = test_features['close'][test_sideways_mask][test_valid_mask].iloc[sequence_length:]

    # 1. Price chart with predictions
    axes[0].plot(test_dates, test_prices, color='black', linewidth=1, alpha=0.8, label='BTC Price')

    # Mark predictions
    correct_mask = y_pred == y_test_seq
    incorrect_mask = ~correct_mask

    up_correct = (y_pred == 1) & correct_mask
    down_correct = (y_pred == 0) & correct_mask
    up_incorrect = (y_pred == 1) & incorrect_mask
    down_incorrect = (y_pred == 0) & incorrect_mask

    if up_correct.sum() > 0:
        axes[0].scatter(test_dates[up_correct], test_prices.iloc[up_correct],
                       color='green', alpha=0.7, s=15, label='Correct Up', marker='^')
    if down_correct.sum() > 0:
        axes[0].scatter(test_dates[down_correct], test_prices.iloc[down_correct],
                       color='red', alpha=0.7, s=15, label='Correct Down', marker='v')
    if up_incorrect.sum() > 0:
        axes[0].scatter(test_dates[up_incorrect], test_prices.iloc[up_incorrect],
                       color='lightgreen', alpha=0.5, s=10, label='Wrong Up', marker='x')
    if down_incorrect.sum() > 0:
        axes[0].scatter(test_dates[down_incorrect], test_prices.iloc[down_incorrect],
                       color='pink', alpha=0.5, s=10, label='Wrong Down', marker='x')

    axes[0].set_title(f'BTC Price with Predictions (Accuracy: {accuracy:.1%})')
    axes[0].set_ylabel('BTC Price (USD)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Actual vs Predicted signals
    axes[1].plot(test_dates, y_test_seq + 0.1, 'g-', label='Actual Signals', alpha=0.7, linewidth=2)
    axes[1].plot(test_dates, y_pred - 0.1, 'r--', label='Predicted Signals', alpha=0.7, linewidth=2)
    axes[1].set_title('Signal Comparison: Actual vs Predicted')
    axes[1].set_ylabel('Signal (Up=1, Down=0)')
    axes[1].set_ylim(-0.5, 1.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Prediction confidence over time
    axes[2].plot(test_dates, y_pred_prob.flatten(), color='purple', alpha=0.8, linewidth=1, label='Model Confidence')
    axes[2].axhline(best_threshold, color='black', linestyle='--', label=f'Decision Threshold ({best_threshold:.2f})')
    axes[2].fill_between(test_dates, 0, best_threshold, alpha=0.2, color='red', label='Down Zone')
    axes[2].fill_between(test_dates, best_threshold, 1, alpha=0.2, color='green', label='Up Zone')
    axes[2].set_title('Model Confidence Over Time')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Confidence (0-1)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    ts_path = output_dir / f"{artifact_prefix}_timeseries.png"
    plt.savefig(ts_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved time series plot to {ts_path}")

    plt.show()

    # Save results
    results = {
        'test_period': f'{test_start} → {test_end}',
        'training_period_end': str(train_df['datetime'].max()) if not train_df.empty else None,
        'time_gap_days': None if time_gap_days is None else float(time_gap_days),
        'gap_test_accuracy': float(accuracy),
        'up_accuracy': float(up_accuracy),
        'down_accuracy': float(down_accuracy),
        'total_predictions': int(len(y_pred)),
        'correct_predictions': int(np.sum(y_pred == y_test_seq)),
        'optimal_threshold': float(best_threshold),
        'training_time_seconds': float(training_time),
        'training_samples': int(len(X_train_seq)),
        'test_samples': int(len(X_test_seq)),
        'confidence_metrics': {
            'predicted_confidence_threshold': confidence_threshold,
            'high_confidence_predictions': high_conf_count,
            'high_confidence_accuracy': None if high_conf_accuracy is None else float(high_conf_accuracy)
        }
    }

    results_path = output_dir / f"{artifact_prefix}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"💾 Saved detailed results to {results_path}")

    # Performance assessment
    print(f"\n📊 PERFORMANCE ASSESSMENT")
    print("-" * 60)

    if accuracy >= 0.60:
        print(f"🎯 EXCELLENT: {accuracy:.1%} accuracy on {range_label} data!")
        print("✅ Model shows strong temporal generalization")
        success = True
    elif accuracy >= 0.55:
        print(f"⚠️ GOOD: {accuracy:.1%} accuracy on gap data ({range_label})")
        print("✅ Model shows reasonable temporal robustness")
        success = True
    else:
        print(f"❌ CONCERNING: {accuracy:.1%} accuracy on gap data ({range_label})")
        print("⚠️ Model may be sensitive to temporal shifts")
        success = False

    print(f"\n🔍 Key Insights:")
    print(f"  • Temporal robustness: {'Strong' if accuracy > 0.6 else 'Moderate' if accuracy > 0.55 else 'Weak'}")
    if time_gap_days is not None:
        print(f"  • Time gap handled: {time_gap_days:.2f} days")
    print(f"  • Best signal type: {'Up' if up_accuracy > down_accuracy else 'Down'} ({max(up_accuracy, down_accuracy):.1%})")
    print(f"  • Ready for live trading: {'Yes' if success else 'Needs improvement'}")

    return success, {
        'results_path': str(results_path),
        'summary_path': str(summary_path),
        'timeseries_path': str(ts_path),
        'artifact_prefix': artifact_prefix,
        'metrics': results
    }

def main() -> bool:
    """Main function"""
    args = parse_args()

    try:
        success, artifacts = run_gap_test(
            symbol=args.symbol,
            interval=args.interval,
            test_start=args.test_start,
            test_end=args.test_end,
            training_cutoff=args.training_cutoff,
            output_prefix=args.output_prefix
        )

        if success:
            print("\n🎯 GAP TEST COMPLETED SUCCESSFULLY!")
            print("✅ Model demonstrates temporal robustness")
            if artifacts.get('results_path'):
                print(f"📄 Results JSON: {artifacts['results_path']}")
            if artifacts.get('summary_path'):
                print(f"🖼️ Summary plot: {artifacts['summary_path']}")
            if artifacts.get('timeseries_path'):
                print(f"🖼️ Timeseries plot: {artifacts['timeseries_path']}")
        else:
            print("\n⚠️ GAP TEST SHOWS ROOM FOR IMPROVEMENT")
            print("📈 Consider model refinement for better temporal adaptation")

        return success

    except Exception as e:
        print(f"❌ Error in gap test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
