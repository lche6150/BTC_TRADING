#!/usr/bin/env python3
"""
Create Figure_1.png style comprehensive trading visualization
Matches the exact format of the original graph with 3 panels:
1. Entry/Exit points with P&L boxes
2. Trading signals & confidence levels
3. Cumulative performance
"""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
import sqlite3
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

from trading.confidence_trading_system import ComprehensiveTradingSystem, MarketData
from src.data.features import EnhancedFeatureEngine
from src.data.database import TradingDatabase
from scripts.gap_test_real_data import (
    load_optimized_config,
    create_gru_model,
    create_sequences,
)
from src.models.stop_aligned_labeler import generate_stop_aligned_labels
import pickle

DEFAULT_LOG_FILENAME = "trading_log.txt"
DEFAULT_FIGURE_FILENAME = "Figure_1.png"
BACKTEST_RESULTS_ROOT = Path("results/backtests")

MODEL_FEATURES = [
    'rsi_14', 'rsi_14_1h', 'rsi_14_4h',
    'bb_position', 'bb_position_1h', 'bb_position_4h',
    'macd_signal', 'macd_signal_1h',
    'volume_sma_ratio', 'volume_sma_ratio_1h',
    'price_position', 'hl_ratio', 'momentum_10', 'momentum_20',
    'volatility', 'volatility_1h', 'atr_normalized',
    'williams_r', 'stoch_k', 'cci', 'adx', 'roc'
]


def _purge_gru_backtest_artifacts() -> None:
    """Remove cached GRU artifacts so each backtest retrains from scratch."""
    artifacts_dir = Path('models/gru_backtest')
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)


def _to_timestamp_ms(date_str: str, inclusive_end: bool = False) -> int:
    """Convert a date string to a Binance-compatible millisecond timestamp."""
    dt = pd.to_datetime(date_str)
    if inclusive_end:
        dt = dt + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    return int(dt.timestamp() * 1000)


def load_backtest_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Load backtest data between start_date and end_date (inclusive)."""
    print("📊 Loading backtest data...")

    db_path = 'data/trading_data.db'
    conn = sqlite3.connect(db_path)

    start_ms = _to_timestamp_ms(start_date)
    end_ms = _to_timestamp_ms(end_date, inclusive_end=True)

    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM ohlcv_data
    WHERE timestamp >= ? AND timestamp <= ?
    ORDER BY timestamp
    """

    data = pd.read_sql_query(query, conn, params=(start_ms, end_ms))
    conn.close()

    if data.empty:
        raise ValueError(f"No OHLCV records found between {start_date} and {end_date}")

    # Convert timestamps and prepare features
    data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')
    data['returns'] = data['close'].pct_change()
    data['volume_ma'] = data['volume'].rolling(24).mean()
    data['volatility'] = data['returns'].rolling(24).std()
    data['price_change_1h'] = data['close'].pct_change(4)
    data['price_change_4h'] = data['close'].pct_change(16)

    # Simple trend
    data['sma_short'] = data['close'].rolling(12).mean()
    data['sma_long'] = data['close'].rolling(48).mean()
    data['trend_1h'] = np.where(
        data['sma_short'] > data['sma_long'], 'BULLISH',
        np.where(data['sma_short'] < data['sma_long'], 'BEARISH', 'NEUTRAL')
    )

    data = data.dropna().reset_index(drop=True)
    print(f"✅ Loaded {len(data)} clean data points")
    return data


def retrain_gru_model_until(cutoff_datetime: datetime,
                            symbol: str = "BTCUSDT",
                            interval: str = "15m") -> Dict[str, float]:
    """Retrain the GRU model using all data up to (and including) cutoff_datetime."""

    cutoff = pd.to_datetime(cutoff_datetime)

    artifacts_dir = Path('models/gru_backtest')
    model_path = artifacts_dir / 'model.keras'
    scaler_path = artifacts_dir / 'scaler.pkl'
    meta_path = artifacts_dir / 'meta.json'

    db = TradingDatabase()
    df = db.get_ohlcv_data(symbol, interval)
    if df.empty:
        raise ValueError("No OHLCV data available in database for retraining")

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df[df['datetime'] <= cutoff].copy()

    if len(df) < 2000:
        raise ValueError("Insufficient data available before cutoff for retraining")

    feature_engine = EnhancedFeatureEngine(performance_mode=True)
    features = feature_engine.create_all_features(df, include_multiframe=True, min_periods=100)

    available_features = [f for f in MODEL_FEATURES if f in features.columns]
    if not available_features:
        raise ValueError("No overlapping features found for GRU retraining")

    config = load_optimized_config()
    sequence_length = config['sequence_length']
    future_periods = 6

    labels = generate_stop_aligned_labels(features, future_periods=future_periods)
    targets = labels[labels != -1]

    X_raw = features.loc[targets.index, available_features]
    valid_mask = ~X_raw.isnull().any(axis=1)
    X_clean = X_raw[valid_mask]
    y_clean = targets.loc[valid_mask]

    if len(X_clean) <= sequence_length:
        raise ValueError("Not enough clean feature rows to build GRU sequences")

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_clean),
        columns=available_features,
        index=X_clean.index
    )

    X_seq, y_seq = create_sequences(X_scaled, y_clean, sequence_length)

    model = create_gru_model(config, X_seq.shape[2])

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=0,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.7,
            patience=max(1, config['patience'] // 2),
            min_lr=1e-6,
            verbose=0,
            mode='max'
        )
    ]

    model.fit(
        X_seq, y_seq,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    with scaler_path.open('wb') as f:
        pickle.dump(scaler, f)

    meta = {
        'available_features': available_features,
        'sequence_length': sequence_length,
        'best_threshold': 0.5,
    }
    with meta_path.open('w') as f:
        json.dump(meta, f, indent=2)

    return {
        'samples': len(X_seq),
        'available_features': len(available_features),
        'sequence_length': sequence_length,
    }


def _train_gru_and_predict(start_date: str, end_date: str) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Train (or load cached) GRU model and return probability predictions."""

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Remove any cached GRU artifacts so this backtest retrains from scratch.
    _purge_gru_backtest_artifacts()

    db = TradingDatabase()
    df = db.get_ohlcv_data("BTCUSDT", "15m")
    if df.empty:
        raise ValueError("No OHLCV data available in database")

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    train_mask = df['datetime'] < start_dt
    test_mask = (df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    if len(train_df) < 2000:
        raise ValueError("Insufficient training data before start date")
    if len(test_df) == 0:
        raise ValueError("No test data in specified window")

    feature_engine = EnhancedFeatureEngine(performance_mode=True)
    train_features = feature_engine.create_all_features(train_df, include_multiframe=True, min_periods=100)
    test_features = feature_engine.create_all_features(test_df, include_multiframe=True, min_periods=100)

    available_features = [f for f in MODEL_FEATURES if f in train_features.columns]
    if len(available_features) == 0:
        raise ValueError("No overlapping features found for GRU model")

    future_periods = 6

    retrained = True

    config = load_optimized_config()
    sequence_length = config['sequence_length']

    train_labels = generate_stop_aligned_labels(train_features, future_periods=future_periods)
    train_targets = train_labels[train_labels != -1]

    X_train_raw = train_features[available_features].loc[train_targets.index]
    train_valid_mask = ~X_train_raw.isnull().any(axis=1)
    X_train_clean = X_train_raw[train_valid_mask]
    y_train_clean = train_targets.loc[train_valid_mask]

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_clean),
        columns=available_features,
        index=X_train_clean.index
    )

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_clean, sequence_length)
    model = create_gru_model(config, X_train_seq.shape[2])

    total_sequences = len(X_train_seq)
    if total_sequences == 0:
        raise ValueError("Insufficient sequence data after cleaning to train GRU model")

    val_size = max(1, int(total_sequences * 0.2)) if total_sequences > 1 else 0
    train_size = total_sequences - val_size

    if train_size <= 0:
        train_size = total_sequences
        val_size = 0

    X_train_sequences = X_train_seq[:train_size]
    y_train_sequences = y_train_seq[:train_size]

    if val_size > 0:
        X_val_sequences = X_train_seq[train_size:]
        y_val_sequences = y_train_seq[train_size:]
        validation_data = (X_val_sequences, y_val_sequences)
    else:
        X_val_sequences = np.empty((0,))
        y_val_sequences = np.empty((0,))
        validation_data = None

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=0,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.7,
            patience=max(1, config['patience'] // 2),
            min_lr=1e-6,
            verbose=0,
            mode='max'
        )
    ]

    fit_kwargs = dict(
        x=X_train_sequences,
        y=y_train_sequences,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=0
    )
    if validation_data and len(X_val_sequences) > 0:
        fit_kwargs["validation_data"] = validation_data
    else:
        fit_kwargs["validation_split"] = 0.0

    model.fit(**fit_kwargs)

    # Determine best threshold using validation sequences
    best_threshold = 0.5
    if validation_data and len(X_val_sequences) > 0:
        val_probs = model.predict(X_val_sequences, verbose=0).flatten()
        thresholds = np.arange(0.4, 0.7, 0.05)
        best_acc = -1.0
        for thresh in thresholds:
            preds_temp = (val_probs > thresh).astype(int)
            acc_temp = (preds_temp == y_val_sequences).mean()
            if acc_temp > best_acc:
                best_acc = acc_temp
                best_threshold = float(thresh)

    # Evaluation targets for metrics
    test_labels = generate_stop_aligned_labels(test_features, future_periods=future_periods)
    test_targets = test_labels[test_labels != -1]
    X_test_eval_raw = test_features.loc[test_targets.index, available_features]
    test_eval_valid_mask = ~X_test_eval_raw.isnull().any(axis=1)
    X_test_eval_clean = X_test_eval_raw[test_eval_valid_mask]
    y_test_eval_clean = test_targets.loc[test_eval_valid_mask]
    X_test_eval_scaled = pd.DataFrame(
        scaler.transform(X_test_eval_clean),
        columns=available_features,
        index=X_test_eval_clean.index
    )
    X_test_eval_seq, y_test_eval_seq = create_sequences(X_test_eval_scaled, y_test_eval_clean, sequence_length)

    X_test_all_raw = test_features[available_features]
    test_all_valid_mask = ~X_test_all_raw.isnull().any(axis=1)
    X_test_all_clean = X_test_all_raw[test_all_valid_mask]
    X_test_all_scaled = pd.DataFrame(
        scaler.transform(X_test_all_clean),
        columns=available_features,
        index=X_test_all_clean.index
    )
    dummy_targets = pd.Series(np.zeros(len(X_test_all_scaled)), index=X_test_all_scaled.index)
    X_test_all_seq, _ = create_sequences(X_test_all_scaled, dummy_targets, sequence_length)

    prob_up_all = model.predict(X_test_all_seq, verbose=0).flatten()

    prediction_indices = X_test_all_scaled.index[sequence_length:]
    predictions_df = pd.DataFrame({
        'prob_up': prob_up_all,
    }, index=prediction_indices)
    predictions_df['prob_down'] = 1.0 - predictions_df['prob_up']
    predictions_df['confidence'] = predictions_df[['prob_up', 'prob_down']].max(axis=1)
    predictions_df['datetime'] = test_features.loc[prediction_indices, 'datetime'].values

    metrics: Dict[str, float] = {'retrained': retrained}
    if len(X_test_eval_seq) > 0:
        from sklearn.metrics import accuracy_score

        eval_indices = X_test_eval_scaled.index[sequence_length:]
        eval_probs = predictions_df.loc[eval_indices, 'prob_up']

        preds_eval = (eval_probs > best_threshold).astype(int)
        best_acc = accuracy_score(y_test_eval_seq, preds_eval)
        metrics['accuracy'] = float(best_acc)
        metrics['best_threshold'] = float(best_threshold)
        metrics['eval_samples'] = int(len(y_test_eval_seq))

        high_conf_mask = predictions_df.loc[eval_indices, 'confidence'] >= 0.55
        if high_conf_mask.any():
            high_conf_acc = accuracy_score(
                y_test_eval_seq[high_conf_mask.values],
                (eval_probs[high_conf_mask] > best_threshold).astype(int)
            )
            metrics['high_confidence_accuracy'] = float(high_conf_acc)
            metrics['high_confidence_samples'] = int(high_conf_mask.sum())
    else:
        metrics['best_threshold'] = float(best_threshold)

    return predictions_df, metrics

def generate_signals_with_tracking(
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    predictions: pd.DataFrame | None = None,
    cached_metrics: Dict[str, float] | None = None,
):
    """Generate trading signals using trained GRU model"""
    if predictions is None:
        print("🤖 训练 GRU 模型并生成信号...")
        predictions_df, model_metrics = _train_gru_and_predict(start_date, end_date)
    else:
        predictions_df = predictions.copy()
        model_metrics = cached_metrics or {}

    merged = data.merge(predictions_df[['datetime', 'prob_up', 'prob_down', 'confidence']],
                        on='datetime', how='left')

    predictions = []
    confidences = []
    signal_strength = []

    decision_threshold = model_metrics.get('best_threshold', 0.5)

    for i in range(len(merged)):
        prob_up = merged['prob_up'].iloc[i]
        if pd.isna(prob_up):
            predictions.append(None)
            confidences.append(0.0)
            signal_strength.append('None')
            continue

        prob_down = merged['prob_down'].iloc[i]
        confidence = merged['confidence'].iloc[i]

        prediction = 1 if prob_up >= decision_threshold else 0

        if confidence >= 0.80:
            strength = 'Strong'
        elif confidence >= 0.68:
            strength = 'Medium'
        else:
            strength = 'Weak'

        predictions.append(prediction)
        confidences.append(confidence)
        signal_strength.append(strength)

    merged['prediction'] = predictions
    merged['confidence'] = confidences
    merged['signal_strength'] = signal_strength
    merged['direction'] = merged['prediction'].apply(lambda p: 'UP' if p == 1 else ('DOWN' if p == 0 else None))

    return merged, model_metrics

def run_trading_simulation_with_tracking(data, system_config=None, initial_capital=20000.0):
    """Run trading simulation and track all trades"""
    print("🚀 Running trading simulation with detailed tracking...")

    trading_system = ComprehensiveTradingSystem(initial_capital, config=system_config)
    trades_log = []
    all_signals = []

    for i in range(100, len(data)):
        current_time = data['datetime'].iloc[i]
        current_price = data['close'].iloc[i]

        # Ensure we evaluate the trading window in UTC (dataset timestamps are UTC)
        if current_time.tzinfo is None:
            current_time_utc = current_time.tz_localize('UTC')
        else:
            current_time_utc = current_time.tz_convert('UTC')
        is_high_volume_hours = 8 <= current_time_utc.hour < 22

        # Create market data
        market_data = MarketData(
            current_price=current_price,
            volume_15m=data['volume'].iloc[i],
            avg_volume_24h=data['volume_ma'].iloc[i] if not pd.isna(data['volume_ma'].iloc[i]) else data['volume'].iloc[i],
            volatility_1h=data['volatility'].iloc[i] if not pd.isna(data['volatility'].iloc[i]) else 0.02,
            price_change_4h=data['price_change_4h'].iloc[i] if not pd.isna(data['price_change_4h'].iloc[i]) else 0.0,
            trend_1h=data['trend_1h'].iloc[i],
            is_high_volume_hours=is_high_volume_hours,
            recent_news_events=False
        )

        # Update existing positions
        updates = trading_system.update_positions(current_price, current_time)
        for update in updates:
            trades_log.append({
                'timestamp': current_time,
                'action': 'CLOSE',
                'reason': update['exit_reason'],
                'price': current_price,
                'pnl': update['realized_pnl'],
                'exit_fee': update.get('exit_fee', 0.0),
                'position_id': update['position_id']
            })

        # Record all signals (not just trades)
        prediction = data['prediction'].iloc[i]
        confidence = data['confidence'].iloc[i]
        direction = data['direction'].iloc[i]
        strength = data['signal_strength'].iloc[i]
        prob_up = data['prob_up'].iloc[i] if 'prob_up' in data.columns else None

        all_signals.append({
            'timestamp': current_time,
            'price': current_price,
            'direction': direction,
            'confidence': confidence,
            'strength': strength,
            'prob_up': prob_up,
            'executed': False
        })

        if direction is None or confidence < trading_system.min_confidence:
            continue

        position = trading_system.process_signal(
            direction,
            confidence,
            market_data,
            signal_time=pd.to_datetime(current_time).to_pydatetime(),
        )

        if position:
            # Mark this signal as executed
            all_signals[-1]['executed'] = True

            trades_log.append({
                'timestamp': current_time,
                'action': 'OPEN',
                'direction': direction,
                'price': current_price,
                'confidence': confidence,
                'prob_up': prob_up,
                'position_size_pct': position.position_size_pct,
                'position_value': position.position_value,
                'entry_fee': position.entry_fee,
                'position_id': position.signal_id
            })

    # Final position flush at session end
    final_time = data['datetime'].iloc[-1] + timedelta(minutes=15)
    final_price = data['close'].iloc[-1]

    final_updates = trading_system.update_positions(final_price, final_time)
    for update in final_updates:
        trades_log.append({
            'timestamp': final_time,
            'action': 'CLOSE',
            'reason': update['exit_reason'],
            'price': final_price,
            'pnl': update['realized_pnl'],
            'exit_fee': update.get('exit_fee', 0.0),
            'position_id': update['position_id']
        })

    # Force-close any remaining positions for end-of-test accounting
    for position in trading_system.active_positions.copy():
        update = trading_system._close_position(position, final_price, final_time, 'SESSION_END')
        trades_log.append({
            'timestamp': final_time,
            'action': 'CLOSE',
            'reason': update['exit_reason'],
            'price': final_price,
            'pnl': update['realized_pnl'],
            'exit_fee': update.get('exit_fee', 0.0),
            'position_id': update['position_id']
        })

    return trading_system, trades_log, all_signals

def create_trading_log(trade_pairs,
                       data: pd.DataFrame,
                       start_date: str,
                       end_date: str,
                       save_path='trading_log.txt'):
    """Create detailed trading log file with P&L information"""
    print("📝 Creating trading log file...")

    start_label = pd.to_datetime(start_date).strftime('%b %d, %Y')
    end_label = pd.to_datetime(end_date).strftime('%b %d, %Y')
    date_range_label = f"{start_label} - {end_label}"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    total_net_after_all = 0.0
    total_fees = 0.0
    winning_trades = 0
    daily_net = defaultdict(float)

    volume_ma_24 = data['volume'].rolling(window=24, min_periods=24).mean()
    high_volume_mask = data['volume'] >= (2.0 * volume_ma_24)
    high_volume_timestamps = set(data.loc[high_volume_mask, 'datetime'])
    high_volume_pnl = 0.0
    normal_volume_pnl = 0.0
    total_exit_pnl = 0.0

    with save_path.open('w') as f:
        f.write(f"TRADING LOG - {date_range_label} (Backtest)\n")
        f.write("=" * 50 + "\n\n")

        for trade_num, (pair_id, pair) in enumerate(trade_pairs.items(), 1):
            open_trade = pair.get('open')
            closes = pair.get('closes', [])
            if not open_trade or not closes:
                continue

            net_exit_pnl = sum(close.get('pnl', 0.0) for close in closes)
            entry_fee = open_trade.get('entry_fee', 0.0)
            exit_fees = sum(close.get('exit_fee', 0.0) for close in closes)
            fees = entry_fee + exit_fees
            net_after_all_fees = net_exit_pnl - entry_fee
            gross_pnl = net_exit_pnl + fees

            total_net_after_all += net_after_all_fees
            total_fees += fees
            total_exit_pnl += net_exit_pnl
            if net_after_all_fees > 0:
                winning_trades += 1

            trade_type = "LONG" if open_trade['direction'] == "UP" else "SHORT"

            f.write(f"Trade {trade_num} (E{trade_num} → X{trade_num}):\n")
            f.write(f"  Entry Time:  {open_trade['timestamp'].strftime('%m/%d %H:%M')}\n")
            f.write(f"  Entry Price: ${open_trade['price']:,.2f}\n")
            f.write(f"  Trade Type:  {trade_type}\n")
            f.write(f"  Direction:   {open_trade['direction']}\n")
            f.write(f"  Confidence:  {open_trade['confidence']:.1%}\n")
            position_pct = open_trade.get('position_size_pct', 0.0) * 100
            position_value = open_trade.get('position_value', 0.0)
            f.write(f"  Position:    {position_pct:.2f}% (${position_value:,.2f})\n")
            f.write(f"  Entry Fee:   ${entry_fee:,.2f}\n")

            last_bucket = None
            for idx, close_trade in enumerate(closes, 1):
                f.write(
                    f"  Exit {idx}:    {close_trade['timestamp'].strftime('%m/%d %H:%M')} | Reason: {close_trade['reason']} | "
                    f"P&L: ${close_trade.get('pnl', 0.0):+.2f} | Exit Fee: ${close_trade.get('exit_fee', 0.0):.2f}\n"
                )
                exit_ts = close_trade['timestamp']
                exit_pnl = close_trade.get('pnl', 0.0)
                if exit_ts in high_volume_timestamps:
                    high_volume_pnl += exit_pnl
                    last_bucket = 'high'
                else:
                    normal_volume_pnl += exit_pnl
                    last_bucket = 'normal'

            if entry_fee > 0:
                if last_bucket == 'high':
                    high_volume_pnl -= entry_fee
                else:
                    normal_volume_pnl -= entry_fee

            final_close_time = closes[-1]['timestamp']
            duration_hours = (final_close_time - open_trade['timestamp']).total_seconds() / 3600
            daily_net[final_close_time.date()] += net_after_all_fees
            f.write(f"  Gross P&L:   ${gross_pnl:+.2f}\n")
            f.write(f"  Fees Paid:   ${fees:.2f}\n")
            f.write(f"  Net P&L (exit fees only): ${net_exit_pnl:+.2f}\n")
            f.write(f"  Net P&L (all fees):       ${net_after_all_fees:+.2f}\n")
            f.write(f"  Duration:    {duration_hours:.1f}h\n")
            f.write("-" * 30 + "\n")

        trade_count = len([pair for pair in trade_pairs.values() if pair.get('open') and pair.get('closes')])
        win_rate = (winning_trades / trade_count) if trade_count else 0.0
        avg_pnl = (total_net_after_all / trade_count) if trade_count else 0.0

        f.write("\nSUMMARY:\n")
        f.write(f"Total Trades:    {trade_count}\n")
        f.write(f"Winning Trades:  {winning_trades}\n")
        f.write(f"Win Rate:        {win_rate:.1%}\n")
        f.write(f"Gross P&L:       ${total_net_after_all + total_fees:+.2f}\n")
        f.write(f"Total Fees:      ${total_fees:.2f}\n")
        f.write(f"Net P&L:         ${total_net_after_all:+.2f}\n")
        f.write(f"Avg Net/Trade:   ${avg_pnl:+.2f}\n")

        if daily_net:
            best_day = max(daily_net.items(), key=lambda x: x[1])
            worst_day = min(daily_net.items(), key=lambda x: x[1])
            f.write(f"Best Day:        {best_day[0]} (${best_day[1]:+.2f})\n")
            f.write(f"Worst Day:       {worst_day[0]} (${worst_day[1]:+.2f})\n")

        f.write("\nVOLUME SENSITIVITY:\n")
        f.write(f">2× 24-bar Avg Volume P&L: ${high_volume_pnl:+.2f}\n")
        f.write(f"Normal Volume P&L:         ${normal_volume_pnl:+.2f}\n")

    print(f"💾 Trading log saved to {save_path}")


def create_figure1_style_chart(data, trades_log, all_signals, date_range_label: str, save_path='Figure_1.png'):
    """Create comprehensive chart matching Figure_1.png style"""
    print("📈 Creating Figure_1.png style comprehensive chart...")

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 14))

    # Create gridspec for better control
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1.5], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])  # Top: Entry/Exit with P&L
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Middle: Signals & Confidence
    ax3 = fig.add_subplot(gs[2], sharex=ax1)  # Bottom: Cumulative P&L

    # --- PANEL 1: Entry/Exit Points with P&L ---
    ax1.plot(data['datetime'], data['close'], color='gray', linewidth=1, alpha=0.8, label='BTC Price')

    # Process trades for entry/exit boxes
    open_trades = [t for t in trades_log if t['action'] == 'OPEN']
    close_trades = [t for t in trades_log if t['action'] == 'CLOSE']

    # Create trade pairs for P&L boxes
    trade_pairs = {}
    for trade in open_trades:
        trade_pairs[trade['position_id']] = {'open': trade, 'close': None}

    for trade in close_trades:
        if trade['position_id'] in trade_pairs:
            trade_pairs[trade['position_id']]['close'] = trade

    # Get price range for label positioning
    price_min = data['close'].min()
    price_max = data['close'].max()
    price_range = price_max - price_min

    # Define label area below the price chart
    label_area_top = price_min - price_range * 0.05  # 5% below minimum price
    label_area_height = price_range * 0.15  # Use 15% of price range for labels

    # Draw P&L boxes and add trade numbers with vertical line mapping
    trade_counter = 1
    entry_labels = []  # Store entry label info for spacing
    exit_labels = []   # Store exit label info for spacing

    for pair_id, pair in trade_pairs.items():
        if pair['open'] and pair['close']:
            open_trade = pair['open']
            close_trade = pair['close']

            # Calculate box position and size
            x_start = open_trade['timestamp']
            x_end = close_trade['timestamp']
            y_bottom = min(open_trade['price'], close_trade['price']) - 50
            y_top = max(open_trade['price'], close_trade['price']) + 50

            # Color based on P&L
            pnl = close_trade['pnl']
            color = 'green' if pnl > 0 else 'red'
            alpha = 0.3

            # Draw rectangle
            width = (x_end - x_start).total_seconds() / 3600 / 24  # Convert to days for matplotlib
            rect = Rectangle((mdates.date2num(x_start), y_bottom),
                           mdates.date2num(x_end) - mdates.date2num(x_start),
                           y_top - y_bottom,
                           facecolor=color, alpha=alpha, edgecolor=color)
            ax1.add_patch(rect)

            # Store label information for later positioning
            entry_labels.append({
                'timestamp': x_start,
                'price': open_trade['price'],
                'label': f'E{trade_counter}',
                'counter': trade_counter
            })

            exit_labels.append({
                'timestamp': x_end,
                'price': close_trade['price'],
                'label': f'X{trade_counter}',
                'counter': trade_counter
            })

            trade_counter += 1

    # Sort labels by timestamp for proper horizontal spacing
    entry_labels.sort(key=lambda x: x['timestamp'])
    exit_labels.sort(key=lambda x: x['timestamp'])

    # Position entry labels in lower area with vertical lines
    for i, entry in enumerate(entry_labels):
        # Calculate horizontal position based on order
        label_y = label_area_top - (i % 3) * (label_area_height / 4)  # 3 rows of labels

        # Draw vertical line from actual price point to label
        ax1.plot([entry['timestamp'], entry['timestamp']],
                [entry['price'], label_y + 20],
                color='blue', linewidth=1, alpha=0.7, linestyle='-')

        # Add entry label
        ax1.text(entry['timestamp'], label_y, entry['label'],
                ha='center', va='center', fontsize=6,
                color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='blue', alpha=0.8))

    # Position exit labels in lower area with vertical lines
    for i, exit in enumerate(exit_labels):
        # Calculate horizontal position based on order, offset from entry labels
        label_y = label_area_top - label_area_height/2 - (i % 3) * (label_area_height / 4)  # Below entry labels

        # Draw vertical line from actual price point to label
        ax1.plot([exit['timestamp'], exit['timestamp']],
                [exit['price'], label_y + 20],
                color='red', linewidth=1, alpha=0.7, linestyle='-')

        # Add exit label
        ax1.text(exit['timestamp'], label_y, exit['label'],
                ha='center', va='center', fontsize=6,
                color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='red', alpha=0.8))

    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title('Trading System Backtest Results - Sept 25-29, 2025 (Unseen Data)\n$20,000 Initial Capital',
                  fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # --- PANEL 2: Trading Signals & Confidence ---
    # Plot BTC price on secondary y-axis
    ax2_price = ax2.twinx()
    ax2_price.plot(data['datetime'], data['close'], color='gray', linewidth=1, alpha=0.6, label='BTC Price')
    ax2_price.set_ylabel('BTC Price ($)', fontsize=12)
    ax2_price.tick_params(axis='y', labelcolor='gray')

    # Plot confidence levels and thresholds
    ax2.axhline(y=0.65, color='orange', linestyle='--', alpha=0.7, label='Min Confidence (65%)')
    ax2.axhline(y=0.60, color='purple', linestyle='--', alpha=0.7, label='Medium Threshold (60%)')
    ax2.axhline(y=0.80, color='pink', linestyle='--', alpha=0.7, label='High Threshold (80%)')

    # Get confidence range for label positioning
    confidence_min = 0.5
    confidence_max = 1.0
    confidence_range = confidence_max - confidence_min

    # Define label area below the confidence chart
    signal_label_area_top = confidence_min - confidence_range * 0.02  # 2% below minimum confidence (much closer)
    signal_label_area_height = confidence_range * 0.12  # Use 12% of confidence range for labels (smaller area)

    # Plot all signals with different styles based on strength and execution
    signal_df = pd.DataFrame(all_signals)

    # Filter signals for plotting (every 4th point to avoid overcrowding)
    plot_signals = signal_df[::4].copy()

    # Store signal labels for positioning below
    entry_signal_labels = []
    exit_signal_labels = []

    # Track executed trade numbers for signal panel
    executed_trade_counter = 1
    for _, signal in plot_signals.iterrows():
        confidence = signal['confidence']
        executed = signal['executed']
        direction = signal['direction']
        strength = signal['strength']

        # Determine color and size
        if executed:
            color = 'yellow' if direction == 'UP' else 'orange'
            size = 60
            marker = 'o'
        else:
            if strength == 'Strong':
                color = 'darkblue' if direction == 'UP' else 'darkred'
                alpha = 0.8
            elif strength == 'Medium':
                color = 'blue' if direction == 'UP' else 'red'
                alpha = 0.6
            else:  # Weak
                color = 'lightblue' if direction == 'UP' else 'lightcoral'
                alpha = 0.4
            size = 20
            marker = '^' if direction == 'UP' else 'v'

        ax2.scatter(signal['timestamp'], confidence,
                   c=color, s=size, marker=marker, alpha=0.7,
                   edgecolors='black' if executed else 'none', linewidth=0.5)

        # Store entry signal labels for later positioning
        if executed:
            entry_signal_labels.append({
                'timestamp': signal['timestamp'],
                'confidence': confidence,
                'label': f'E{executed_trade_counter}',
                'counter': executed_trade_counter
            })
            executed_trade_counter += 1

    # Add exit signals for profitable trades (not stop losses)
    exit_trade_counter = 1
    open_trades = [t for t in trades_log if t['action'] == 'OPEN']
    close_trades = [t for t in trades_log if t['action'] == 'CLOSE']

    # Create trade pairs to match entries with exits
    trade_pairs_signals = {}
    for trade in open_trades:
        trade_pairs_signals[trade['position_id']] = {'open': trade, 'close': None}
    for trade in close_trades:
        if trade['position_id'] in trade_pairs_signals:
            trade_pairs_signals[trade['position_id']]['close'] = trade

    # Plot exit signals only for profit target exits (not stop losses)
    for pair_id, pair in trade_pairs_signals.items():
        if pair['open'] and pair['close']:
            close_trade = pair['close']

            # Only show exit signal if it's a profit target (not stop loss)
            if 'PROFIT_TARGET' in close_trade['reason']:
                # Find the corresponding exit timestamp in the data
                exit_time = close_trade['timestamp']
                exit_price = close_trade['price']

                # Find the closest data point to the exit time for confidence level
                exit_data_idx = data[data['datetime'] <= exit_time].index[-1] if len(data[data['datetime'] <= exit_time]) > 0 else 0
                if exit_data_idx < len(data):
                    exit_confidence = data.iloc[exit_data_idx]['confidence']

                    # Plot exit signal
                    ax2.scatter(exit_time, exit_confidence,
                               c='green', s=80, marker='s', alpha=0.8,
                               edgecolors='darkgreen', linewidth=2)

                    # Store exit signal labels for later positioning
                    exit_signal_labels.append({
                        'timestamp': exit_time,
                        'confidence': exit_confidence,
                        'label': f'X{exit_trade_counter}',
                        'counter': exit_trade_counter
                    })

            exit_trade_counter += 1

    # Sort signal labels by timestamp for proper horizontal spacing
    entry_signal_labels.sort(key=lambda x: x['timestamp'])
    exit_signal_labels.sort(key=lambda x: x['timestamp'])

    # Position entry signal labels with smart spacing to avoid overlaps
    def get_smart_label_position(labels, base_y, min_spacing_hours=2):
        """Calculate positions to avoid horizontal overlaps"""
        positions = []
        if not labels:
            return positions

        # Sort by timestamp
        sorted_labels = sorted(labels, key=lambda x: x['timestamp'])

        for i, label in enumerate(sorted_labels):
            # Start with the base position
            y_pos = base_y - (i % 2) * (signal_label_area_height / 3)  # Alternate between 2 rows

            # Check for horizontal overlap with previous labels
            if positions:
                # Convert timestamp to hours for spacing calculation
                current_time = pd.to_datetime(label['timestamp'])
                for prev_label, prev_y in positions:
                    prev_time = pd.to_datetime(prev_label['timestamp'])
                    time_diff_hours = abs((current_time - prev_time).total_seconds() / 3600)

                    # If too close horizontally, use different row
                    if time_diff_hours < min_spacing_hours and abs(y_pos - prev_y) < signal_label_area_height / 6:
                        y_pos = base_y - signal_label_area_height * 0.6 - (i % 2) * (signal_label_area_height / 4)
                        break

            positions.append((label, y_pos))

        return positions

    # Position entry signal labels with smart spacing
    entry_positions = get_smart_label_position(entry_signal_labels, signal_label_area_top)

    for entry, label_y in entry_positions:
        # Draw simple vertical line from top to bottom of the chart
        ax2.axvline(x=entry['timestamp'], color='blue', linewidth=1, alpha=0.6, linestyle='-')

        # Add entry signal label
        ax2.text(entry['timestamp'], label_y, entry['label'],
                ha='center', va='center', fontsize=7,
                color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='blue', alpha=0.8))

    # Position exit signal labels with smart spacing
    exit_base_y = signal_label_area_top - signal_label_area_height * 0.4  # Offset from entry labels
    exit_positions = get_smart_label_position(exit_signal_labels, exit_base_y)

    for exit, label_y in exit_positions:
        # Draw simple vertical line from top to bottom of the chart
        ax2.axvline(x=exit['timestamp'], color='green', linewidth=1, alpha=0.6, linestyle='-')

        # Add exit signal label
        ax2.text(exit['timestamp'], label_y, exit['label'],
                ha='center', va='center', fontsize=7,
                color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='green', alpha=0.8))

    ax2.set_ylabel('Signal Confidence', fontsize=12, color='blue')
    ax2.set_ylim(0.5, 1.0)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_title('Trading Signals & Confidence Levels with BTC Price', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Create custom legend for signals
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10,
               markeredgecolor='black', label='Entry Signal UP'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10,
               markeredgecolor='black', label='Entry Signal DOWN'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=8,
               markeredgecolor='darkgreen', label='Exit Signal (Profit)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='darkblue', markersize=8,
               label='Strong UP'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='darkred', markersize=8,
               label='Strong DOWN'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='lightblue', markersize=6,
               label='Weak UP'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='lightcoral', markersize=6,
               label='Weak DOWN'),
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # --- PANEL 3: Cumulative Performance ---
    # Calculate cumulative P&L
    cumulative_pnl = []
    trade_times = []
    running_pnl = 0

    for trade in close_trades:
        trade_times.append(trade['timestamp'])
        running_pnl += trade['pnl']
        cumulative_pnl.append(running_pnl)

    if trade_times:
        ax3.plot(trade_times, cumulative_pnl, color='blue', linewidth=2, marker='o', markersize=4)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

        # Fill area
        ax3.fill_between(trade_times, cumulative_pnl, 0,
                        where=np.array(cumulative_pnl) >= 0, alpha=0.3, color='green')
        ax3.fill_between(trade_times, cumulative_pnl, 0,
                        where=np.array(cumulative_pnl) < 0, alpha=0.3, color='red')

        # Add final P&L and trade count text
        final_pnl = cumulative_pnl[-1]
        total_trades = len(close_trades)
        ax3.text(0.02, 0.98, f'Final P&L: ${final_pnl:.2f}\nTotal Trades: {total_trades}',
                transform=ax3.transAxes, fontsize=12, weight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax3.set_ylabel('Cumulative P&L ($)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Cumulative Trading Performance', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Format x-axis for shorter period (daily detail)
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Figure_1.png style chart saved to {save_path}")

    return fig

def main():
    """Generate Figure_1.png style visualization"""
    print("🎨 GENERATING FIGURE_1.PNG STYLE VISUALIZATION")
    print("=" * 60)

    parser = argparse.ArgumentParser(description='Generate Figure_1 style visualization')
    parser.add_argument('--start', dest='start_date', default='2025-09-25', help='Start date (UTC) for backtest window, e.g. 2025-08-01')
    parser.add_argument('--end', dest='end_date', default='2025-09-29', help='End date (UTC) for backtest window, e.g. 2025-09-30')
    parser.add_argument('--log', dest='log_path', default=DEFAULT_LOG_FILENAME, help='Path to save trading log output')
    parser.add_argument('--figure', dest='figure_path', default=DEFAULT_FIGURE_FILENAME, help='Path to save generated figure')
    parser.add_argument('--config', dest='config_path', help='Optional JSON file specifying trading-system parameters')
    args = parser.parse_args()

    # Organize outputs inside a dedicated backtest subdirectory when using defaults
    start_label = args.start_date.replace('-', '')
    end_label = args.end_date.replace('-', '')
    backtest_dir = BACKTEST_RESULTS_ROOT / f"{start_label}_to_{end_label}"

    log_path = Path(args.log_path)
    figure_path = Path(args.figure_path)

    if args.log_path == DEFAULT_LOG_FILENAME:
        backtest_dir.mkdir(parents=True, exist_ok=True)
        log_path = backtest_dir / DEFAULT_LOG_FILENAME
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.figure_path == DEFAULT_FIGURE_FILENAME:
        backtest_dir.mkdir(parents=True, exist_ok=True)
        figure_path = backtest_dir / DEFAULT_FIGURE_FILENAME
    else:
        figure_path.parent.mkdir(parents=True, exist_ok=True)

    args.log_path = log_path
    args.figure_path = figure_path

    success = False
    try:
        # Load data and generate signals
        data = load_backtest_data(args.start_date, args.end_date)
        data, model_metrics = generate_signals_with_tracking(data, args.start_date, args.end_date)

        # Run trading simulation
        system_config = None
        if args.config_path:
            config_file = Path(args.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            with config_file.open('r') as f:
                system_config = json.load(f)

        trading_system, trades_log, all_signals = run_trading_simulation_with_tracking(
            data,
            system_config=system_config,
        )

        # Create trading log file
        open_trades = [t for t in trades_log if t['action'] == 'OPEN']
        close_trades = [t for t in trades_log if t['action'] == 'CLOSE']

        trade_pairs = {}
        for trade in open_trades:
            trade_pairs[trade['position_id']] = {'open': trade, 'closes': []}
        for trade in close_trades:
            pair = trade_pairs.setdefault(trade['position_id'], {'open': None, 'closes': []})
            pair['closes'].append(trade)

        create_trading_log(trade_pairs, data, args.start_date, args.end_date, save_path=args.log_path)

        # Persist raw signal diagnostics (mirrors paper trader output)
        if all_signals:
            signal_log_path = Path(args.log_path).with_name('signal_log.jsonl')
            with signal_log_path.open('w') as signal_file:
                for signal in all_signals:
                    serializable = signal.copy()
                    ts = serializable.get('timestamp')
                    if isinstance(ts, pd.Timestamp):
                        serializable['timestamp'] = ts.isoformat()
                    signal_file.write(json.dumps(serializable, default=str) + '\n')
            print(f"📝 Signal log saved to {signal_log_path}")

        start_label = pd.to_datetime(args.start_date).strftime('%b %d, %Y')
        end_label = pd.to_datetime(args.end_date).strftime('%b %d, %Y')
        date_range_label = f"{start_label} - {end_label}"

        # Create Figure_1.png style chart (temporarily disabled)
        # fig = create_figure1_style_chart(data, trades_log, all_signals, date_range_label, save_path=args.figure_path)

        metrics = trading_system.get_performance_summary()

        if model_metrics:
            print("\n🤖 模型评估指标：")
            for key, value in model_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        print("\n📊 BACKTEST SUMMARY:")
        print(f"Total trades: {metrics['total_trades']}")
        print(f"Final P&L: ${metrics['realized_pnl']:.2f}")
        print(f"Win rate: {metrics['win_rate']*100:.1f}%")

        metrics_path = Path(args.figure_path).with_suffix('').parent / f"{Path(args.figure_path).stem}_metrics.json"
        with metrics_path.open('w') as f:
            json.dump(metrics, f, indent=2, default=float)
        print(f"📄 Performance metrics: {metrics_path}")

        print("\n✅ Figure_1.png style visualization completed!")
        success = True

    except Exception as e:
        print(f"❌ Error generating visualization: {e}")
        import traceback
        traceback.print_exc()
        success = False
    finally:
        _purge_gru_backtest_artifacts()

    return success

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Figure_1.png style chart generated successfully!")
    else:
        print("\n❌ Chart generation failed.")
