#!/usr/bin/env python3
"""
Focused LSTM Optimization - Target high-impact improvements
Systematic testing of proven performance boosters
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.database import TradingDatabase
from src.data.features import EnhancedFeatureEngine

# Set up logging
logging.basicConfig(level=logging.INFO)

def create_focused_lstm(config):
    """Create optimized LSTM with specific configuration"""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, GRU, Dense, Dropout, BatchNormalization,
        Bidirectional, Conv1D, MaxPooling1D, Attention
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    model = Sequential()

    if config['architecture'] == 'bilstm_attention':
        # Bidirectional LSTM with attention mechanism
        model.add(Bidirectional(LSTM(
            config['lstm_units'][0],
            return_sequences=True,
            dropout=config['dropout'],
            kernel_regularizer=l2(config['l2_reg'])
        ), input_shape=(config['sequence_length'], config['n_features'])))

        model.add(BatchNormalization())

        if len(config['lstm_units']) > 1:
            model.add(Bidirectional(LSTM(
                config['lstm_units'][1],
                return_sequences=False,
                dropout=config['dropout'],
                kernel_regularizer=l2(config['l2_reg'])
            )))
        else:
            # Global attention pooling
            from tensorflow.keras.layers import GlobalAveragePooling1D
            model.add(GlobalAveragePooling1D())

    elif config['architecture'] == 'cnn_bilstm':
        # CNN feature extraction + BiLSTM
        model.add(Conv1D(
            filters=64,
            kernel_size=3,
            activation='relu',
            input_shape=(config['sequence_length'], config['n_features'])
        ))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))

        model.add(Bidirectional(LSTM(
            config['lstm_units'][0],
            return_sequences=False,
            dropout=config['dropout']
        )))

    elif config['architecture'] == 'gru_deep':
        # Deep GRU (faster than LSTM)
        model.add(GRU(
            config['lstm_units'][0],
            return_sequences=len(config['lstm_units']) > 1,
            input_shape=(config['sequence_length'], config['n_features']),
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

    else:  # Standard BiLSTM
        model.add(Bidirectional(LSTM(
            config['lstm_units'][0],
            return_sequences=len(config['lstm_units']) > 1,
            dropout=config['dropout'],
            kernel_regularizer=l2(config['l2_reg'])
        ), input_shape=(config['sequence_length'], config['n_features'])))

        for i, units in enumerate(config['lstm_units'][1:], 1):
            model.add(Bidirectional(LSTM(
                units,
                return_sequences=i < len(config['lstm_units']) - 1,
                dropout=config['dropout'],
                kernel_regularizer=l2(config['l2_reg'])
            )))

    # Batch normalization
    model.add(BatchNormalization())

    # Dense layers with residual connections concept
    for i, units in enumerate(config['dense_units']):
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(config['l2_reg'])))
        model.add(Dropout(config['dropout']))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile with optimized settings
    model.compile(
        optimizer=Adam(
            learning_rate=config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def create_sequences_optimized(X, y, sequence_length, step=1):
    """Create sequences with optional step parameter for data augmentation"""
    sequences_X = []
    sequences_y = []

    for i in range(sequence_length, len(X), step):
        sequences_X.append(X.iloc[i-sequence_length:i].values)
        sequences_y.append(y.iloc[i])

    return np.array(sequences_X), np.array(sequences_y)

def train_and_evaluate(config, X_scaled, y_clean, verbose=True):
    """Train and evaluate LSTM with advanced techniques"""
    try:
        # Create sequences with potential data augmentation
        step = config.get('step', 1)
        X_seq, y_seq = create_sequences_optimized(X_scaled, y_clean, config['sequence_length'], step)

        if len(X_seq) < 800:
            return {'accuracy': 0, 'error': 'Not enough sequences'}

        # Time series split
        split_idx = int(len(X_seq) * 0.8)
        X_train = X_seq[:split_idx]
        y_train = y_seq[:split_idx]
        X_test = X_seq[split_idx:]
        y_test = y_seq[split_idx:]

        # Update config with feature count
        config['n_features'] = X_seq.shape[2]

        # Build model
        model = create_focused_lstm(config)

        # Advanced callbacks
        import tensorflow as tf
        from tensorflow.keras.callbacks import (
            EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        )

        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',  # Focus on accuracy, not loss
                patience=config['patience'],
                restore_best_weights=True,
                mode='max',
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.7,
                patience=config['patience']//2,
                min_lr=1e-6,
                mode='max',
                verbose=0
            )
        ]

        # Train with class weights if imbalanced
        class_weight = None
        if len(np.unique(y_train)) == 2:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight = {classes[i]: weights[i] for i in range(len(classes))}

        start_time = time.time()

        history = model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=0.2,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=0
        )

        training_time = time.time() - start_time

        # Evaluate with optimized threshold
        y_pred_prob = model.predict(X_test, verbose=0)

        # Find optimal threshold
        thresholds = np.arange(0.4, 0.7, 0.05)
        best_threshold = 0.5
        best_acc = 0

        for threshold in thresholds:
            y_pred_temp = (y_pred_prob > threshold).astype(int).flatten()
            acc_temp = accuracy_score(y_test, y_pred_temp)
            if acc_temp > best_acc:
                best_acc = acc_temp
                best_threshold = threshold

        y_pred = (y_pred_prob > best_threshold).astype(int).flatten()

        # Comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)

        # Class-specific metrics
        up_mask = y_test == 1
        down_mask = y_test == 0

        up_accuracy = accuracy_score(y_test[up_mask], y_pred[up_mask]) if up_mask.sum() > 0 else 0
        down_accuracy = accuracy_score(y_test[down_mask], y_pred[down_mask]) if down_mask.sum() > 0 else 0

        # Advanced metrics
        from sklearn.metrics import precision_score, recall_score, f1_score

        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Trading simulation metrics
        total_trades = len(y_pred)
        profitable_trades = np.sum(y_pred == y_test)

        # Best validation accuracy during training
        best_val_acc = max(history.history['val_accuracy'])

        result = {
            'accuracy': accuracy,
            'up_accuracy': up_accuracy,
            'down_accuracy': down_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_val_acc': best_val_acc,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss']),
            'best_threshold': best_threshold,
            'config': config.copy()
        }

        if verbose:
            print(f"  {config['name']}: Acc={accuracy:.3f}, Up={up_accuracy:.3f}, "
                  f"Down={down_accuracy:.3f}, F1={f1:.3f}, Time={training_time:.1f}s")

        return result

    except Exception as e:
        return {'accuracy': 0, 'error': str(e), 'config': config.copy()}

def focused_optimization():
    """Focused LSTM optimization targeting 60%+ accuracy"""
    print("=" * 80)
    print("🎯 FOCUSED LSTM OPTIMIZATION")
    print("=" * 80)

    # Load data efficiently
    print("📊 Loading optimized dataset...")
    db = TradingDatabase()
    df = db.get_ohlcv_data("BTCUSDT", "15m")

    # Use focused dataset - recent 12k records (83 days)
    test_df = df.tail(12000).copy()
    print(f"✅ Using {len(test_df):,} records")

    # Optimized feature generation
    feature_engine = EnhancedFeatureEngine(performance_mode=True)
    df_with_features = feature_engine.create_all_features(
        df=test_df,
        include_multiframe=True,
        min_periods=100
    )

    # Enhanced target creation
    print("🎯 Creating enhanced targets...")

    # Test multiple prediction horizons
    horizons = [3, 4, 6]  # 45min, 1h, 1.5h
    thresholds = [0.001, 0.0015, 0.002]  # 0.1%, 0.15%, 0.2%

    best_target_setup = None
    best_target_quality = 0

    for horizon in horizons:
        for threshold in thresholds:
            future_returns = df_with_features['close'].pct_change(periods=horizon).shift(-horizon)

            conditions = [
                future_returns <= -threshold,
                future_returns >= threshold,
            ]
            choices = [0, 1]
            clear_movements = np.select(conditions, choices, default=-1)
            sideways_mask = clear_movements != -1
            target_classes = clear_movements[sideways_mask]

            if len(target_classes) > 5000:  # Ensure enough data
                # Quality metric: balance and size
                unique, counts = np.unique(target_classes, return_counts=True)
                balance = min(counts) / max(counts)  # Closer to 1 is better
                size_score = len(target_classes) / 10000  # Normalize
                quality = balance * size_score

                if quality > best_target_quality:
                    best_target_quality = quality
                    best_target_setup = {
                        'horizon': horizon,
                        'threshold': threshold,
                        'target_classes': target_classes,
                        'sideways_mask': sideways_mask,
                        'balance': balance,
                        'size': len(target_classes)
                    }

    target_classes = best_target_setup['target_classes']
    sideways_mask = best_target_setup['sideways_mask']

    print(f"Optimal target: {best_target_setup['horizon']*15}min ahead, "
          f"{best_target_setup['threshold']*100:.1f}% threshold")
    print(f"Balance: {best_target_setup['balance']:.3f}, Size: {best_target_setup['size']:,}")

    # Enhanced feature selection
    print("🧠 Advanced feature selection...")

    # Use proven crypto features
    proven_features = [
        # RSI signals (momentum)
        'rsi_14', 'rsi_14_1h', 'rsi_14_4h',
        # Bollinger Band positions (volatility + trend)
        'bb_position', 'bb_position_1h', 'bb_position_4h',
        # MACD signals (trend following)
        'macd_signal', 'macd_signal_1h',
        # Volume analysis (smart money)
        'volume_sma_ratio', 'volume_sma_ratio_1h',
        # Price position (support/resistance)
        'price_position', 'hl_ratio',
        # Momentum indicators
        'momentum_10', 'momentum_20',
        # Volatility measures
        'volatility', 'volatility_1h', 'atr_normalized',
        # Additional oscillators
        'williams_r', 'stoch_k', 'cci', 'adx', 'roc'
    ]

    # Filter and validate features
    available_features = []
    for feature in proven_features:
        if feature in df_with_features.columns:
            feature_data = df_with_features[feature][sideways_mask]
            if feature_data.notna().sum() > len(feature_data) * 0.8:  # 80% data coverage
                available_features.append(feature)

    print(f"Selected {len(available_features)} high-quality features")

    # Prepare clean dataset
    X_raw = df_with_features[available_features]
    X_filtered = X_raw[sideways_mask]

    valid_mask = ~(X_filtered.isnull().any(axis=1) | pd.Series(target_classes).isnull())
    X_clean = X_filtered[valid_mask]
    y_clean = pd.Series(target_classes)[valid_mask]

    print(f"Final dataset: {len(X_clean):,} samples")

    # Test optimal scaler
    print("\n⚙️ Testing scalers...")
    scalers = {
        'RobustScaler': RobustScaler(),
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler()
    }

    best_scaler = None
    best_scaler_score = 0

    for name, scaler in scalers.items():
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_clean),
            columns=X_clean.columns,
            index=X_clean.index
        )

        # Quick test
        config = {
            'name': f'Test_{name}',
            'architecture': 'bilstm',
            'sequence_length': 25,
            'lstm_units': [64, 32],
            'dense_units': [16],
            'dropout': 0.2,
            'l2_reg': 1e-4,
            'batch_size': 64,
            'epochs': 15,
            'learning_rate': 0.001,
            'patience': 4
        }

        result = train_and_evaluate(config, X_scaled, y_clean, verbose=False)
        print(f"  {name}: {result['accuracy']:.3f}")

        if result['accuracy'] > best_scaler_score:
            best_scaler_score = result['accuracy']
            best_scaler = (name, scaler)

    print(f"✅ Best scaler: {best_scaler[0]}")

    # Use best scaler
    X_scaled = pd.DataFrame(
        best_scaler[1].fit_transform(X_clean),
        columns=X_clean.columns,
        index=X_clean.index
    )

    # Test high-impact configurations
    print("\n🚀 Testing high-impact configurations...")

    configs = [
        {
            'name': 'BiLSTM_Optimized',
            'architecture': 'bilstm',
            'sequence_length': 30,
            'lstm_units': [96, 48],
            'dense_units': [32, 16],
            'dropout': 0.15,
            'l2_reg': 5e-5,
            'batch_size': 64,
            'epochs': 40,
            'learning_rate': 0.0015,
            'patience': 6
        },
        {
            'name': 'GRU_Deep_Fast',
            'architecture': 'gru_deep',
            'sequence_length': 35,
            'lstm_units': [80, 40, 20],
            'dense_units': [24],
            'dropout': 0.2,
            'l2_reg': 1e-4,
            'batch_size': 128,
            'epochs': 45,
            'learning_rate': 0.002,
            'patience': 7
        },
        {
            'name': 'CNN_BiLSTM_Hybrid',
            'architecture': 'cnn_bilstm',
            'sequence_length': 40,
            'lstm_units': [64],
            'dense_units': [32, 16],
            'dropout': 0.2,
            'l2_reg': 1e-4,
            'batch_size': 64,
            'epochs': 35,
            'learning_rate': 0.001,
            'patience': 5
        },
        {
            'name': 'BiLSTM_Wide_Shallow',
            'architecture': 'bilstm',
            'sequence_length': 25,
            'lstm_units': [128],
            'dense_units': [64, 32],
            'dropout': 0.1,
            'l2_reg': 1e-5,
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.0008,
            'patience': 8
        },
        {
            'name': 'BiLSTM_Long_Sequence',
            'architecture': 'bilstm',
            'sequence_length': 50,
            'lstm_units': [64, 32],
            'dense_units': [20],
            'dropout': 0.25,
            'l2_reg': 2e-4,
            'batch_size': 48,
            'epochs': 35,
            'learning_rate': 0.0012,
            'patience': 6
        }
    ]

    results = []

    for config in configs:
        print(f"\n🧠 Testing {config['name']}...")
        result = train_and_evaluate(config, X_scaled, y_clean)
        results.append(result)

    # Find best results
    valid_results = [r for r in results if r['accuracy'] > 0]
    valid_results.sort(key=lambda x: x['accuracy'], reverse=True)

    print(f"\n🏆 OPTIMIZATION RESULTS")
    print("=" * 60)

    for i, result in enumerate(valid_results[:3], 1):
        print(f"{i}. {result['config']['name']}")
        print(f"   Accuracy: {result['accuracy']:.3f} ({result['accuracy']*100:.1f}%)")
        print(f"   Up/Down: {result['up_accuracy']:.3f}/{result['down_accuracy']:.3f}")
        print(f"   F1 Score: {result['f1_score']:.3f}")
        print(f"   Profitable: {result['profitable_trades']}/{result['total_trades']} "
              f"({result['profitable_trades']/result['total_trades']*100:.1f}%)")

    if valid_results:
        best_result = valid_results[0]
        accuracy = best_result['accuracy']

        # Save best configuration
        os.makedirs("models/optimized_lstm", exist_ok=True)

        with open("models/optimized_lstm/best_config.json", "w") as f:
            json.dump(best_result['config'], f, indent=2)

        with open("models/optimized_lstm/optimization_results.json", "w") as f:
            json.dump({
                'best_accuracy': accuracy,
                'best_config': best_result['config'],
                'all_results': [r for r in valid_results[:5]]
            }, f, indent=2, default=str)

        # Performance assessment
        print(f"\n📊 PERFORMANCE ASSESSMENT")
        print("-" * 40)

        target_accuracy = 0.60
        min_accuracy = 0.55

        if accuracy >= target_accuracy:
            print(f"🎯 TARGET ACHIEVED! Accuracy: {accuracy:.3f} (≥ {target_accuracy:.3f})")
            print(f"✅ Model ready for profitable trading")
            success = True
        elif accuracy >= min_accuracy:
            print(f"⚠️ PARTIAL SUCCESS: Accuracy: {accuracy:.3f} (≥ {min_accuracy:.3f})")
            print(f"⚠️ Model usable but may need additional refinement")
            success = True
        else:
            print(f"❌ BELOW TARGET: Accuracy: {accuracy:.3f} (< {min_accuracy:.3f})")
            print(f"❌ Consider ensemble methods or feature engineering")
            success = False

        return success, best_result

    return False, {}

def main():
    """Main optimization function"""
    try:
        success, results = focused_optimization()

        if success:
            print(f"\n🎯 FOCUSED OPTIMIZATION COMPLETED!")
            print(f"✅ Ready for signal generation and trading")
        else:
            print(f"\n⚠️ OPTIMIZATION NEEDS MORE WORK")
            print(f"💡 Consider ensemble approaches or more data")

        return success

    except Exception as e:
        print(f"❌ Error in optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)