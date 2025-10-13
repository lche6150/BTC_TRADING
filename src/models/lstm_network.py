#!/usr/bin/env python3
"""
Advanced LSTM Neural Network for Cryptocurrency Trading
Implements state-of-the-art 2024 architectures including CNN-LSTM, Attention mechanisms, and Hybrid approaches
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
import pickle
import os
from pathlib import Path
import time
from datetime import datetime
import warnings

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, Bidirectional, Attention,
    MultiHeadAttention, LayerNormalization, Input,
    GlobalMaxPooling1D, Concatenate, TimeDistributed
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard, CSVLogger
)

# ML Libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optimization
import optuna
try:
    from optuna.integration import TensorFlowPruningCallback
except ImportError:
    TensorFlowPruningCallback = None

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Configure GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

@dataclass
class LSTMConfig:
    """Configuration for LSTM Neural Network"""
    # Architecture Parameters
    model_type: str = "hybrid_cnn_lstm"  # "lstm", "bilstm", "cnn_lstm", "hybrid_cnn_lstm", "transformer_lstm"
    sequence_length: int = 60  # Time steps (15 hours for 15-min intervals)
    lstm_units: List[int] = None  # LSTM layer sizes
    dense_units: List[int] = None  # Dense layer sizes

    # CNN Parameters (for CNN-LSTM variants)
    cnn_filters: List[int] = None
    cnn_kernel_sizes: List[int] = None

    # Attention Parameters
    attention_heads: int = 8
    attention_dropout: float = 0.1

    # Regularization
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.0  # Often disabled for GPU acceleration
    l2_reg: float = 1e-4
    batch_normalization: bool = True

    # Training Parameters
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    optimizer: str = "adam"  # "adam", "rmsprop"

    # Loss and Metrics
    loss_function: str = "mse"  # "mse", "mae", "huber", "gmadl"
    metrics: List[str] = None

    # Validation
    validation_split: float = 0.15
    test_split: float = 0.15
    shuffle_validation: bool = False  # Keep False for time series

    # Callbacks
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5

    # Feature Engineering
    scaler_type: str = "minmax"  # "minmax", "standard"
    target_scaling: bool = True

    # XGBoost Hybrid Parameters
    enable_xgb_hybrid: bool = True
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1

    # Random State
    random_seed: int = 42

    def __post_init__(self):
        """Set default values for mutable defaults"""
        if self.lstm_units is None:
            self.lstm_units = [64, 32]
        if self.dense_units is None:
            self.dense_units = [64, 32]
        if self.cnn_filters is None:
            self.cnn_filters = [64, 32]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 3]
        if self.metrics is None:
            self.metrics = ['mae']

@dataclass
class TrainingResults:
    """Results from LSTM training"""
    # Model Performance
    train_loss: float = 0.0
    val_loss: float = 0.0
    test_loss: float = 0.0
    train_mae: float = 0.0
    val_mae: float = 0.0
    test_mae: float = 0.0
    train_mape: float = 0.0
    val_mape: float = 0.0
    test_mape: float = 0.0
    directional_accuracy: float = 0.0
    r2_score: float = 0.0

    # Training Info
    epochs_trained: int = 0
    training_time: float = 0.0
    best_epoch: int = 0

    # Model Paths
    model_path: str = ""
    scaler_path: str = ""
    config_path: str = ""

    # Predictions
    train_predictions: np.ndarray = None
    val_predictions: np.ndarray = None
    test_predictions: np.ndarray = None

    # Feature Importance (XGBoost hybrid)
    feature_importance: Dict[str, float] = None

    def __post_init__(self):
        if self.feature_importance is None:
            self.feature_importance = {}

class CustomLosses:
    """Custom loss functions for cryptocurrency trading"""

    @staticmethod
    def generalized_mean_absolute_directional_loss(alpha: float = 1.0):
        """
        Generalized Mean Absolute Directional Loss (GMADL)
        Superior performance for directional prediction in crypto trading
        """
        def gmadl(y_true, y_pred):
            # Calculate directional errors
            true_direction = tf.sign(y_true[1:] - y_true[:-1])
            pred_direction = tf.sign(y_pred[1:] - y_pred[:-1])

            # Directional accuracy component
            directional_error = tf.abs(true_direction - pred_direction)

            # Magnitude error component
            magnitude_error = tf.abs(y_true - y_pred)

            # Combined loss with weighting
            combined_loss = alpha * tf.reduce_mean(directional_error) + \
                          (1 - alpha) * tf.reduce_mean(magnitude_error)

            return combined_loss

        return gmadl

    @staticmethod
    def weighted_empirical_stretching_loss(class_weights: Dict[int, float]):
        """
        Weighted Empirical Stretching (WES) Loss
        Different penalties based on label distribution
        """
        def wes_loss(y_true, y_pred):
            # Convert to class predictions
            class_true = tf.argmax(y_true, axis=1)
            class_pred = tf.argmax(y_pred, axis=1)

            # Calculate weighted loss
            weights = tf.gather([class_weights.get(i, 1.0) for i in range(len(class_weights))],
                              class_true)

            # Standard categorical crossentropy with weights
            ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            weighted_loss = ce_loss * weights

            return tf.reduce_mean(weighted_loss)

        return wes_loss

class AdvancedLSTMPredictor:
    """
    Advanced LSTM Neural Network for Cryptocurrency Trading
    Implements state-of-the-art 2024 architectures and techniques
    """

    def __init__(self, config: LSTMConfig = None):
        """Initialize the LSTM predictor"""
        self.config = config or LSTMConfig()
        self.logger = self._setup_logging()

        # Set random seeds for reproducibility
        self._set_random_seeds()

        # Initialize components
        self.model = None
        self.xgb_model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_columns = None
        self.training_results = TrainingResults()

        self.logger.info("Advanced LSTM Predictor initialized")
        self.logger.info(f"Model type: {self.config.model_type}")
        self.logger.info(f"Sequence length: {self.config.sequence_length}")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        np.random.seed(self.config.random_seed)
        tf.random.set_seed(self.config.random_seed)

        # For XGBoost
        import random
        random.seed(self.config.random_seed)

    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        Uses sliding window technique with specified sequence length
        """
        sequences = []
        targets = []

        for i in range(self.config.sequence_length, len(X)):
            # Feature sequence
            seq = X[i-self.config.sequence_length:i]
            sequences.append(seq)

            # Target (if provided)
            if y is not None:
                targets.append(y[i])

        X_seq = np.array(sequences)
        y_seq = np.array(targets) if y is not None else None

        self.logger.info(f"Created {len(sequences)} sequences with length {self.config.sequence_length}")

        return X_seq, y_seq

    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training with proper scaling and sequencing
        """
        self.logger.info("Preparing data for LSTM training...")

        # Store feature columns
        self.feature_columns = X.columns.tolist()

        # Initialize scalers
        if self.config.scaler_type == "minmax":
            self.feature_scaler = MinMaxScaler()
            if self.config.target_scaling:
                self.target_scaler = MinMaxScaler()
        else:  # standard
            self.feature_scaler = StandardScaler()
            if self.config.target_scaling:
                self.target_scaler = StandardScaler()

        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X.values)

        # Scale target if enabled
        if self.config.target_scaling and self.target_scaler is not None:
            y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
        else:
            y_scaled = y.values

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)

        # Train/validation/test split (time series aware)
        n_samples = len(X_seq)
        train_size = int(n_samples * (1 - self.config.validation_split - self.config.test_split))
        val_size = int(n_samples * self.config.validation_split)

        # Split data
        X_train = X_seq[:train_size]
        X_val = X_seq[train_size:train_size + val_size]
        X_test = X_seq[train_size + val_size:]

        y_train = y_seq[:train_size]
        y_val = y_seq[train_size:train_size + val_size]
        y_test = y_seq[train_size + val_size:]

        self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build standard LSTM model"""
        model = Sequential()

        # LSTM layers
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1

            if i == 0:
                model.add(LSTM(
                    units,
                    return_sequences=return_sequences,
                    input_shape=input_shape,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.recurrent_dropout
                ))
            else:
                model.add(LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.recurrent_dropout
                ))

            if self.config.batch_normalization:
                model.add(BatchNormalization())

        # Dense layers
        for units in self.config.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.config.dropout_rate))

        # Output layer
        model.add(Dense(1, activation='linear'))

        return model

    def _build_bilstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build Bidirectional LSTM model with temporal attention"""
        model = Sequential()

        # Bidirectional LSTM layers
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1

            if i == 0:
                model.add(Bidirectional(LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.recurrent_dropout
                ), input_shape=input_shape))
            else:
                model.add(Bidirectional(LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.recurrent_dropout
                )))

            if self.config.batch_normalization:
                model.add(BatchNormalization())

        # Dense layers
        for units in self.config.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.config.dropout_rate))

        # Output layer
        model.add(Dense(1, activation='linear'))

        return model

    def _build_cnn_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build CNN-LSTM hybrid model"""
        model = Sequential()

        # CNN layers for feature extraction
        for i, (filters, kernel_size) in enumerate(zip(self.config.cnn_filters, self.config.cnn_kernel_sizes)):
            if i == 0:
                model.add(Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    input_shape=input_shape
                ))
            else:
                model.add(Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation='relu'
                ))

            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(self.config.dropout_rate))

        # LSTM layers
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1

            model.add(LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout
            ))

            if self.config.batch_normalization:
                model.add(BatchNormalization())

        # Dense layers
        for units in self.config.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.config.dropout_rate))

        # Output layer
        model.add(Dense(1, activation='linear'))

        return model

    def _build_hybrid_cnn_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build advanced hybrid CNN-LSTM model with attention
        Based on 2024 state-of-the-art architectures
        """
        # Input layer
        inputs = Input(shape=input_shape)

        # CNN branch for local feature extraction
        cnn_branch = inputs
        for i, (filters, kernel_size) in enumerate(zip(self.config.cnn_filters, self.config.cnn_kernel_sizes)):
            cnn_branch = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same'
            )(cnn_branch)
            cnn_branch = BatchNormalization()(cnn_branch) if self.config.batch_normalization else cnn_branch
            cnn_branch = Dropout(self.config.dropout_rate)(cnn_branch)

        # LSTM branch for temporal dependencies
        lstm_branch = inputs
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = True  # Always return sequences for attention

            lstm_branch = Bidirectional(LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout
            ))(lstm_branch)

            if self.config.batch_normalization:
                lstm_branch = BatchNormalization()(lstm_branch)

        # Attention mechanism
        attention_output = MultiHeadAttention(
            num_heads=self.config.attention_heads,
            key_dim=self.config.lstm_units[-1] * 2,  # Bidirectional doubles the size
            dropout=self.config.attention_dropout
        )(lstm_branch, lstm_branch)

        attention_output = LayerNormalization()(attention_output + lstm_branch)  # Residual connection

        # Global pooling
        cnn_pooled = GlobalMaxPooling1D()(cnn_branch)
        lstm_pooled = GlobalMaxPooling1D()(attention_output)

        # Concatenate branches
        merged = Concatenate()([cnn_pooled, lstm_pooled])

        # Dense layers
        x = merged
        for units in self.config.dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.config.dropout_rate)(x)
            if self.config.batch_normalization:
                x = BatchNormalization()(x)

        # Output layer
        outputs = Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def _build_transformer_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build Transformer-LSTM hybrid (Helformer-inspired)
        Combines transformer attention with LSTM temporal modeling
        """
        inputs = Input(shape=input_shape)

        # Multi-head self-attention (Transformer encoder)
        attention_output = MultiHeadAttention(
            num_heads=self.config.attention_heads,
            key_dim=input_shape[1],  # Feature dimension
            dropout=self.config.attention_dropout
        )(inputs, inputs)

        # Add & Norm
        attention_output = LayerNormalization()(attention_output + inputs)

        # Feed-forward network
        ffn_output = Dense(input_shape[1] * 4, activation='relu')(attention_output)
        ffn_output = Dropout(self.config.dropout_rate)(ffn_output)
        ffn_output = Dense(input_shape[1])(ffn_output)

        # Add & Norm
        transformer_output = LayerNormalization()(ffn_output + attention_output)

        # LSTM decoder
        lstm_output = transformer_output
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1

            lstm_output = LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout
            )(lstm_output)

            if self.config.batch_normalization:
                lstm_output = BatchNormalization()(lstm_output)

        # Dense layers
        x = lstm_output
        for units in self.config.dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.config.dropout_rate)(x)

        # Output layer
        outputs = Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def _build_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build model based on configuration"""
        self.logger.info(f"Building {self.config.model_type} model...")

        if self.config.model_type == "lstm":
            model = self._build_lstm_model(input_shape)
        elif self.config.model_type == "bilstm":
            model = self._build_bilstm_model(input_shape)
        elif self.config.model_type == "cnn_lstm":
            model = self._build_cnn_lstm_model(input_shape)
        elif self.config.model_type == "hybrid_cnn_lstm":
            model = self._build_hybrid_cnn_lstm_model(input_shape)
        elif self.config.model_type == "transformer_lstm":
            model = self._build_transformer_lstm_model(input_shape)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        return model

    def _get_optimizer(self) -> optimizers.Optimizer:
        """Get optimizer based on configuration"""
        if self.config.optimizer.lower() == "adam":
            return Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer.lower() == "rmsprop":
            return RMSprop(learning_rate=self.config.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _get_loss_function(self):
        """Get loss function based on configuration"""
        if self.config.loss_function == "mse":
            return "mse"
        elif self.config.loss_function == "mae":
            return "mae"
        elif self.config.loss_function == "huber":
            return tf.keras.losses.Huber()
        elif self.config.loss_function == "gmadl":
            return CustomLosses.generalized_mean_absolute_directional_loss(alpha=0.7)
        else:
            return self.config.loss_function

    def _get_callbacks(self, model_path: str) -> List[callbacks.Callback]:
        """Get training callbacks"""
        callback_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]

        return callback_list

    def train(self, X: pd.DataFrame, y: pd.Series, save_dir: str = "models") -> TrainingResults:
        """
        Train the LSTM neural network
        """
        self.logger.info("Starting LSTM neural network training...")

        start_time = time.time()

        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data(X, y)

        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, n_features)
        self.model = self._build_model(input_shape)

        # Compile model
        optimizer = self._get_optimizer()
        loss_fn = self._get_loss_function()

        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=self.config.metrics
        )

        self.logger.info("Model architecture:")
        self.model.summary()

        # Prepare callbacks
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "best_lstm_model.keras")
        callback_list = self._get_callbacks(model_path)

        # Train model
        self.logger.info("Training LSTM model...")

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            callbacks=callback_list,
            verbose=1,
            shuffle=self.config.shuffle_validation
        )

        training_time = time.time() - start_time

        # Keep the trained model (avoid loading issues)
        # The model checkpoint callback already saved the best weights
        # self.model = keras.models.load_model(model_path)

        # Make predictions
        self.logger.info("Making predictions...")

        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        test_pred = self.model.predict(X_test)

        # Inverse transform if scaling was applied
        if self.config.target_scaling and self.target_scaler is not None:
            y_train_orig = self.target_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
            y_val_orig = self.target_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
            y_test_orig = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

            train_pred_orig = self.target_scaler.inverse_transform(train_pred).ravel()
            val_pred_orig = self.target_scaler.inverse_transform(val_pred).ravel()
            test_pred_orig = self.target_scaler.inverse_transform(test_pred).ravel()
        else:
            y_train_orig = y_train
            y_val_orig = y_val
            y_test_orig = y_test
            train_pred_orig = train_pred.ravel()
            val_pred_orig = val_pred.ravel()
            test_pred_orig = test_pred.ravel()

        # Calculate metrics
        train_mse = mean_squared_error(y_train_orig, train_pred_orig)
        val_mse = mean_squared_error(y_val_orig, val_pred_orig)
        test_mse = mean_squared_error(y_test_orig, test_pred_orig)

        train_mae = mean_absolute_error(y_train_orig, train_pred_orig)
        val_mae = mean_absolute_error(y_val_orig, val_pred_orig)
        test_mae = mean_absolute_error(y_test_orig, test_pred_orig)

        # MAPE calculation
        def calculate_mape(actual, predicted):
            mask = actual != 0
            return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

        train_mape = calculate_mape(y_train_orig, train_pred_orig)
        val_mape = calculate_mape(y_val_orig, val_pred_orig)
        test_mape = calculate_mape(y_test_orig, test_pred_orig)

        # Directional accuracy
        def calculate_directional_accuracy(actual, predicted):
            actual_direction = np.sign(np.diff(actual))
            pred_direction = np.sign(np.diff(predicted))
            return np.mean(actual_direction == pred_direction) * 100

        directional_acc = calculate_directional_accuracy(y_test_orig, test_pred_orig)

        # R² score
        r2 = r2_score(y_test_orig, test_pred_orig)

        # Store results
        best_epoch = np.argmin(history.history['val_loss']) + 1

        self.training_results = TrainingResults(
            train_loss=train_mse,
            val_loss=val_mse,
            test_loss=test_mse,
            train_mae=train_mae,
            val_mae=val_mae,
            test_mae=test_mae,
            train_mape=train_mape,
            val_mape=val_mape,
            test_mape=test_mape,
            directional_accuracy=directional_acc,
            r2_score=r2,
            epochs_trained=len(history.history['loss']),
            training_time=training_time,
            best_epoch=best_epoch,
            model_path=model_path,
            train_predictions=train_pred_orig,
            val_predictions=val_pred_orig,
            test_predictions=test_pred_orig
        )

        # XGBoost hybrid if enabled
        if self.config.enable_xgb_hybrid:
            self.logger.info("Training XGBoost hybrid model...")
            self._train_xgb_hybrid(X_train, X_val, X_test, y_train, y_val, y_test)

        # Save additional components
        self._save_components(save_dir)

        self.logger.info("LSTM training completed!")
        self.logger.info(f"Training time: {training_time:.2f} seconds")
        self.logger.info(f"Test MSE: {test_mse:.6f}")
        self.logger.info(f"Test MAE: {test_mae:.6f}")
        self.logger.info(f"Test MAPE: {test_mape:.2f}%")
        self.logger.info(f"Directional accuracy: {directional_acc:.2f}%")
        self.logger.info(f"R² score: {r2:.4f}")

        return self.training_results

    def _train_xgb_hybrid(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                         y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
        """Train XGBoost hybrid model using LSTM features"""

        # Extract LSTM features (before final dense layer)
        try:
            # For Functional models
            if hasattr(self.model, 'input') and self.model.input is not None:
                feature_extractor = Model(
                    inputs=self.model.input,
                    outputs=self.model.layers[-2].output
                )
            else:
                # For Sequential models, we need to build the input first
                self.logger.info("Sequential model detected, skipping XGBoost hybrid")
                return
        except AttributeError:
            # Sequential models don't have input defined until called
            self.logger.info("Cannot create feature extractor for Sequential model, skipping XGBoost hybrid")
            return

        # Get LSTM features
        lstm_features_train = feature_extractor.predict(X_train)
        lstm_features_val = feature_extractor.predict(X_val)
        lstm_features_test = feature_extractor.predict(X_test)

        # Combine with original features (flatten last time step)
        orig_features_train = X_train[:, -1, :]  # Last time step
        orig_features_val = X_val[:, -1, :]
        orig_features_test = X_test[:, -1, :]

        # Create combined feature matrix
        combined_train = np.concatenate([lstm_features_train, orig_features_train], axis=1)
        combined_val = np.concatenate([lstm_features_val, orig_features_val], axis=1)
        combined_test = np.concatenate([lstm_features_test, orig_features_test], axis=1)

        # Train XGBoost
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            random_state=self.config.random_seed
        )

        # Fit on combined train + validation set
        X_combined = np.concatenate([combined_train, combined_val])
        y_combined = np.concatenate([y_train, y_val])

        self.xgb_model.fit(X_combined, y_combined)

        # Get feature importance
        feature_names = [f'lstm_feat_{i}' for i in range(lstm_features_train.shape[1])] + \
                       [f'orig_feat_{i}' for i in range(orig_features_train.shape[1])]

        importance_scores = self.xgb_model.feature_importances_
        self.training_results.feature_importance = dict(zip(feature_names, importance_scores))

        self.logger.info("XGBoost hybrid model trained successfully")

    def _save_components(self, save_dir: str):
        """Save model components"""
        # Save scalers
        if self.feature_scaler is not None:
            scaler_path = os.path.join(save_dir, "feature_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            self.training_results.scaler_path = scaler_path

        if self.target_scaler is not None:
            target_scaler_path = os.path.join(save_dir, "target_scaler.pkl")
            with open(target_scaler_path, 'wb') as f:
                pickle.dump(self.target_scaler, f)

        # Save config
        config_path = os.path.join(save_dir, "lstm_config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=4)
        self.training_results.config_path = config_path

        # Save XGBoost model if exists
        if self.xgb_model is not None:
            xgb_path = os.path.join(save_dir, "xgb_hybrid_model.pkl")
            with open(xgb_path, 'wb') as f:
                pickle.dump(self.xgb_model, f)

        # Save feature columns
        feature_cols_path = os.path.join(save_dir, "feature_columns.pkl")
        with open(feature_cols_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Scale features
        X_scaled = self.feature_scaler.transform(X.values)

        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled)

        # Make predictions
        predictions = self.model.predict(X_seq)

        # Inverse transform if scaling was applied
        if self.config.target_scaling and self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions)

        return predictions.ravel()

    def create_visualizations(self, save_dir: str = "visualizations"):
        """Create comprehensive visualizations of training results"""
        os.makedirs(save_dir, exist_ok=True)

        # 1. Training history
        self._plot_training_history(save_dir)

        # 2. Predictions vs actual
        self._plot_predictions(save_dir)

        # 3. Feature importance (if XGBoost hybrid)
        if self.training_results.feature_importance:
            self._plot_feature_importance(save_dir)

        # 4. Residual analysis
        self._plot_residuals(save_dir)

        self.logger.info(f"Visualizations saved to {save_dir}")

    def _plot_training_history(self, save_dir: str):
        """Plot training history"""
        # This would require storing history - simplified version
        plt.figure(figsize=(12, 4))

        # Placeholder for actual history plotting
        plt.subplot(1, 2, 1)
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(1, 2, 2)
        plt.title("Model Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_history.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_predictions(self, save_dir: str):
        """Plot predictions vs actual values"""
        if self.training_results.test_predictions is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Test predictions time series
        axes[0, 0].plot(self.training_results.test_predictions, label='Predicted', alpha=0.7)
        axes[0, 0].set_title('Test Predictions Time Series')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Scatter plot
        if hasattr(self, '_test_actual'):  # Would need to store actual values
            axes[0, 1].scatter(self._test_actual, self.training_results.test_predictions, alpha=0.6)
            axes[0, 1].plot([self._test_actual.min(), self._test_actual.max()],
                          [self._test_actual.min(), self._test_actual.max()], 'r--', lw=2)
            axes[0, 1].set_xlabel('Actual')
            axes[0, 1].set_ylabel('Predicted')
            axes[0, 1].set_title('Actual vs Predicted')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "predictions.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(self, save_dir: str):
        """Plot feature importance from XGBoost hybrid"""
        if not self.training_results.feature_importance:
            return

        # Sort by importance
        sorted_features = sorted(
            self.training_results.feature_importance.items(),
            key=lambda x: x[1], reverse=True
        )[:20]  # Top 20

        features, importance = zip(*sorted_features)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('XGBoost Hybrid - Top 20 Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_residuals(self, save_dir: str):
        """Plot residual analysis"""
        # Simplified residual plot
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.title("Residuals Distribution")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        plt.title("Residuals vs Predictions")
        plt.xlabel("Predictions")
        plt.ylabel("Residuals")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "residuals.png"), dpi=300, bbox_inches='tight')
        plt.close()

def optimize_lstm_hyperparameters(X: pd.DataFrame, y: pd.Series,
                                n_trials: int = 20) -> Tuple[LSTMConfig, TrainingResults]:
    """
    Optimize LSTM hyperparameters using Optuna
    """
    def objective(trial):
        # Suggest hyperparameters
        lstm_unit_1 = trial.suggest_int('lstm_unit_1', 32, 128)
        lstm_unit_2 = trial.suggest_int('lstm_unit_2', 16, 64)
        dense_unit_1 = trial.suggest_int('dense_unit_1', 32, 128)

        config = LSTMConfig(
            model_type=trial.suggest_categorical('model_type',
                ['lstm', 'bilstm', 'cnn_lstm', 'hybrid_cnn_lstm']),
            sequence_length=trial.suggest_int('sequence_length', 30, 120),
            lstm_units=[lstm_unit_1, lstm_unit_2],
            dense_units=[dense_unit_1],
            dropout_rate=trial.suggest_float('dropout_rate', 0.1, 0.5),
            learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
            epochs=15,  # Reduce for optimization
        )

        # Train model
        predictor = AdvancedLSTMPredictor(config)
        try:
            results = predictor.train(X, y, save_dir=f"temp_models/trial_{trial.number}")
            return results.val_mae  # Minimize validation MAE
        except Exception as e:
            logging.error(f"Trial {trial.number} failed: {e}")
            return float('inf')

    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # Get best parameters and reconstruct config manually
    best_trial = study.best_trial
    best_config = LSTMConfig(
        model_type=best_trial.params['model_type'],
        sequence_length=best_trial.params['sequence_length'],
        lstm_units=[best_trial.params['lstm_unit_1'], best_trial.params['lstm_unit_2']],
        dense_units=[best_trial.params['dense_unit_1']],
        dropout_rate=best_trial.params['dropout_rate'],
        learning_rate=best_trial.params['learning_rate'],
        batch_size=best_trial.params['batch_size'],
        epochs=25  # More epochs for final training
    )

    # Train final model with best parameters
    predictor = AdvancedLSTMPredictor(best_config)
    final_results = predictor.train(X, y, save_dir="models/optimized_lstm")

    return best_config, final_results