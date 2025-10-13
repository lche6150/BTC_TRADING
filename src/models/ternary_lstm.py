#!/usr/bin/env python3
"""
Ternary LSTM Neural Network for BTC/USDT Trading
BiLSTM + Custom Sharpe Ratio Loss Function as per 2024 project requirements

Key Features:
- BiLSTM architecture with return_sequences=False for 3-layer: 64→32→16 units
- Custom loss function combining cross-entropy + Sharpe ratio optimization
- Ternary classification: Buy=2, Hold=1, Sell=0
- 60 timesteps with 20 XGBoost-selected features
- >1.2 validation Sharpe ratio target
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization,
    Bidirectional, Input, GlobalMaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

@dataclass
class TernaryLSTMConfig:
    """Configuration for Ternary LSTM as per project requirements"""
    # Architecture (as specified in project instructions)
    sequence_length: int = 60  # 60 timesteps
    n_features: int = 20       # 20 XGBoost-selected features
    num_classes: int = 3       # Buy=2, Hold=1, Sell=0

    # BiLSTM Architecture Specification
    lstm_units: List[int] = None  # [64, 32] - BiLSTM layers
    dense_units: int = 16         # Final dense layer before output

    # Regularization
    dropout_rate: float = 0.2
    l2_reg: float = 1e-4
    batch_normalization: bool = True

    # Training Parameters
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    validation_split: float = 0.15

    # Loss Function Weights
    crossentropy_weight: float = 0.7  # 70% cross-entropy
    sharpe_weight: float = 0.3        # 30% Sharpe ratio optimization

    # Early Stopping
    early_stopping_patience: int = 10
    early_stopping_monitor: str = 'val_sharpe_ratio'  # Monitor Sharpe ratio

    # Target Performance
    target_sharpe_ratio: float = 1.2  # >1.2 validation Sharpe ratio requirement

    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [64, 32]  # Default BiLSTM architecture

class TernaryLSTMNetwork:
    """
    Ternary LSTM Network implementing project requirements:
    - BiLSTM + Custom Sharpe ratio loss function
    - 3-class output (Buy=2, Hold=1, Sell=0)
    - Target >1.2 validation Sharpe ratio
    """

    def __init__(self, config: TernaryLSTMConfig = None):
        self.config = config or TernaryLSTMConfig()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.training_history = None

        # Custom metrics tracking
        self.sharpe_ratios = []
        self.validation_sharpe_ratios = []

    def create_model(self) -> Model:
        """
        Create BiLSTM model with custom Sharpe ratio loss as per project specs

        Network Architecture:
        Input Layer: (batch_size, 60, 20) # 60 timesteps, 20 selected features
        BiLSTM Layer 1: 64 units, return_sequences=True, dropout=0.2
        BiLSTM Layer 2: 32 units, return_sequences=False, dropout=0.2
        Dense Layer: 16 units, activation='relu', dropout=0.3
        Output Layer: 3 units, activation='softmax' # Buy/Hold/Sell probabilities
        """

        self.logger.info("Creating BiLSTM model with custom Sharpe loss...")

        # Input layer
        inputs = Input(shape=(self.config.sequence_length, self.config.n_features))

        # First BiLSTM layer (64 units, return sequences)
        x = Bidirectional(
            LSTM(
                self.config.lstm_units[0],
                return_sequences=True,
                dropout=self.config.dropout_rate,
                kernel_regularizer=keras.regularizers.l2(self.config.l2_reg)
            ),
            name='bilstm_1'
        )(inputs)

        if self.config.batch_normalization:
            x = BatchNormalization()(x)

        # Second BiLSTM layer (32 units, no return sequences)
        x = Bidirectional(
            LSTM(
                self.config.lstm_units[1],
                return_sequences=False,
                dropout=self.config.dropout_rate,
                kernel_regularizer=keras.regularizers.l2(self.config.l2_reg)
            ),
            name='bilstm_2'
        )(x)

        if self.config.batch_normalization:
            x = BatchNormalization()(x)

        # Dense layer (16 units)
        x = Dense(
            self.config.dense_units,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.config.l2_reg),
            name='dense_16'
        )(x)

        x = Dropout(0.3)(x)  # Higher dropout for dense layer

        # Output layer (3 classes: Buy/Hold/Sell)
        outputs = Dense(
            self.config.num_classes,
            activation='softmax',
            name='output_softmax'
        )(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='TernaryBiLSTM')

        # Compile with custom loss function
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=self._combined_loss_function,
            metrics=['accuracy', self._sharpe_ratio_metric]
        )

        self.model = model

        # Print model summary
        self.logger.info("BiLSTM Model Architecture:")
        model.summary(print_fn=self.logger.info)

        return model

    def _combined_loss_function(self, y_true, y_pred):
        """
        Custom loss function combining cross-entropy + Sharpe ratio optimization
        Based on project requirement for Sharpe ratio optimization
        """
        # Component 1: Standard cross-entropy loss for classification
        crossentropy_loss = keras.losses.categorical_crossentropy(y_true, y_pred)

        # Component 2: Sharpe ratio optimization
        # Convert predictions back to returns for Sharpe calculation
        # Buy=2 -> +1, Hold=1 -> 0, Sell=0 -> -1
        predicted_actions = tf.argmax(y_pred, axis=1)
        action_returns = tf.cast(predicted_actions, tf.float32) - 1.0  # Maps 0,1,2 to -1,0,1

        # Calculate Sharpe ratio (negative for minimization)
        mean_return = tf.reduce_mean(action_returns)
        std_return = tf.math.reduce_std(action_returns)
        sharpe_ratio = mean_return / (std_return + 1e-8)  # Add epsilon for stability
        sharpe_loss = -sharpe_ratio  # Negative because we want to maximize Sharpe

        # Combined loss
        combined_loss = (
            self.config.crossentropy_weight * crossentropy_loss +
            self.config.sharpe_weight * sharpe_loss
        )

        return combined_loss

    def _sharpe_ratio_metric(self, y_true, y_pred):
        """Custom metric to track Sharpe ratio during training"""
        predicted_actions = tf.argmax(y_pred, axis=1)
        action_returns = tf.cast(predicted_actions, tf.float32) - 1.0

        mean_return = tf.reduce_mean(action_returns)
        std_return = tf.math.reduce_std(action_returns)
        sharpe_ratio = mean_return / (tf.maximum(std_return, 1e-8))

        return sharpe_ratio

    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for BiLSTM training with proper sequence formatting

        Args:
            X: Feature matrix (samples, features)
            y: Labels (samples,) with values 0, 1, 2

        Returns:
            Tuple of (X_train, X_val, y_train, y_val) in proper format
        """

        self.logger.info("Preparing data for BiLSTM training...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences for LSTM
        X_sequences, y_sequences = self._create_sequences(X_scaled, y)

        # Convert labels to categorical (one-hot encoding)
        y_categorical = keras.utils.to_categorical(y_sequences, num_classes=self.config.num_classes)

        # Time-aware train/validation split
        split_index = int(len(X_sequences) * (1 - self.config.validation_split))

        X_train = X_sequences[:split_index]
        X_val = X_sequences[split_index:]
        y_train = y_categorical[:split_index]
        y_val = y_categorical[split_index:]

        self.logger.info(f"Training set: {X_train.shape[0]} sequences")
        self.logger.info(f"Validation set: {X_val.shape[0]} sequences")
        self.logger.info(f"Sequence shape: ({self.config.sequence_length}, {self.config.n_features})")

        return X_train, X_val, y_train, y_val

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""

        sequences_X = []
        sequences_y = []

        for i in range(self.config.sequence_length, len(X)):
            # Create sequence of past 60 time steps
            sequences_X.append(X[i-self.config.sequence_length:i])
            sequences_y.append(y[i])

        return np.array(sequences_X), np.array(sequences_y)

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the BiLSTM model with early stopping based on Sharpe ratio

        Args:
            X: Feature matrix
            y: Labels (0, 1, 2 for Sell, Hold, Buy)

        Returns:
            Training results and performance metrics
        """

        if self.model is None:
            self.create_model()

        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data(X, y)

        # Define callbacks
        callbacks_list = [
            EarlyStopping(
                monitor=self.config.early_stopping_monitor,
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                mode='max',  # Maximize Sharpe ratio
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath='models/best_ternary_bilstm.h5',
                monitor=self.config.early_stopping_monitor,
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]

        # Train model
        self.logger.info("Starting BiLSTM training with Sharpe ratio optimization...")
        start_time = time.time()

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )

        training_time = time.time() - start_time
        self.training_history = history

        # Evaluate final performance
        val_loss, val_accuracy, val_sharpe = self.model.evaluate(X_val, y_val, verbose=0)

        # Generate predictions for detailed analysis
        y_pred_proba = self.model.predict(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_val, axis=1)

        # Calculate detailed metrics
        accuracy = accuracy_score(y_true, y_pred)
        classification_rep = classification_report(y_true, y_pred, output_dict=True)

        # Check if Sharpe ratio target is met
        sharpe_target_met = val_sharpe >= self.config.target_sharpe_ratio

        results = {
            'training_time': training_time,
            'final_val_loss': val_loss,
            'final_val_accuracy': val_accuracy,
            'final_val_sharpe_ratio': val_sharpe,
            'sharpe_target_met': sharpe_target_met,
            'target_sharpe_ratio': self.config.target_sharpe_ratio,
            'classification_report': classification_rep,
            'class_accuracies': {
                'Sell_0': classification_rep.get('0', {}).get('precision', 0),
                'Hold_1': classification_rep.get('1', {}).get('precision', 0),
                'Buy_2': classification_rep.get('2', {}).get('precision', 0)
            },
            'model_architecture': 'BiLSTM (64→32→16) + Custom Sharpe Loss',
            'loss_function': 'Combined CrossEntropy + Sharpe Optimization'
        }

        # Log results
        self.logger.info("=== TRAINING COMPLETED ===")
        self.logger.info(f"Training Time: {training_time:.2f} seconds")
        self.logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        self.logger.info(f"Validation Sharpe Ratio: {val_sharpe:.4f}")
        self.logger.info(f"Sharpe Target Met: {'✅' if sharpe_target_met else '❌'} (target: {self.config.target_sharpe_ratio})")

        return results

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence scores

        Returns:
            Tuple of (predictions, probabilities)
            predictions: Class predictions (0, 1, 2)
            probabilities: Class probabilities (shape: samples, 3)
        """

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Scale and create sequences
        X_scaled = self.scaler.transform(X)
        X_sequences, _ = self._create_sequences(X_scaled, np.zeros(len(X)))

        # Generate predictions
        probabilities = self.model.predict(X_sequences)
        predictions = np.argmax(probabilities, axis=1)

        return predictions, probabilities

    def save_model(self, filepath: str):
        """Save the trained model and scaler"""

        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        # Save model
        self.model.save(f"{filepath}_model.h5")

        # Save scaler
        import joblib
        joblib.dump(self.scaler, f"{filepath}_scaler.joblib")

        # Save config
        import json
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)

        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model and scaler"""

        # Load model
        self.model = keras.models.load_model(
            f"{filepath}_model.h5",
            custom_objects={
                'combined_loss_function': self._combined_loss_function,
                'sharpe_ratio_metric': self._sharpe_ratio_metric
            }
        )

        # Load scaler
        import joblib
        self.scaler = joblib.load(f"{filepath}_scaler.joblib")

        # Load config
        import json
        with open(f"{filepath}_config.json", 'r') as f:
            config_dict = json.load(f)
            self.config = TernaryLSTMConfig(**config_dict)

        self.logger.info(f"Model loaded from {filepath}")

    def plot_training_history(self, save_path: str = None):
        """Plot training history including Sharpe ratio evolution"""

        if self.training_history is None:
            raise ValueError("No training history available")

        history = self.training_history.history

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('BiLSTM Training History with Sharpe Ratio Optimization', fontsize=16)

        # Loss
        axes[0, 0].plot(history['loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Combined Loss (CrossEntropy + Sharpe)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 1].plot(history['accuracy'], label='Training Accuracy', color='blue')
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Classification Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Sharpe Ratio
        if 'sharpe_ratio_metric' in history:
            axes[1, 0].plot(history['sharpe_ratio_metric'], label='Training Sharpe', color='blue')
            axes[1, 0].plot(history['val_sharpe_ratio_metric'], label='Validation Sharpe', color='red')
            axes[1, 0].axhline(y=self.config.target_sharpe_ratio, color='green', linestyle='--', label=f'Target ({self.config.target_sharpe_ratio})')
            axes[1, 0].set_title('Sharpe Ratio Evolution')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate
        if 'lr' in history:
            axes[1, 1].plot(history['lr'], label='Learning Rate', color='purple')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved to {save_path}")

        plt.show()


def main():
    """Demonstration of Ternary BiLSTM Network"""

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    print("🚀 TERNARY BiLSTM NETWORK - PROJECT REQUIREMENTS IMPLEMENTATION")
    print("=" * 70)
    print("📋 Architecture Specifications:")
    print("   • BiLSTM layers: 64 → 32 units")
    print("   • Dense layer: 16 units")
    print("   • Output: 3 classes (Buy=2, Hold=1, Sell=0)")
    print("   • Custom loss: CrossEntropy + Sharpe ratio optimization")
    print("   • Target: >1.2 validation Sharpe ratio")
    print("=" * 70)

    # Create model configuration
    config = TernaryLSTMConfig(
        sequence_length=60,
        n_features=20,
        lstm_units=[64, 32],
        dense_units=16,
        target_sharpe_ratio=1.2
    )

    # Initialize network
    network = TernaryLSTMNetwork(config)

    # Create model
    model = network.create_model()

    print(f"\n✅ BiLSTM model created successfully!")
    print(f"📊 Model summary:")
    print(f"   • Total parameters: {model.count_params():,}")
    print(f"   • Architecture: BiLSTM + Custom Sharpe Loss")
    print(f"   • Input shape: (60, 20)")
    print(f"   • Output shape: (3,) - softmax probabilities")

    return network

if __name__ == "__main__":
    network = main()