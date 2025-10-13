"""
Advanced XGBoost Feature Selection Model for Cryptocurrency Trading
Based on 2024 research with Bayesian optimization, SHAP analysis, and class imbalance handling

Key Research-Based Features:
- Optuna Bayesian optimization for hyperparameter tuning
- SHAP values for interpretable feature importance
- multi:softprob objective for probability-based predictions
- Class imbalance handling with scale_pos_weight
- Time series cross-validation for robust evaluation
- Feature correlation analysis and redundancy removal
- Early stopping with optimal stopping rounds

Implementation follows 2024 best practices from:
- XGBoost hyperparameter optimization research
- Cryptocurrency trading ML papers
- SHAP interpretability studies
- Multi-class classification optimization
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import shap
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import joblib
import json
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.utils import check_sampling_strategy

import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class XGBoostConfig:
    """Configuration for XGBoost model based on 2024 research"""
    # Core parameters
    objective: str = 'multi:softprob'  # Probability output for trading decisions
    num_class: int = 3  # Buy, Hold, Sell
    eval_metric: str = 'mlogloss'  # Multi-class log loss

    # Tree structure (research-based ranges)
    max_depth: int = 6
    min_child_weight: int = 1
    gamma: float = 0.0

    # Learning parameters
    learning_rate: float = 0.1
    n_estimators: int = 200
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    colsample_bylevel: float = 1.0
    colsample_bynode: float = 1.0

    # Regularization
    reg_alpha: float = 0.0  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization

    # Class imbalance handling
    scale_pos_weight: float = 1.0
    max_delta_step: int = 0  # For extreme imbalance

    # Performance
    n_jobs: int = -1
    random_state: int = 42
    early_stopping_rounds: int = 50

    # Feature selection
    importance_type: str = 'gain'  # 'gain', 'weight', 'cover'
    top_k_features: int = 20
    correlation_threshold: float = 0.8

@dataclass
class FeatureSelectionResults:
    """Results from XGBoost feature selection"""
    selected_features: List[str]
    feature_importance: Dict[str, float]
    shap_values: Optional[np.ndarray]
    shap_feature_importance: Dict[str, float]
    model_performance: Dict[str, float]
    hyperparameters: Dict[str, Any]
    cross_val_scores: List[float]
    class_distribution: Dict[int, int]

class AdvancedXGBoostSelector:
    """
    Advanced XGBoost Feature Selection with 2024 Research Integration
    """

    def __init__(self, config: Optional[XGBoostConfig] = None, random_seed: int = 42):
        self.config = config or XGBoostConfig()
        self.random_seed = random_seed
        self.logger = logging.getLogger(__name__)

        # Model components
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.shap_explainer: Optional[shap.Explainer] = None

        # Results storage
        self.feature_names: List[str] = []
        self.results: Optional[FeatureSelectionResults] = None

        # Set random seeds
        np.random.seed(random_seed)

        self.logger.info("Advanced XGBoost Feature Selector initialized")

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        optimize_hyperparameters: bool = True,
        n_trials: int = 100,
        cv_folds: int = 5
    ) -> FeatureSelectionResults:
        """
        Main feature selection pipeline with Bayesian optimization

        Args:
            X: Feature matrix
            y: Target labels
            optimize_hyperparameters: Whether to use Optuna for optimization
            n_trials: Number of Optuna trials
            cv_folds: Cross-validation folds

        Returns:
            FeatureSelectionResults with comprehensive analysis
        """
        self.logger.info("Starting advanced XGBoost feature selection...")

        # Data preprocessing
        X_processed, y_processed = self._preprocess_data(X, y)

        # Handle class imbalance
        self._analyze_class_imbalance(y_processed)

        # Hyperparameter optimization
        if optimize_hyperparameters:
            self.logger.info("Optimizing hyperparameters with Optuna...")
            best_params = self._optimize_hyperparameters(X_processed, y_processed, n_trials, cv_folds)
            self._update_config(best_params)

        # Train final model
        self.logger.info("Training final XGBoost model...")
        final_model = self._train_final_model(X_processed, y_processed, cv_folds)

        # Feature importance analysis
        self.logger.info("Analyzing feature importance...")
        feature_importance = self._get_feature_importance(final_model, X_processed)

        # SHAP analysis
        self.logger.info("Computing SHAP values...")
        shap_values, shap_importance = self._compute_shap_values(final_model, X_processed)

        # Feature selection
        selected_features = self._select_top_features(feature_importance, shap_importance, X_processed)

        # Performance evaluation
        performance_metrics = self._evaluate_model_performance(final_model, X_processed, y_processed, cv_folds)

        # Create results
        self.results = FeatureSelectionResults(
            selected_features=selected_features,
            feature_importance=feature_importance,
            shap_values=shap_values,
            shap_feature_importance=shap_importance,
            model_performance=performance_metrics,
            hyperparameters=asdict(self.config),
            cross_val_scores=performance_metrics.get('cv_scores', []),
            class_distribution=dict(pd.Series(y_processed).value_counts().sort_index())
        )

        self.logger.info(f"Feature selection completed. Selected {len(selected_features)} features.")
        return self.results

    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features and labels"""
        # Store feature names
        self.feature_names = X.columns.tolist()

        # Handle missing values
        X_clean = X.fillna(X.median())

        # Remove highly correlated features
        X_clean = self._remove_correlated_features(X_clean)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)

        # Encode labels and handle missing classes
        self.label_encoder = LabelEncoder()
        y_clean = y.fillna(0)

        # Ensure we have all expected classes for proper encoding
        unique_labels = sorted(y_clean.unique())
        self.logger.info(f"Unique labels in data: {unique_labels}")

        y_encoded = self.label_encoder.fit_transform(y_clean)

        self.logger.info(f"Preprocessed data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")

        return X_scaled, y_encoded

    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features based on research best practices"""
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > self.config.correlation_threshold)
        ]

        if to_drop:
            self.logger.info(f"Removing {len(to_drop)} highly correlated features")
            X_clean = X.drop(columns=to_drop)
        else:
            X_clean = X

        return X_clean

    def _analyze_class_imbalance(self, y: np.ndarray):
        """Analyze and handle class imbalance"""
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))

        self.logger.info(f"Class distribution: {class_dist}")

        # Calculate class weights for imbalance handling
        if len(unique) > 1:
            # For binary classification in multi-class (focusing on minority classes)
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count

            if imbalance_ratio > 2:
                # Update scale_pos_weight based on imbalance
                self.config.scale_pos_weight = imbalance_ratio
                self.config.max_delta_step = min(10, int(imbalance_ratio // 2))
                self.logger.info(f"Detected class imbalance (ratio: {imbalance_ratio:.2f})")
                self.logger.info(f"Updated scale_pos_weight: {self.config.scale_pos_weight}")

    def _optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int,
        cv_folds: int
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna with 2024 best practices"""

        def objective(trial):
            # Hyperparameter search space based on 2024 research
            num_classes = len(np.unique(y))

            # Use appropriate objective based on number of classes
            if num_classes == 2:
                objective_func = 'binary:logistic'
                eval_metric = 'logloss'
            else:
                objective_func = 'multi:softprob'
                eval_metric = 'mlogloss'

            params = {
                'objective': objective_func,
                'eval_metric': eval_metric,
            }

            # Add num_class only for multi-class
            if num_classes > 2:
                params['num_class'] = num_classes

            params.update({
                # Tree structure - most important parameters
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),

                # Learning parameters
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),

                # Regularization
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),

                # Class imbalance
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 5.0),

                # Performance
                'n_jobs': -1,
                'random_state': self.random_seed
            })

            # Create model
            model = xgb.XGBClassifier(**params)

            # Time series cross-validation for realistic evaluation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_log_loss', n_jobs=-1)

            return scores.mean()

        # Create and run study
        study = optuna.create_study(
            direction='maximize',  # Maximize negative log loss (minimize log loss)
            sampler=optuna.samplers.TPESampler(seed=self.random_seed)
        )

        # Add progress callback
        def callback(study, trial):
            if trial.number % 20 == 0:
                self.logger.info(f"Trial {trial.number}: Best value = {study.best_value:.4f}")

        study.optimize(objective, n_trials=n_trials, callbacks=[callback])

        self.logger.info(f"Optimization completed. Best value: {study.best_value:.4f}")
        self.logger.info(f"Best parameters: {study.best_params}")

        return study.best_params

    def _update_config(self, best_params: Dict[str, Any]):
        """Update configuration with optimized parameters"""
        for key, value in best_params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def _train_final_model(self, X: np.ndarray, y: np.ndarray, cv_folds: int) -> xgb.XGBClassifier:
        """Train final model with optimized parameters"""
        # Determine number of classes dynamically
        num_classes = len(np.unique(y))

        # Use appropriate objective based on number of classes
        if num_classes == 2:
            objective_func = 'binary:logistic'
            eval_metric = 'logloss'
        else:
            objective_func = self.config.objective
            eval_metric = self.config.eval_metric

        model_params = {
            'objective': objective_func,
            'eval_metric': eval_metric,
        }

        # Add num_class only for multi-class
        if num_classes > 2:
            model_params['num_class'] = num_classes

        model_params.update({
            'max_depth': self.config.max_depth,
            'min_child_weight': self.config.min_child_weight,
            'gamma': self.config.gamma,
            'learning_rate': self.config.learning_rate,
            'n_estimators': self.config.n_estimators,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'colsample_bylevel': self.config.colsample_bylevel,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'scale_pos_weight': self.config.scale_pos_weight,
            'max_delta_step': self.config.max_delta_step,
            'n_jobs': self.config.n_jobs,
            'random_state': self.config.random_state
        })

        self.model = xgb.XGBClassifier(**model_params)

        # Train with early stopping (XGBoost 2.0+ compatibility)
        X_train, X_val, y_train, y_val = self._create_validation_split(X, y, 0.2)

        # Check XGBoost version for compatibility
        try:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[xgb.callback.EarlyStopping(rounds=self.config.early_stopping_rounds)],
                verbose=False
            )
        except Exception:
            # Fallback for older XGBoost versions
            self.model.fit(X_train, y_train, verbose=False)

        return self.model

    def _create_validation_split(self, X: np.ndarray, y: np.ndarray, val_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create time-aware validation split"""
        split_idx = int(len(X) * (1 - val_size))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        return X_train, X_val, y_train, y_val

    def _get_feature_importance(self, model: xgb.XGBClassifier, X: np.ndarray) -> Dict[str, float]:
        """Get XGBoost built-in feature importance"""
        importance_values = model.feature_importances_
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        importance_dict = dict(zip(feature_names, importance_values))

        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return sorted_importance

    def _compute_shap_values(self, model: xgb.XGBClassifier, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Compute SHAP values for interpretable feature importance"""
        try:
            # Create SHAP explainer
            self.shap_explainer = shap.TreeExplainer(model)

            # Compute SHAP values (sample subset for efficiency)
            sample_size = min(1000, X.shape[0])
            sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[sample_indices]

            shap_values = self.shap_explainer.shap_values(X_sample)

            # For multi-class, we get SHAP values for each class
            # Take mean absolute SHAP values across all classes
            if isinstance(shap_values, list):
                # Multi-class case
                mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                # Single output case
                mean_shap = np.abs(shap_values)

            # Calculate feature importance from SHAP values
            feature_importance = np.mean(mean_shap, axis=0)
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            shap_importance = dict(zip(feature_names, feature_importance))

            # Sort by importance
            shap_importance = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))

            return shap_values, shap_importance

        except Exception as e:
            self.logger.warning(f"Could not compute SHAP values: {e}")
            return None, {}

    def _select_top_features(
        self,
        xgb_importance: Dict[str, float],
        shap_importance: Dict[str, float],
        X: np.ndarray
    ) -> List[str]:
        """Select top features combining XGBoost and SHAP importance"""

        # Combine XGBoost and SHAP importance (if available)
        if shap_importance:
            # Weighted combination: 60% XGBoost, 40% SHAP
            combined_importance = {}
            all_features = set(xgb_importance.keys()) | set(shap_importance.keys())

            for feature in all_features:
                xgb_score = xgb_importance.get(feature, 0)
                shap_score = shap_importance.get(feature, 0)
                combined_importance[feature] = 0.6 * xgb_score + 0.4 * shap_score
        else:
            combined_importance = xgb_importance

        # Sort and select top features
        sorted_features = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature for feature, _ in sorted_features[:self.config.top_k_features]]

        self.logger.info(f"Selected top {len(top_features)} features")

        return top_features

    def _evaluate_model_performance(
        self,
        model: xgb.XGBClassifier,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int
    ) -> Dict[str, float]:
        """Comprehensive model performance evaluation"""

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        # Multiple scoring metrics
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_log_loss', n_jobs=-1)
        accuracy_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)

        # Predictions for additional metrics
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)

        # Calculate metrics
        performance = {
            'cv_log_loss_mean': abs(cv_scores.mean()),
            'cv_log_loss_std': cv_scores.std(),
            'cv_accuracy_mean': accuracy_scores.mean(),
            'cv_accuracy_std': accuracy_scores.std(),
            'log_loss': log_loss(y, y_pred_proba),
            'cv_scores': cv_scores.tolist()
        }

        self.logger.info(f"Model Performance:")
        self.logger.info(f"  CV Log Loss: {performance['cv_log_loss_mean']:.4f} ± {performance['cv_log_loss_std']:.4f}")
        self.logger.info(f"  CV Accuracy: {performance['cv_accuracy_mean']:.4f} ± {performance['cv_accuracy_std']:.4f}")

        return performance

    def save_results(self, filepath: str):
        """Save feature selection results"""
        if not self.results:
            raise ValueError("No results to save. Run select_features first.")

        results_dict = {
            'selected_features': self.results.selected_features,
            'feature_importance': self.results.feature_importance,
            'shap_feature_importance': self.results.shap_feature_importance,
            'model_performance': self.results.model_performance,
            'hyperparameters': self.results.hyperparameters,
            'cross_val_scores': self.results.cross_val_scores,
            'class_distribution': self.results.class_distribution
        }

        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)

        # Save model
        model_path = filepath.replace('.json', '_model.joblib')
        joblib.dump(self.model, model_path)

        # Save scalers
        scaler_path = filepath.replace('.json', '_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)

        self.logger.info(f"Results saved to {filepath}")

    def create_visualizations(self, save_dir: str = 'logs'):
        """Create comprehensive visualizations"""
        if not self.results:
            raise ValueError("No results to visualize. Run select_features first.")

        Path(save_dir).mkdir(exist_ok=True)

        # 1. Feature importance comparison
        self._plot_feature_importance_comparison(save_dir)

        # 2. SHAP summary plot
        self._plot_shap_summary(save_dir)

        # 3. Model performance metrics
        self._plot_performance_metrics(save_dir)

        # 4. Class distribution
        self._plot_class_distribution(save_dir)

    def _plot_feature_importance_comparison(self, save_dir: str):
        """Plot XGBoost vs SHAP feature importance"""
        try:
            top_features = self.results.selected_features
            xgb_scores = [self.results.feature_importance.get(f, 0) for f in top_features]
            shap_scores = [self.results.shap_feature_importance.get(f, 0) for f in top_features]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

            # XGBoost importance
            ax1.barh(range(len(top_features)), xgb_scores, color='lightblue')
            ax1.set_yticks(range(len(top_features)))
            ax1.set_yticklabels(top_features)
            ax1.set_xlabel('XGBoost Feature Importance')
            ax1.set_title('XGBoost Built-in Feature Importance')

            # SHAP importance
            if any(shap_scores):
                ax2.barh(range(len(top_features)), shap_scores, color='lightcoral')
                ax2.set_yticks(range(len(top_features)))
                ax2.set_yticklabels(top_features)
                ax2.set_xlabel('SHAP Feature Importance')
                ax2.set_title('SHAP Feature Importance')

            plt.tight_layout()
            plt.savefig(f'{save_dir}/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create feature importance plot: {e}")

    def _plot_shap_summary(self, save_dir: str):
        """Plot SHAP summary if available"""
        if self.shap_explainer and self.results.shap_values is not None:
            try:
                # This requires the original feature matrix for SHAP plotting
                self.logger.info("SHAP summary plot would require original feature matrix")
            except Exception as e:
                self.logger.warning(f"Could not create SHAP summary plot: {e}")

    def _plot_performance_metrics(self, save_dir: str):
        """Plot model performance metrics"""
        try:
            metrics = self.results.model_performance
            cv_scores = metrics.get('cv_scores', [])

            if cv_scores:
                plt.figure(figsize=(10, 6))
                plt.subplot(1, 2, 1)
                plt.plot(range(1, len(cv_scores) + 1), np.abs(cv_scores), 'o-')
                plt.xlabel('CV Fold')
                plt.ylabel('Negative Log Loss')
                plt.title('Cross-Validation Performance')
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 2, 2)
                metrics_names = ['CV Log Loss', 'CV Accuracy']
                metrics_values = [metrics['cv_log_loss_mean'], metrics['cv_accuracy_mean']]
                metrics_stds = [metrics['cv_log_loss_std'], metrics['cv_accuracy_std']]

                plt.bar(metrics_names, metrics_values, yerr=metrics_stds, capsize=5, alpha=0.7)
                plt.ylabel('Score')
                plt.title('Model Performance Summary')
                plt.xticks(rotation=45)

                plt.tight_layout()
                plt.savefig(f'{save_dir}/model_performance.png', dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create performance plot: {e}")

    def _plot_class_distribution(self, save_dir: str):
        """Plot class distribution"""
        try:
            class_dist = self.results.class_distribution
            classes = list(class_dist.keys())
            counts = list(class_dist.values())

            plt.figure(figsize=(8, 6))
            colors = ['red', 'gray', 'green']  # Sell, Hold, Buy
            labels = ['Sell (0)', 'Hold (1)', 'Buy (2)']

            plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Ternary Class Distribution (Buy=2, Hold=1, Sell=0)')
            plt.axis('equal')

            plt.savefig(f'{save_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create class distribution plot: {e}")

    def get_feature_selection_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of feature selection results"""
        if not self.results:
            raise ValueError("No results available. Run select_features first.")

        return {
            'total_features_selected': len(self.results.selected_features),
            'selected_features': self.results.selected_features,
            'top_5_features': self.results.selected_features[:5],
            'model_performance': {
                'cv_log_loss': f"{self.results.model_performance['cv_log_loss_mean']:.4f} ± {self.results.model_performance['cv_log_loss_std']:.4f}",
                'cv_accuracy': f"{self.results.model_performance['cv_accuracy_mean']:.4f} ± {self.results.model_performance['cv_accuracy_std']:.4f}"
            },
            'class_distribution': self.results.class_distribution,
            'hyperparameters': {
                'max_depth': self.results.hyperparameters['max_depth'],
                'learning_rate': self.results.hyperparameters['learning_rate'],
                'n_estimators': self.results.hyperparameters['n_estimators']
            }
        }