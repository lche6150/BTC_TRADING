#!/usr/bin/env python3
"""
Retrain Trading System with Ternary Classification
Update the entire ML pipeline to use Buy=2, Hold=1, Sell=0 classification
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.database import TradingDatabase
from src.data.features import EnhancedFeatureEngine

def main():
    """Main retraining pipeline for ternary classification system"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    print("=" * 80)
    print("🔄 RETRAINING SYSTEM WITH TERNARY CLASSIFICATION")
    print("=" * 80)
    print("📋 New Strategy Requirements:")
    print("   • Ternary labels: Buy=2, Hold=1, Sell=0")
    print("   • 20 XGBoost-selected features (vs previous 13)")
    print("   • Multi-class classification with balanced distribution (30-40% each)")
    print("   • >35% validation accuracy requirement")
    print("=" * 80)

    try:
        # Step 1: Load Historical Data
        logger.info("📊 Loading historical data...")
        db = TradingDatabase()

        # Get data before the gap test period (before Aug 1, 2025)
        df = db.get_ohlcv_data(limit=35000, end_timestamp=1722448800)  # July 31, 2025

        if df.empty:
            raise ValueError("No historical data found")

        logger.info(f"✅ Loaded {len(df)} historical records")

        # Step 2: Generate Enhanced Features (45+)
        logger.info("🔧 Generating enhanced feature set...")
        feature_engine = EnhancedFeatureEngine()
        features_df = feature_engine.engineer_features(df)

        logger.info(f"✅ Generated {features_df.shape[1]} features")
        print(f"   Features include: {list(features_df.columns[:10])}...")

        # Step 3: Create Ternary Labels using Triple Barrier Method
        logger.info("🎯 Creating ternary classification labels...")

        # Calculate volatility-adjusted ternary labels
        features_df['returns'] = df['close'].pct_change()
        features_df['volatility'] = features_df['returns'].rolling(96).std()  # 24h volatility

        labels = []
        for i in range(len(features_df) - 6):  # Look ahead 6 periods (90 min)
            if pd.isna(features_df['volatility'].iloc[i]):
                labels.append(np.nan)
                continue

            current_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+7]  # Next 6 periods
            volatility = features_df['volatility'].iloc[i]

            # Volatility-adjusted thresholds
            buy_threshold = volatility * 1.5  # 1.5x volatility for buy signal
            sell_threshold = volatility * 1.5  # 1.5x volatility for sell signal

            # Calculate maximum movement in either direction
            max_return = (future_prices.max() - current_price) / current_price
            min_return = (future_prices.min() - current_price) / current_price

            # Ternary classification
            if max_return >= buy_threshold:
                labels.append(2)  # Buy
            elif abs(min_return) >= sell_threshold:
                labels.append(0)  # Sell
            else:
                labels.append(1)  # Hold

        # Pad with NaN for remaining samples
        labels.extend([np.nan] * 6)
        features_df['labels'] = labels

        # Remove NaN labels
        valid_mask = ~pd.isna(features_df['labels'])
        features_clean = features_df[valid_mask].copy()

        logger.info(f"✅ Ternary labels created")

        # Check class distribution
        class_counts = features_clean['labels'].value_counts().sort_index()
        class_props = class_counts / len(features_clean)
        logger.info(f"   Class distribution:")
        for label, count in class_counts.items():
            prop = class_props[label]
            class_name = {0: 'Sell', 1: 'Hold', 2: 'Buy'}[int(label)]
            logger.info(f"     {class_name} ({int(label)}): {count} samples ({prop:.1%})")

        # Step 4: XGBoost Feature Selection (simplified version)
        logger.info("🚀 Running XGBoost feature selection...")

        # Prepare data
        feature_cols = [col for col in features_clean.columns if col not in ['labels', 'returns', 'volatility']]
        X = features_clean[feature_cols].fillna(0)
        y = features_clean['labels'].astype(int)

        # Simple XGBoost feature selection
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train XGBoost
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_scaled, y_train)

        # Get feature importance
        importance_scores = model.feature_importances_
        feature_importance = dict(zip(feature_cols, importance_scores))

        # Select top 20 features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:20]]

        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = (y_pred == y_test).mean()

        logger.info("✅ XGBoost feature selection completed")
        logger.info(f"   Selected {len(selected_features)} features")
        logger.info(f"   Test accuracy: {accuracy:.3f}")

        # Step 5: Save Results
        logger.info("💾 Saving retrained models and results...")

        # Create results directory
        results_dir = Path('../results/ternary_retraining')
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save selected features
        import json
        with open(results_dir / 'selected_features.json', 'w') as f:
            json.dump(selected_features, f, indent=2)

        # Save labels
        features_clean[['labels']].to_csv(results_dir / 'ternary_labels.csv')

        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(df),
                'training_samples': len(features_clean),
                'features_generated': len(feature_cols),
                'features_selected': len(selected_features)
            },
            'label_distribution': {
                'Sell_0': int(class_counts.get(0, 0)),
                'Hold_1': int(class_counts.get(1, 0)),
                'Buy_2': int(class_counts.get(2, 0))
            },
            'model_performance': {
                'test_accuracy': float(accuracy)
            },
            'selected_features': selected_features,
            'requirements_check': {
                'accuracy_requirement_met': accuracy > 0.35,
                'feature_count_correct': len(selected_features) == 20
            }
        }

        with open(results_dir / 'retraining_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✅ Results saved to {results_dir}")

        # Step 6: Print Final Summary
        print("\n" + "=" * 80)
        print("🎉 TERNARY RETRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("📊 FINAL RESULTS:")
        print(f"   📈 Model Accuracy: {accuracy:.3f} ({'✅' if accuracy > 0.35 else '❌'} >35% requirement)")
        print(f"   🎯 Selected Features: {len(selected_features)} ({'✅' if len(selected_features) == 20 else '❌'} 20 required)")

        print(f"\n📋 CLASS DISTRIBUTION:")
        for label, count in class_counts.items():
            prop = count / len(features_clean)
            class_name = {0: 'Sell', 1: 'Hold', 2: 'Buy'}[int(label)]
            status = '✅' if 0.30 <= prop <= 0.40 else '⚠️'
            print(f"   {status} {class_name} ({int(label)}): {count:,} samples ({prop:.1%})")

        print(f"\n🔝 TOP 10 SELECTED FEATURES:")
        for i, feature in enumerate(selected_features[:10], 1):
            importance = feature_importance.get(feature, 0)
            print(f"   {i:2d}. {feature:<30} (importance: {importance:.4f})")

        print("\n🚀 NEXT STEPS:")
        print("   1. Update LSTM architecture with BiLSTM + custom Sharpe loss")
        print("   2. Retrain LSTM with new 20 features and ternary labels")
        print("   3. Implement probability threshold optimization")
        print("   4. Update trading system with new signal generation")
        print("=" * 80)

        return True

    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        print(f"\n❌ RETRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)