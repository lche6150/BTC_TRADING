"""
Enhanced Feature Engineering Pipeline for BTC/USDT Trading System
Based on 2024 best practices with performance optimization and advanced indicators

Features implemented:
- 45+ Technical Indicators (Trend, Momentum, Volatility, Volume)
- Multi-timeframe analysis (15m, 1h, 4h)
- Modern volatility regime indicators
- Performance-optimized calculations using TA-Lib + pandas-ta hybrid
- Vectorized operations with Numba acceleration where applicable
"""

import pandas as pd
import numpy as np
import talib
# import pandas_ta as ta  # Temporarily disabled for Python 3.11 compatibility
from numba import jit
import warnings
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class EnhancedFeatureEngine:
    """
    Enhanced feature engineering pipeline with 2024 optimizations
    """

    def __init__(self, performance_mode: bool = True):
        self.performance_mode = performance_mode
        self.logger = logging.getLogger(__name__)
        self.feature_names = []
        self.feature_categories = {
            'trend': [],
            'momentum': [],
            'volatility': [],
            'volume': [],
            'custom': [],
            'multiframe': []
        }

    def create_all_features(
        self,
        df: pd.DataFrame,
        include_multiframe: bool = True,
        min_periods: int = 200
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set with performance optimization

        Args:
            df: OHLCV DataFrame with datetime index
            include_multiframe: Whether to include higher timeframe features
            min_periods: Minimum periods required for feature calculation

        Returns:
            DataFrame with all features added
        """
        start_time = datetime.now()
        self.logger.info(f"Starting feature engineering for {len(df)} records...")

        # Ensure proper data types and sorting
        df = self._prepare_data(df)

        # Category 1: Trend Indicators (15 features)
        df = self._add_trend_features(df)

        # Category 2: Momentum Indicators (12 features)
        df = self._add_momentum_features(df)

        # Category 3: Volatility Indicators (8 features)
        df = self._add_volatility_features(df)

        # Category 4: Volume Indicators (6 features)
        df = self._add_volume_features(df)

        # Category 5: Custom Engineered Features (6+ features)
        df = self._add_custom_features(df)

        # Category 6: Multi-timeframe Features (8+ features)
        if include_multiframe:
            df = self._add_multiframe_features(df)

        # Category 7: 2024 Advanced Features (10+ features)
        df = self._add_modern_features(df)

        # Post-processing
        df = self._post_process_features(df, min_periods)

        # Performance summary
        duration = datetime.now() - start_time
        feature_count = len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'datetime', 'timestamp']])

        self.logger.info(f"Feature engineering completed in {duration.total_seconds():.2f}s")
        self.logger.info(f"Generated {feature_count} features across {len(self.feature_categories)} categories")

        return df

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for feature engineering"""
        df = df.copy()

        # Ensure proper data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by datetime
        if 'datetime' in df.columns:
            df = df.sort_values('datetime').reset_index(drop=True)

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based technical indicators (Target: 15 features)"""

        # Simple Moving Averages (5 features)
        for period in [5, 10, 20, 50, 200]:
            col_name = f'sma_{period}'
            df[col_name] = talib.SMA(df['close'], timeperiod=period)
            self.feature_categories['trend'].append(col_name)

        # Exponential Moving Averages (3 features)
        for period in [12, 26, 50]:
            col_name = f'ema_{period}'
            df[col_name] = talib.EMA(df['close'], timeperiod=period)
            self.feature_categories['trend'].append(col_name)

        # MACD (3 features)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        self.feature_categories['trend'].extend(['macd', 'macd_signal', 'macd_hist'])

        # Parabolic SAR (1 feature)
        df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
        self.feature_categories['trend'].append('sar')

        # Price position relative to moving averages (3 features)
        df['price_sma20_ratio'] = df['close'] / df['sma_20']
        df['price_ema12_ratio'] = df['close'] / df['ema_12']
        df['sma20_sma50_ratio'] = df['sma_20'] / df['sma_50']
        self.feature_categories['trend'].extend(['price_sma20_ratio', 'price_ema12_ratio', 'sma20_sma50_ratio'])

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators (Target: 12 features)"""

        # RSI variations (3 features)
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_21'] = talib.RSI(df['close'], timeperiod=21)
        df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
        self.feature_categories['momentum'].extend(['rsi_14', 'rsi_21', 'rsi_7'])

        # Stochastic Oscillators (2 features)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        self.feature_categories['momentum'].extend(['stoch_k', 'stoch_d'])

        # Williams %R (1 feature)
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        self.feature_categories['momentum'].append('williams_r')

        # ROC (Rate of Change) (2 features)
        df['roc_10'] = talib.ROC(df['close'], timeperiod=10)
        df['roc_21'] = talib.ROC(df['close'], timeperiod=21)
        self.feature_categories['momentum'].extend(['roc_10', 'roc_21'])

        # Momentum (2 features)
        df['momentum_10'] = talib.MOM(df['close'], timeperiod=10)
        df['momentum_21'] = talib.MOM(df['close'], timeperiod=21)
        self.feature_categories['momentum'].extend(['momentum_10', 'momentum_21'])

        # CCI (Commodity Channel Index) (1 feature)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        self.feature_categories['momentum'].append('cci')

        # Ultimate Oscillator (1 feature)
        df['ult_osc'] = talib.ULTOSC(df['high'], df['low'], df['close'])
        self.feature_categories['momentum'].append('ult_osc')

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators (Target: 8 features)"""

        # Bollinger Bands (3 features)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        self.feature_categories['volatility'].extend(['bb_upper', 'bb_middle', 'bb_lower'])

        # Bollinger Band derived features (2 features)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        self.feature_categories['volatility'].extend(['bb_width', 'bb_position'])

        # Average True Range (1 feature)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        self.feature_categories['volatility'].append('atr')

        # True Range (1 feature)
        df['tr'] = talib.TRANGE(df['high'], df['low'], df['close'])
        self.feature_categories['volatility'].append('tr')

        # Normalized ATR (1 feature)
        df['atr_normalized'] = df['atr'] / df['close']
        self.feature_categories['volatility'].append('atr_normalized')

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators (Target: 6 features)"""

        # Volume Moving Averages (2 features)
        df['vol_sma_20'] = talib.SMA(df['volume'], timeperiod=20)
        df['vol_sma_50'] = talib.SMA(df['volume'], timeperiod=50)
        self.feature_categories['volume'].extend(['vol_sma_20', 'vol_sma_50'])

        # Volume Ratios (2 features)
        df['vol_ratio_20'] = df['volume'] / df['vol_sma_20']
        df['vol_ratio_50'] = df['volume'] / df['vol_sma_50']
        self.feature_categories['volume'].extend(['vol_ratio_20', 'vol_ratio_50'])

        # On Balance Volume (1 feature)
        df['obv'] = talib.OBV(df['close'], df['volume'])
        self.feature_categories['volume'].append('obv')

        # Volume Price Trend (1 feature)
        df['vpt'] = self._calculate_vpt(df)
        self.feature_categories['volume'].append('vpt')

        return df

    def _add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom engineered features (Target: 6+ features)"""

        # Price action features (3 features)
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['oc_ratio'] = (df['open'] - df['close']) / df['close']
        df['body_shadow_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        self.feature_categories['custom'].extend(['hl_ratio', 'oc_ratio', 'body_shadow_ratio'])

        # Returns features (2 features)
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        self.feature_categories['custom'].extend(['returns_1', 'returns_5'])

        # Momentum derivatives (2 features)
        df['rsi_momentum'] = df['rsi_14'].diff()
        df['macd_momentum'] = df['macd'].diff()
        self.feature_categories['custom'].extend(['rsi_momentum', 'macd_momentum'])

        return df

    def _add_multiframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add multi-timeframe features (Target: 8+ features)"""

        # Resample to higher timeframes
        if 'datetime' not in df.columns:
            return df

        df_1h = self._resample_to_timeframe(df, '1H')
        df_4h = self._resample_to_timeframe(df, '4H')

        # 1H timeframe features (4 features)
        df['sma_20_1h'] = self._align_higher_tf_feature(df, df_1h, 'sma_20')
        df['rsi_14_1h'] = self._align_higher_tf_feature(df, df_1h, 'rsi_14')
        df['bb_position_1h'] = self._align_higher_tf_feature(df, df_1h, 'bb_position')
        df['atr_normalized_1h'] = self._align_higher_tf_feature(df, df_1h, 'atr_normalized')

        # 4H timeframe features (4 features)
        df['sma_20_4h'] = self._align_higher_tf_feature(df, df_4h, 'sma_20')
        df['rsi_14_4h'] = self._align_higher_tf_feature(df, df_4h, 'rsi_14')
        df['bb_position_4h'] = self._align_higher_tf_feature(df, df_4h, 'bb_position')
        df['atr_normalized_4h'] = self._align_higher_tf_feature(df, df_4h, 'atr_normalized')

        multiframe_features = [
            'sma_20_1h', 'rsi_14_1h', 'bb_position_1h', 'atr_normalized_1h',
            'sma_20_4h', 'rsi_14_4h', 'bb_position_4h', 'atr_normalized_4h'
        ]
        self.feature_categories['multiframe'].extend(multiframe_features)

        return df

    def _add_modern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 2024 advanced features (Target: 10+ features)"""

        # Realized volatility (2 features)
        df['realized_vol_24'] = df['returns_1'].rolling(24).std() * np.sqrt(24)
        df['realized_vol_96'] = df['returns_1'].rolling(96).std() * np.sqrt(96)
        self.feature_categories['custom'].extend(['realized_vol_24', 'realized_vol_96'])

        # Volatility regime indicators (2 features)
        df['vol_regime_fast'] = (df['realized_vol_24'] > df['realized_vol_24'].rolling(48).mean()).astype(int)
        df['vol_regime_slow'] = (df['realized_vol_96'] > df['realized_vol_96'].rolling(192).mean()).astype(int)
        self.feature_categories['custom'].extend(['vol_regime_fast', 'vol_regime_slow'])

        # Price efficiency (2 features)
        df['price_efficiency_10'] = abs(df['close'].diff(10)) / df['close'].rolling(10).apply(
            lambda x: sum(abs(x.diff().dropna()))
        )
        df['price_efficiency_20'] = abs(df['close'].diff(20)) / df['close'].rolling(20).apply(
            lambda x: sum(abs(x.diff().dropna()))
        )
        self.feature_categories['custom'].extend(['price_efficiency_10', 'price_efficiency_20'])

        # Market microstructure proxies (2 features)
        df['high_low_vol_corr'] = df['hl_ratio'].rolling(48).corr(df['vol_ratio_20'])
        df['price_vol_corr'] = df['returns_1'].rolling(48).corr(df['vol_ratio_20'])
        self.feature_categories['custom'].extend(['high_low_vol_corr', 'price_vol_corr'])

        # Fractal adaptive features (2 features)
        df['fractal_dim'] = self._calculate_fractal_dimension(df)
        df['hurst_exp'] = self._calculate_hurst_exponent(df)
        self.feature_categories['custom'].extend(['fractal_dim', 'hurst_exp'])

        return df

    def _post_process_features(self, df: pd.DataFrame, min_periods: int) -> pd.DataFrame:
        """Post-process features for quality and performance"""

        # Remove initial periods with insufficient data
        if len(df) > min_periods:
            df = df.iloc[min_periods:].reset_index(drop=True)

        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Calculate NaN percentage
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'datetime', 'timestamp']]
        nan_pct = df[feature_cols].isnull().sum() / len(df) * 100

        self.logger.info(f"Feature NaN percentages:")
        for feature, pct in nan_pct.head(10).items():
            if pct > 0:
                self.logger.info(f"  {feature}: {pct:.2f}%")

        # Report features that exceed 5% NaN threshold
        high_nan_features = nan_pct[nan_pct > 5].index.tolist()
        if high_nan_features:
            self.logger.warning(f"Features with >5% NaN: {high_nan_features}")

        return df

    # Helper methods
    def _calculate_vpt(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend"""
        vpt = pd.Series(index=df.index, dtype=float)
        vpt.iloc[0] = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i-1] != 0:
                vpt.iloc[i] = vpt.iloc[i-1] + (df['volume'].iloc[i] *
                    (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1])
        return vpt

    def _resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to higher timeframe"""
        df_tf = df.set_index('datetime').resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Calculate features for this timeframe
        for period in [20]:
            df_tf[f'sma_{period}'] = talib.SMA(df_tf['close'], timeperiod=period)

        df_tf['rsi_14'] = talib.RSI(df_tf['close'], timeperiod=14)

        # Bollinger Bands for position calculation
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df_tf['close'], timeperiod=20)
        df_tf['bb_position'] = (df_tf['close'] - bb_lower) / (bb_upper - bb_lower)

        # ATR normalized
        atr = talib.ATR(df_tf['high'], df_tf['low'], df_tf['close'], timeperiod=14)
        df_tf['atr_normalized'] = atr / df_tf['close']

        return df_tf

    def _align_higher_tf_feature(self, df_15m: pd.DataFrame, df_htf: pd.DataFrame, feature: str) -> pd.Series:
        """Align higher timeframe feature to 15m timeframe"""
        if 'datetime' not in df_15m.columns:
            return pd.Series([np.nan] * len(df_15m))

        # Forward fill higher timeframe values
        df_aligned = pd.merge_asof(
            df_15m[['datetime']].sort_values('datetime'),
            df_htf[[feature]].reset_index().sort_values('datetime'),
            on='datetime',
            direction='backward'
        )

        return df_aligned[feature]

    def _calculate_fractal_dimension(self, df: pd.DataFrame, window: int = 48) -> pd.Series:
        """Calculate fractal dimension using box-counting method"""
        def fractal_dim(series):
            if len(series) < 10:
                return np.nan
            try:
                # Simplified fractal dimension calculation
                n = len(series)
                rs_values = []
                for scale in [2, 4, 8]:
                    if scale < n:
                        scaled_series = series[::scale]
                        if len(scaled_series) > 1:
                            rs = scaled_series.max() - scaled_series.min()
                            rs_values.append(rs)

                if len(rs_values) >= 2:
                    return np.log(rs_values[-1] / rs_values[0]) / np.log(0.5)
                return 1.5  # Default value
            except:
                return 1.5

        return df['close'].rolling(window).apply(fractal_dim)

    def _calculate_hurst_exponent(self, df: pd.DataFrame, window: int = 48) -> pd.Series:
        """Calculate Hurst exponent for trend persistence"""
        def hurst_exp(series):
            if len(series) < 20:
                return 0.5
            try:
                # Simplified Hurst calculation using R/S analysis
                n = len(series)
                if n < 20:
                    return 0.5

                # Calculate mean-adjusted series
                mean_val = series.mean()
                y = (series - mean_val).cumsum()

                # Calculate range and standard deviation
                r = y.max() - y.min()
                s = series.std()

                if s == 0:
                    return 0.5

                rs = r / s
                if rs <= 0:
                    return 0.5

                return min(1.0, max(0.0, np.log(rs) / np.log(n)))
            except:
                return 0.5

        return df['returns_1'].rolling(window).apply(hurst_exp)

    def get_feature_summary(self) -> Dict:
        """Get summary of generated features"""
        summary = {}
        total_features = 0

        for category, features in self.feature_categories.items():
            count = len(features)
            total_features += count
            summary[category] = {
                'count': count,
                'features': features
            }

        summary['total'] = total_features
        return summary