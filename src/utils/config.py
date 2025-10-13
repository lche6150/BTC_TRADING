import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Centralized configuration management"""

    # Binance API Configuration
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
    BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = 1200
    RATE_LIMIT_BUFFER = 0.9  # Use 90% of limit for safety

    # Data Configuration
    DEFAULT_SYMBOL = "BTCUSDT"
    DEFAULT_INTERVAL = "15m"
    HISTORICAL_LOOKBACK_DAYS = 365

    # Database Configuration
    DATABASE_PATH = "data/trading_data.db"

    # Trading Configuration
    MAX_POSITION_SIZE = 0.03  # 3% per trade
    MAX_PORTFOLIO_HEAT = 0.06  # 6% total exposure
    DEFAULT_STOP_LOSS = 0.02  # 2% stop loss
    MIN_RISK_REWARD_RATIO = 2.0

    # Model Configuration
    XGBOOST_FEATURE_COUNT = 20
    LSTM_SEQUENCE_LENGTH = 60
    VALIDATION_SPLIT = 0.2

    @classmethod
    def get_binance_config(cls) -> Dict[str, Any]:
        """Get Binance-specific configuration"""
        return {
            "api_key": cls.BINANCE_API_KEY,
            "api_secret": cls.BINANCE_SECRET_KEY,
            "testnet": cls.BINANCE_TESTNET,
            "rate_limit": cls.MAX_REQUESTS_PER_MINUTE
        }

    @classmethod
    def validate_api_keys(cls) -> bool:
        """Validate that required API keys are set"""
        return bool(cls.BINANCE_API_KEY and cls.BINANCE_SECRET_KEY)