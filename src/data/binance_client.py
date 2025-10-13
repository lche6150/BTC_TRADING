import asyncio
import time
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from binance.client import Client
from binance.enums import *
import pandas as pd
import logging
from ..utils.config import Config

class RateLimiter:
    """Rate limiter for API requests"""

    def __init__(self, max_requests: int = 1200, window_seconds: int = 60):
        self.max_requests = int(max_requests * Config.RATE_LIMIT_BUFFER)
        self.window_seconds = window_seconds
        self.requests = []
        self._lock = asyncio.Lock()

    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        async with self._lock:
            current_time = time.time()
            # Remove old requests outside the window
            self.requests = [req_time for req_time in self.requests
                           if current_time - req_time < self.window_seconds]

            if len(self.requests) >= self.max_requests:
                sleep_time = self.window_seconds - (current_time - self.requests[0]) + 1
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            self.requests.append(current_time)

class BinanceAPIClient:
    """
    Extensible Binance API client for data fetching and future trading operations.
    Designed to handle both market data collection and order execution.
    """

    def __init__(self):
        self.config = Config.get_binance_config()
        self.client: Optional[Client] = None
        self.rate_limiter = RateLimiter()
        self.logger = logging.getLogger(__name__)
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Binance client with proper configuration"""
        if not Config.validate_api_keys():
            self.logger.warning("API keys not configured. Some features will be unavailable.")
            return

        try:
            self.client = Client(
                api_key=self.config["api_key"],
                api_secret=self.config["api_secret"],
                testnet=self.config["testnet"]
            )
            # Test connection
            self.client.ping()
            self.logger.info("Binance client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {e}")
            self.client = None

    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_str: Optional[str] = None,
        end_str: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Fetch historical kline/candlestick data

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '15m', '1h', '1d')
            start_str: Start time string (optional)
            end_str: End time string (optional)
            limit: Number of records to return (max 1000)

        Returns:
            List of kline data dictionaries
        """
        if not self.client:
            raise Exception("Binance client not initialized")

        await self.rate_limiter.wait_if_needed()

        try:
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=limit
            )

            # Convert to structured format
            formatted_klines = []
            for kline in klines:
                formatted_klines.append({
                    'timestamp': int(kline[0]),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': int(kline[6]),
                    'quote_asset_volume': float(kline[7]),
                    'number_of_trades': int(kline[8]),
                    'taker_buy_base_asset_volume': float(kline[9]),
                    'taker_buy_quote_asset_volume': float(kline[10])
                })

            return formatted_klines

        except Exception as e:
            self.logger.error(f"Error fetching historical klines: {e}")
            raise

    async def get_recent_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch recent kline data

        Args:
            symbol: Trading pair
            interval: Kline interval
            limit: Number of recent klines

        Returns:
            List of recent kline data
        """
        if not self.client:
            raise Exception("Binance client not initialized")

        await self.rate_limiter.wait_if_needed()

        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )

            # Convert to structured format (same as historical)
            formatted_klines = []
            for kline in klines:
                formatted_klines.append({
                    'timestamp': int(kline[0]),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': int(kline[6]),
                    'quote_asset_volume': float(kline[7]),
                    'number_of_trades': int(kline[8]),
                    'taker_buy_base_asset_volume': float(kline[9]),
                    'taker_buy_quote_asset_volume': float(kline[10])
                })

            return formatted_klines

        except Exception as e:
            self.logger.error(f"Error fetching recent klines: {e}")
            raise

    async def get_symbol_ticker(self, symbol: str) -> Dict:
        """Get 24hr ticker statistics for a symbol"""
        if not self.client:
            raise Exception("Binance client not initialized")

        await self.rate_limiter.wait_if_needed()

        try:
            ticker = self.client.get_ticker(symbol=symbol)
            return {
                'symbol': ticker['symbol'],
                'price_change': float(ticker['priceChange']),
                'price_change_percent': float(ticker['priceChangePercent']),
                'weighted_avg_price': float(ticker['weightedAvgPrice']),
                'prev_close_price': float(ticker['prevClosePrice']),
                'last_price': float(ticker['lastPrice']),
                'last_qty': float(ticker['lastQty']),
                'bid_price': float(ticker['bidPrice']),
                'ask_price': float(ticker['askPrice']),
                'open_price': float(ticker['openPrice']),
                'high_price': float(ticker['highPrice']),
                'low_price': float(ticker['lowPrice']),
                'volume': float(ticker['volume']),
                'quote_volume': float(ticker['quoteVolume']),
                'open_time': int(ticker['openTime']),
                'close_time': int(ticker['closeTime']),
                'count': int(ticker['count'])
            }
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {e}")
            raise

    async def collect_historical_data(
        self,
        symbol: str,
        interval: str,
        days_back: int = 365
    ) -> pd.DataFrame:
        """
        Collect comprehensive historical data for the specified period

        Args:
            symbol: Trading pair
            interval: Data interval
            days_back: Number of days to collect

        Returns:
            DataFrame with historical OHLCV data
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)

        all_data = []
        current_start = start_time

        # Fetch data in chunks to handle API limits
        while current_start < end_time:
            current_end = min(current_start + timedelta(days=30), end_time)

            self.logger.info(f"Fetching data from {current_start} to {current_end}")

            klines = await self.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=current_start.strftime('%Y-%m-%d'),
                end_str=current_end.strftime('%Y-%m-%d'),
                limit=1000
            )

            all_data.extend(klines)
            current_start = current_end

            # Small delay to be respectful to the API
            await asyncio.sleep(0.1)

        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

        self.logger.info(f"Collected {len(df)} data points for {symbol}")
        return df

    # Future trading methods (extensible design)
    async def get_account_info(self) -> Dict:
        """Get account information (for future trading features)"""
        if not self.client:
            raise Exception("Binance client not initialized")

        await self.rate_limiter.wait_if_needed()

        try:
            return self.client.get_account()
        except Exception as e:
            self.logger.error(f"Error fetching account info: {e}")
            raise

    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders (for future trading features)"""
        if not self.client:
            raise Exception("Binance client not initialized")

        await self.rate_limiter.wait_if_needed()

        try:
            return self.client.get_open_orders(symbol=symbol)
        except Exception as e:
            self.logger.error(f"Error fetching open orders: {e}")
            raise

    async def create_order(self, **kwargs) -> Dict:
        """
        Create a new order (for future trading implementation)
        This method is a placeholder for future trading functionality
        """
        if not self.client:
            raise Exception("Binance client not initialized")

        await self.rate_limiter.wait_if_needed()

        # Log the order attempt for safety
        self.logger.warning(f"Order creation attempted with params: {kwargs}")

        # In testnet mode or for development, you would implement:
        # return self.client.create_order(**kwargs)

        raise NotImplementedError("Order creation not yet implemented for safety")

    def get_client_status(self) -> Dict[str, Any]:
        """Get client status and configuration"""
        return {
            'initialized': self.client is not None,
            'testnet': self.config['testnet'],
            'rate_limit': self.config['rate_limit'],
            'api_key_configured': bool(self.config['api_key'])
        }