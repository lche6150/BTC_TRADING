import sqlite3
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path
from ..utils.config import Config

class TradingDatabase:
    """SQLite database manager for trading data storage"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or Config.DATABASE_PATH
        self.logger = logging.getLogger(__name__)
        self._ensure_db_directory()
        self._init_database()

    def _ensure_db_directory(self):
        """Ensure database directory exists"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Main OHLCV data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    interval_type TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    close_time INTEGER NOT NULL,
                    quote_asset_volume REAL,
                    number_of_trades INTEGER,
                    taker_buy_base_asset_volume REAL,
                    taker_buy_quote_asset_volume REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, symbol, interval_type)
                )
            """)

            # Index for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp_symbol
                ON ohlcv_data(timestamp, symbol, interval_type)
            """)

            # Data collection metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_collection_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval_type TEXT NOT NULL,
                    start_timestamp INTEGER NOT NULL,
                    end_timestamp INTEGER NOT NULL,
                    records_count INTEGER NOT NULL,
                    collection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'completed'
                )
            """)

            # Features table (for future use)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    interval_type TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, symbol, interval_type, feature_name)
                )
            """)

            conn.commit()

        self.logger.info(f"Database initialized at {self.db_path}")

    def insert_ohlcv_data(self, data: List[Dict], symbol: str, interval: str) -> int:
        """
        Insert OHLCV data into database

        Args:
            data: List of OHLCV dictionaries from Binance API
            symbol: Trading pair symbol
            interval: Time interval

        Returns:
            Number of records inserted
        """
        if not data:
            return 0

        insert_query = """
            INSERT OR IGNORE INTO ohlcv_data (
                timestamp, symbol, interval_type, open, high, low, close, volume,
                close_time, quote_asset_volume, number_of_trades,
                taker_buy_base_asset_volume, taker_buy_quote_asset_volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        records_to_insert = []
        for record in data:
            records_to_insert.append((
                record['timestamp'],
                symbol,
                interval,
                record['open'],
                record['high'],
                record['low'],
                record['close'],
                record['volume'],
                record['close_time'],
                record.get('quote_asset_volume', 0),
                record.get('number_of_trades', 0),
                record.get('taker_buy_base_asset_volume', 0),
                record.get('taker_buy_quote_asset_volume', 0)
            ))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(insert_query, records_to_insert)
            inserted_count = cursor.rowcount
            conn.commit()

        self.logger.info(f"Inserted {inserted_count} new records for {symbol} {interval}")
        return inserted_count

    def get_ohlcv_data(
        self,
        symbol: str,
        interval: str,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV data from database

        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_timestamp: Start timestamp (optional)
            end_timestamp: End timestamp (optional)
            limit: Maximum number of records (optional)

        Returns:
            DataFrame with OHLCV data
        """
        query = """
            SELECT timestamp, open, high, low, close, volume,
                   close_time, quote_asset_volume, number_of_trades,
                   taker_buy_base_asset_volume, taker_buy_quote_asset_volume
            FROM ohlcv_data
            WHERE symbol = ? AND interval_type = ?
        """
        params = [symbol, interval]

        if start_timestamp:
            query += " AND timestamp >= ?"
            params.append(start_timestamp)

        if end_timestamp:
            query += " AND timestamp <= ?"
            params.append(end_timestamp)

        query += " ORDER BY timestamp ASC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df

    def get_data_range(self, symbol: str, interval: str) -> Tuple[Optional[datetime], Optional[datetime], int]:
        """
        Get the timestamp range and count for existing data

        Returns:
            Tuple of (earliest_datetime, latest_datetime, total_count)
        """
        query = """
            SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
            FROM ohlcv_data
            WHERE symbol = ? AND interval_type = ?
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (symbol, interval))
            min_ts, max_ts, count = cursor.fetchone()

        min_dt = datetime.fromtimestamp(min_ts / 1000) if min_ts else None
        max_dt = datetime.fromtimestamp(max_ts / 1000) if max_ts else None

        return min_dt, max_dt, count

    def log_data_collection(
        self,
        symbol: str,
        interval: str,
        start_timestamp: int,
        end_timestamp: int,
        records_count: int,
        status: str = "completed"
    ):
        """Log data collection activity"""
        query = """
            INSERT INTO data_collection_log
            (symbol, interval_type, start_timestamp, end_timestamp, records_count, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(query, (symbol, interval, start_timestamp, end_timestamp, records_count, status))
            conn.commit()

    def get_collection_history(self, symbol: str, interval: str) -> pd.DataFrame:
        """Get data collection history"""
        query = """
            SELECT * FROM data_collection_log
            WHERE symbol = ? AND interval_type = ?
            ORDER BY collection_date DESC
        """

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=(symbol, interval))

    def get_missing_data_gaps(
        self,
        symbol: str,
        interval: str,
        expected_interval_minutes: int = 15
    ) -> List[Tuple[datetime, datetime]]:
        """
        Identify gaps in the data where records are missing

        Args:
            symbol: Trading pair symbol
            interval: Time interval
            expected_interval_minutes: Expected interval between records in minutes

        Returns:
            List of (start_gap, end_gap) datetime tuples
        """
        df = self.get_ohlcv_data(symbol, interval)

        if df.empty:
            return []

        df = df.sort_values('timestamp')
        gaps = []

        expected_interval_ms = expected_interval_minutes * 60 * 1000

        for i in range(1, len(df)):
            current_ts = df.iloc[i]['timestamp']
            prev_ts = df.iloc[i-1]['timestamp']

            time_diff = current_ts - prev_ts

            if time_diff > expected_interval_ms * 1.5:  # Allow for some tolerance
                gap_start = datetime.fromtimestamp(prev_ts / 1000)
                gap_end = datetime.fromtimestamp(current_ts / 1000)
                gaps.append((gap_start, gap_end))

        return gaps

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total records
            cursor.execute("SELECT COUNT(*) FROM ohlcv_data")
            total_records = cursor.fetchone()[0]

            # Records by symbol
            cursor.execute("""
                SELECT symbol, interval_type, COUNT(*), MIN(timestamp), MAX(timestamp)
                FROM ohlcv_data
                GROUP BY symbol, interval_type
            """)
            symbol_stats = cursor.fetchall()

            # Database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]

        return {
            'total_records': total_records,
            'database_size_bytes': db_size,
            'database_size_mb': db_size / (1024 * 1024),
            'symbol_stats': [
                {
                    'symbol': stat[0],
                    'interval': stat[1],
                    'count': stat[2],
                    'start_date': datetime.fromtimestamp(stat[3] / 1000) if stat[3] else None,
                    'end_date': datetime.fromtimestamp(stat[4] / 1000) if stat[4] else None
                }
                for stat in symbol_stats
            ]
        }

    def cleanup_old_data(self, symbol: str, interval: str, keep_days: int = 400):
        """Remove data older than specified days"""
        cutoff_timestamp = int((datetime.now().timestamp() - (keep_days * 24 * 60 * 60)) * 1000)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM ohlcv_data WHERE symbol = ? AND interval_type = ? AND timestamp < ?",
                (symbol, interval, cutoff_timestamp)
            )
            deleted_count = cursor.rowcount
            conn.commit()

        self.logger.info(f"Cleaned up {deleted_count} old records for {symbol} {interval}")
        return deleted_count