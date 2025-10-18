#!/usr/bin/env python3
"""
Create two isolated SQLite databases derived from the master trading database.

You can either:
- Provide explicit calendar boundaries for training/testing, or
- Supply ``--train-ratio`` to split the selected history contiguously
  (e.g., 0.8 → first 80 % for training, final 20 % for testing).

The original database is left untouched.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

CREATE_OHLCV_SQL = """
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
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp_symbol
ON ohlcv_data(timestamp, symbol, interval_type)
"""


def ensure_schema(database_path: Path) -> None:
    with sqlite3.connect(database_path) as conn:
        conn.execute(CREATE_OHLCV_SQL)
        conn.execute(CREATE_INDEX_SQL)
        conn.commit()


def copy_range(
    source_db: Path,
    dest_db: Path,
    start_date: str,
    end_date: str | None,
    symbol: str = "BTCUSDT",
    interval: str = "15m",

) -> int:
    """
    Copy OHLCV rows based on calendar date boundaries (inclusive).

    Retained for backward compatibility with explicit date inputs.
    """
    ensure_schema(dest_db)

    with sqlite3.connect(source_db) as src_conn, sqlite3.connect(dest_db) as dest_conn:
        max_ts = _max_timestamp(src_conn)
        if max_ts == 0:
            return 0

        start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
        if end_date:
            end_ms = int(
                (pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)).timestamp()
                * 1000
            )
        else:
            end_ms = max_ts

        if start_ms > end_ms:
            return 0

        query = """
        SELECT timestamp, open, high, low, close, volume,
               close_time, quote_asset_volume, number_of_trades,
               taker_buy_base_asset_volume, taker_buy_quote_asset_volume
        FROM ohlcv_data
        WHERE symbol = ? AND interval_type = ?
          AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp
        """

        end_bound = min(end_ms, max_ts)
        df = pd.read_sql_query(query, src_conn, params=(symbol, interval, start_ms, end_bound))

        if df.empty:
            return 0

        df.insert(2, "symbol", symbol)
        df.insert(3, "interval_type", interval)
        df.to_sql("ohlcv_data", dest_conn, if_exists="append", index=False)
        dest_conn.commit()

    return len(df)


def copy_range_by_ts(
    source_db: Path,
    dest_db: Path,
    start_ts: int,
    end_ts: Optional[int],
    symbol: str = "BTCUSDT",
    interval: str = "15m",
) -> int:
    """Copy OHLCV rows using exact timestamp boundaries (inclusive)."""
    ensure_schema(dest_db)

    with sqlite3.connect(source_db) as src_conn, sqlite3.connect(dest_db) as dest_conn:
        max_ts = _max_timestamp(src_conn)
        if max_ts == 0:
            return 0

        effective_start = max(start_ts, 0)
        effective_end = end_ts if end_ts is not None else max_ts
        effective_end = min(effective_end, max_ts)

        if effective_start > effective_end:
            return 0

        query = """
        SELECT timestamp, open, high, low, close, volume,
               close_time, quote_asset_volume, number_of_trades,
               taker_buy_base_asset_volume, taker_buy_quote_asset_volume
        FROM ohlcv_data
        WHERE symbol = ? AND interval_type = ?
          AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp
        """

        df = pd.read_sql_query(query, src_conn, params=(symbol, interval, effective_start, effective_end))
        if df.empty:
            return 0

        df.insert(2, "symbol", symbol)
        df.insert(3, "interval_type", interval)
        df.to_sql("ohlcv_data", dest_conn, if_exists="append", index=False)
        dest_conn.commit()

    return len(df)


def _max_timestamp(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COALESCE(MAX(timestamp), 0) FROM ohlcv_data")
    return cur.fetchone()[0]


def _to_timestamp(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    ts = pd.Timestamp(date_str, tz="UTC")
    return int(ts.timestamp() * 1000)


def determine_ratio_split(
    source_db: Path,
    symbol: str,
    interval: str,
    train_ratio: float,
    min_start_ts: Optional[int],
    max_end_ts: Optional[int],
) -> Tuple[int, int, int, int]:
    """
    Determine contiguous timestamps that split the dataset into training/testing
    portions according to the requested ratio.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1 (exclusive)")

    with sqlite3.connect(source_db) as conn:
        conditions = ["symbol = ?", "interval_type = ?"]
        params = [symbol, interval]

        if min_start_ts is not None:
            conditions.append("timestamp >= ?")
            params.append(min_start_ts)
        if max_end_ts is not None:
            conditions.append("timestamp <= ?")
            params.append(max_end_ts)

        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT timestamp
        FROM ohlcv_data
        WHERE {where_clause}
        ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        raise ValueError("No OHLCV data available for the requested filters.")

    total_rows = len(df)
    if total_rows < 2:
        raise ValueError("Not enough rows to create train/test splits.")

    cutoff = int(total_rows * train_ratio)
    cutoff = max(1, min(cutoff, total_rows - 1))

    start_ts = int(df.iloc[0]["timestamp"])
    train_end_ts = int(df.iloc[cutoff - 1]["timestamp"])
    test_start_ts = int(df.iloc[cutoff]["timestamp"])
    end_ts = int(df.iloc[-1]["timestamp"])

    return start_ts, train_end_ts, test_start_ts, end_ts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create isolated training/testing SQLite databases.")
    parser.add_argument("--source", type=Path, default=Path("data/trading_data.db"), help="Source master database.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/isolated"), help="Directory for isolated databases.")
    parser.add_argument("--train-start", default="2022-01-01", help="Training data start date (UTC).")
    parser.add_argument("--train-end", default="2024-06-30", help="Training data end date (UTC, inclusive).")
    parser.add_argument("--test-start", default="2026-07-01", help="Testing data start date (UTC).")
    parser.add_argument("--overall-end", default=None, help="Optional overall end date (UTC) when using ratio split.")
    parser.add_argument("--train-ratio", type=float, default=None,
                        help="If provided, automatically split the data by ratio (e.g., 0.8 for 80/20).")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol to filter.")
    parser.add_argument("--interval", default="15m", help="Interval type to filter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_db = args.output_dir / "isolated_train.db"
    test_db = args.output_dir / "isolated_test.db"

    print(f"📁 Source database: {args.source}")
    print(f"📁 Training DB:    {train_db}")
    print(f"📁 Testing DB:     {test_db}")

    if train_db.exists():
        train_db.unlink()
    if test_db.exists():
        test_db.unlink()

    if args.train_ratio is not None:
        min_start_ts = _to_timestamp(args.train_start)
        max_end_ts = _to_timestamp(args.overall_end)

        start_ts, train_end_ts, test_start_ts, end_ts = determine_ratio_split(
            args.source,
            args.symbol,
            args.interval,
            args.train_ratio,
            min_start_ts,
            max_end_ts,
        )

        train_rows = copy_range_by_ts(args.source, train_db, start_ts, train_end_ts, args.symbol, args.interval)
        test_rows = copy_range_by_ts(args.source, test_db, test_start_ts, end_ts, args.symbol, args.interval)

        print(f"📐 Automatic ratio split (train_ratio={args.train_ratio:.2f}):")
        print(f"   Training  : {pd.Timestamp(start_ts, unit='ms', tz='UTC')} → {pd.Timestamp(train_end_ts, unit='ms', tz='UTC')}")
        print(f"   Testing   : {pd.Timestamp(test_start_ts, unit='ms', tz='UTC')} → {pd.Timestamp(end_ts, unit='ms', tz='UTC')}")
    else:
        train_rows = copy_range(args.source, train_db, args.train_start, args.train_end, args.symbol, args.interval)
        test_rows = copy_range(args.source, test_db, args.test_start, None, args.symbol, args.interval)

    print(f"✅ Training DB rows copied: {train_rows}")
    print(f"✅ Testing DB rows copied:  {test_rows}")
    print("Done.")


if __name__ == "__main__":
    main()
