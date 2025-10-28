#!/usr/bin/env python3
"""
Split the primary trading database into pre-/post-2025 shards for training/evaluation experiments.

Usage:
    PYTHONPATH=. python tests/model_retrain_eval/split_database.py

Outputs:
    data/trading_data_pre2025.db
    data/trading_data_post2025.db
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.data.database import TradingDatabase

SOURCE_DB = Path("data/trading_data.db")
PRE_DB = Path("data/trading_data_pre2025.db")
POST_DB = Path("data/trading_data_post2025.db")
CUTOFF_TS = pd.Timestamp("2025-01-01", tz="UTC")


def ensure_schema(db_path: Path) -> None:
    TradingDatabase(db_path=str(db_path))


def copy_split_data() -> None:
    if not SOURCE_DB.exists():
        raise FileNotFoundError(f"Source database not found: {SOURCE_DB}")

    ensure_schema(PRE_DB)
    ensure_schema(POST_DB)

    with sqlite3.connect(SOURCE_DB) as conn:
        df = pd.read_sql_query("SELECT * FROM ohlcv_data", conn)

    if df.empty:
        raise RuntimeError("Source database contains no OHLCV data.")

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df_pre = df[df["datetime"] < CUTOFF_TS].copy()
    df_post = df[df["datetime"] >= CUTOFF_TS].copy()

    for target_db, shard in [(PRE_DB, df_pre), (POST_DB, df_post)]:
        shard = shard.drop(columns=["datetime"])
        shard.to_sql("ohlcv_data", sqlite3.connect(target_db), if_exists="append", index=False)
        print(f"Wrote {len(shard):,} rows to {target_db}")


def main() -> None:
    copy_split_data()
    print("✅ Database split complete.")


if __name__ == "__main__":
    main()
