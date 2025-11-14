#!/usr/bin/env python3
"""
Replay paper-trader signals using the same stop/target logic as training labels.

Usage:
    PYTHONPATH=. python scripts/evaluate_signal_accuracy.py \
        --log results/live_trading/btcusdt_paper_log_20251110_135555.txt \
        --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.data.database import TradingDatabase

DEFAULT_FUTURE_PERIODS = 20
DEFAULT_INTERVAL_MIN = 15
STOP_LOW = 0.003
STOP_HIGH = 0.02
TARGET_MULTIPLIER = 1.75


def parse_log(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = rec.get("signal_timestamp") or rec.get("timestamp")
            direction = rec.get("direction")
            price = rec.get("price")
            if not ts or direction not in ("UP", "DOWN") or price is None:
                continue
            rec["parsed_ts"] = pd.Timestamp(ts)
            records.append(rec)
    return records


def load_ohlcv(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    db = TradingDatabase()
    start_ts = int(start.timestamp() * 1000)
    end_ts = int(end.timestamp() * 1000)
    df = db.get_ohlcv_data(symbol, interval, start_timestamp=start_ts, end_timestamp=end_ts)
    if df.empty:
        raise ValueError(f"No OHLCV data for {symbol} between {start} and {end}")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("datetime").sort_index()


def evaluate_signal(
    entry_price: float,
    future_prices: pd.Series,
    direction: str,
    stop_pct: float,
) -> str:
    if direction == "UP":
        stop_price = entry_price * (1 - stop_pct)
        target_price = entry_price * (1 + stop_pct * TARGET_MULTIPLIER)
        for price in future_prices:
            if price >= target_price:
                return "target"
            if price <= stop_price:
                return "stop"
    else:
        stop_price = entry_price * (1 + stop_pct)
        target_price = entry_price * (1 - stop_pct * TARGET_MULTIPLIER)
        for price in future_prices:
            if price <= target_price:
                return "target"
            if price >= stop_price:
                return "stop"
    return "no_event"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paper-trader signals with stop-aligned logic.")
    parser.add_argument("--log", required=True, type=Path, help="Path to paper_log_*.txt")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (default BTCUSDT)")
    parser.add_argument("--interval", default="15m", help="Interval (default 15m)")
    parser.add_argument("--future-periods", type=int, default=DEFAULT_FUTURE_PERIODS,
                        help="Number of future bars to inspect (default 20)")
    args = parser.parse_args()

    records = parse_log(args.log)
    if not records:
        print("No usable signals found in log.")
        return

    # Determine time range for OHLCV query
    timestamps = [rec["parsed_ts"] for rec in records]
    start_ts = min(timestamps) - pd.Timedelta(days=1)
    interval_minutes = DEFAULT_INTERVAL_MIN
    if args.interval.endswith("m"):
        try:
            interval_minutes = int(args.interval.rstrip("m"))
        except ValueError:
            pass
    horizon = pd.Timedelta(minutes=args.future_periods * interval_minutes)
    end_ts = max(timestamps) + horizon + pd.Timedelta(hours=1)

    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    ohlcv = load_ohlcv(args.symbol, args.interval, start_ts, end_ts)

    summary = {
        "total": 0,
        "target": 0,
        "stop": 0,
        "no_event": 0,
        "by_direction": {"UP": {"target": 0, "stop": 0, "no_event": 0},
                         "DOWN": {"target": 0, "stop": 0, "no_event": 0}},
    }

    for rec in records:
        entry_time = rec["parsed_ts"]
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        else:
            entry_time = entry_time.tz_convert("UTC")
        entry_price = float(rec["price"])
        vol = rec.get("volatility_1h")
        if vol is None:
            stop_pct = STOP_LOW
        else:
            stop_pct = float(vol)
        stop_pct = float(max(STOP_LOW, min(STOP_HIGH, abs(stop_pct))))

        future = ohlcv.loc[(ohlcv.index > entry_time) & (ohlcv.index <= entry_time + horizon), "close"].head(args.future_periods)
        if len(future) < args.future_periods:
            continue

        outcome = evaluate_signal(entry_price, future, rec["direction"], stop_pct)
        summary["total"] += 1
        summary[outcome] += 1
        summary["by_direction"][rec["direction"]][outcome] += 1

    if summary["total"] == 0:
        print("No signals could be evaluated with the current OHLC data.")
        return

    print(f"Log: {args.log}")
    print(f"Symbol: {args.symbol}")
    print(f"Evaluated signals: {summary['total']}")
    for outcome in ("target", "stop", "no_event"):
        count = summary[outcome]
        pct = count / summary["total"] * 100
        print(f"  {outcome}: {count} ({pct:.2f}%)")

    print("\nBy direction:")
    for direction, stats in summary["by_direction"].items():
        total_dir = sum(stats.values())
        if total_dir == 0:
            continue
        print(f"  {direction} (total {total_dir}): "
              f"target={stats['target']} stop={stats['stop']} no_event={stats['no_event']}")


if __name__ == "__main__":
    main()
