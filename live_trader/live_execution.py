#!/usr/bin/env python3
"""
Real trading loop using the ComprehensiveTradingSystem and Binance spot API.

Usage (dry-run / testnet first):
    python live_trader/live_trader.py \
        --symbol BTCUSDT \
        --interval 15m \
        --lookback-days 30 \
        --config best_config.json \
        --log results/live_trading/live_trades.log

To place actual orders, add `--live` (ensure BINANCE_TESTNET=false and real API keys).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from live_trader.model_loader import GRUSignalModel  # noqa: E402
from live_trader.trading_api import BinanceSpotTradingAPI  # noqa: E402
from src.data.database import TradingDatabase  # noqa: E402
from src.data.features import EnhancedFeatureEngine  # noqa: E402
from src.trading.confidence_trading_system import ComprehensiveTradingSystem, MarketData  # noqa: E402
from trading_backtest import (  # noqa: E402
    MODEL_FEATURES,
    retrain_gru_model_until,
    _purge_gru_backtest_artifacts,
)
from src.utils.artifacts import get_gru_backtest_dir, get_gru_signal_dir

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "15m"
DEFAULT_LOOKBACK_DAYS = 30
MAX_HISTORY_ROWS = 6000
EVALUATION_WINDOW_MINUTES = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live trading loop powered by Binance futures.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Trading symbol (default BTCUSDT)")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Kline interval (default 15m)")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help="Historical days to bootstrap features")
    parser.add_argument("--iterations", type=int, default=0,
                        help="Number of polling iterations (0 for infinite)")
    parser.add_argument("--sleep", type=int, default=60,
                        help="Seconds to sleep between polling cycles (default 60)")
    parser.add_argument("--config", type=Path, default=Path("best_config.json"),
                        help="Trading system configuration JSON")
    parser.add_argument("--log", type=Path, default=Path("results/live_trading/live_trades.log"),
                        help="Path to append trade execution events")
    parser.add_argument("--signal-log-dir", type=Path, default=Path("results/live_trading"),
                        help="Directory to write signal diagnostics")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate executions without touching the exchange (default).")
    parser.add_argument("--live", action="store_true",
                        help="Explicitly place orders on the configured exchange (disables dry-run).")
    parser.add_argument("--min-notional", type=float, default=10.0,
                        help="Minimum USD notional to trade (additional guardrail).")
    return parser.parse_args()


def load_trading_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Trading config not found: {path}")
    with path.open("r") as f:
        return json.load(f)


def klines_to_dataframe(klines: List[List]) -> pd.DataFrame:
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(klines, columns=columns)
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def enrich_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)
    df["returns"] = df["close"].pct_change()
    df["volume_ma"] = df["volume"].rolling(24).mean()
    df["volatility"] = df["returns"].rolling(24).std()
    df["price_change_1h"] = df["close"].pct_change(4)
    df["price_change_4h"] = df["close"].pct_change(16)
    df["sma_short"] = df["close"].rolling(12).mean()
    df["sma_long"] = df["close"].rolling(48).mean()
    df["trend_1h"] = np.where(
        df["sma_short"] > df["sma_long"], "BULLISH",
        np.where(df["sma_short"] < df["sma_long"], "BEARISH", "NEUTRAL")
    )
    return df


def prepare_feature_dataframe(df: pd.DataFrame, feature_engine: EnhancedFeatureEngine) -> pd.DataFrame:
    engineered = feature_engine.create_all_features(df, include_multiframe=True, min_periods=100)
    required_cols = [c for c in MODEL_FEATURES if c in engineered.columns]
    engineered = engineered.dropna(subset=required_cols + ["datetime", "close", "volume"])
    engineered = engineered.reset_index(drop=True)
    engineered["datetime"] = pd.to_datetime(engineered["datetime"])
    return engineered


def get_client(testnet: bool) -> Client:
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")
    return Client(api_key=api_key, api_secret=api_secret, testnet=testnet)


def fetch_initial_data(client: Client, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - lookback_days * 24 * 60 * 60 * 1000
    klines = client.get_historical_klines(symbol=symbol, interval=interval,
                                          start_str=start_ts, end_str=end_ts)
    if not klines:
        raise RuntimeError("Failed to fetch initial klines from Binance")
    return klines_to_dataframe(klines)


def fetch_new_klines(
    client: Client,
    symbol: str,
    interval: str,
    start_time_ms: int,
    max_retries: int = 3,
    retry_delay: int = 60,
) -> Tuple[pd.DataFrame, bool]:
    attempts = 0
    while attempts < max_retries:
        try:
            klines = client.get_klines(symbol=symbol, interval=interval, startTime=start_time_ms)
            if not klines:
                return pd.DataFrame(), attempts > 0
            return klines_to_dataframe(klines), attempts > 0

        except BinanceAPIException as exc:
            print(f"⚠️  Binance API error: {exc}")
            return pd.DataFrame(), attempts > 0

        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            attempts += 1
            if attempts >= max_retries:
                print(f"⚠️  Network timeout fetching klines: {exc}. Giving up after {attempts} attempts.")
                break
            print(f"⚠️  Network timeout fetching klines: {exc}. Retrying in {retry_delay}s "
                  f"({attempts}/{max_retries})")
            time.sleep(retry_delay)

    return pd.DataFrame(), True


def log_event(log_file, event_type: str, payload: Dict[str, Any]) -> None:
    if log_file is None:
        return
    entry = {"type": event_type, **payload}
    log_file.write(json.dumps(entry, default=str) + "\n")
    log_file.flush()


def log_signal_decision(signal_log_path: Path, payload: Dict[str, Any]) -> None:
    if signal_log_path is None:
        return
    with signal_log_path.open("a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def register_pending_evaluation(pending_list: List[Dict[str, Any]], record: Dict[str, Any]) -> None:
    pending_list.append(record)


def update_signal_log_evaluation(
    signal_log_path: Path,
    log_id: str,
    status: str,
    evaluation_time: pd.Timestamp,
    evaluation_price: float,
) -> None:
    if not signal_log_path.exists():
        return

    lines = signal_log_path.read_text().splitlines()
    updated = False

    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("log_id") == log_id:
            record["evaluation_status"] = status
            record["evaluation_timestamp"] = evaluation_time.isoformat()
            record["evaluation_price"] = evaluation_price
            lines[idx] = json.dumps(record, default=str)
            updated = True
            break

    if updated:
        signal_log_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def evaluate_pending_signals(
    pending_list: List[Dict[str, Any]],
    price_df: pd.DataFrame,
    signal_log_path: Path,
) -> None:
    if not pending_list or price_df.empty:
        return

    df_sorted = price_df.sort_values("datetime")
    max_timestamp = df_sorted["datetime"].iloc[-1]

    remaining: List[Dict[str, Any]] = []

    for item in pending_list:
        deadline: datetime = item["deadline"]
        if max_timestamp < pd.Timestamp(deadline):
            remaining.append(item)
            continue

        future_rows = df_sorted[df_sorted["datetime"] >= pd.Timestamp(deadline)]
        if future_rows.empty:
            remaining.append(item)
            continue

        future_row = future_rows.iloc[0]
        evaluation_price = float(future_row["close"])
        direction = item["direction"]
        entry_price = item["entry_price"]

        if direction == "UP":
            status = "hit" if evaluation_price >= entry_price else "miss"
        else:
            status = "hit" if evaluation_price <= entry_price else "miss"

        update_signal_log_evaluation(
            signal_log_path,
            item["log_id"],
            status,
            future_row["datetime"],
            evaluation_price,
        )

    pending_list[:] = remaining


def sync_signal_model_artifacts(symbol: str) -> bool:
    source_dir = get_gru_backtest_dir(symbol)
    target_dir = get_gru_signal_dir(symbol)

    if not source_dir.exists():
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    required_files = ["model.keras", "scaler.pkl", "meta.json"]
    copied_any = False
    for name in required_files:
        src = source_dir / name
        dst = target_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            copied_any = True
    return copied_any


def upsert_initial_data(df: pd.DataFrame, symbol: str, interval: str) -> None:
    if df.empty:
        return

    records = []
    for row in df.itertuples(index=False):
        records.append({
            "timestamp": int(row.open_time),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
            "close_time": int(row.close_time),
            "quote_asset_volume": float(row.quote_asset_volume),
            "number_of_trades": int(row.number_of_trades),
            "taker_buy_base_asset_volume": float(row.taker_buy_base),
            "taker_buy_quote_asset_volume": float(row.taker_buy_quote),
        })

    if not records:
        return

    db = TradingDatabase()
    db.insert_ohlcv_data(records, symbol, interval)


def compute_order_quantity(
    system: ComprehensiveTradingSystem,
    position,
    current_price: float,
    api: Optional[BinanceSpotTradingAPI],
    min_notional: float,
    direction: str,
) -> float:
    notional = max(system.current_capital * position.position_size_pct, 0.0)
    if notional < max(min_notional, 1.0):
        return 0.0

    qty = notional / current_price
    if not api:
        return max(qty, 0.0)

    if direction == "DOWN":
        base_asset = api.symbol.replace("USDT", "")
        base_balance = api.get_balance(base_asset)
        qty = min(base_balance, qty)
    else:
        quote_balance = api.get_balance("USDT")
        tradable_notional = min(quote_balance, notional)
        qty = tradable_notional / current_price

    qty = api.quantize_quantity(qty)
    return max(qty, 0.0)


def process_new_signal(
    system: ComprehensiveTradingSystem,
    api: Optional[BinanceSpotTradingAPI],
    position_records: Dict[str, Dict[str, Any]],
    signal_row: pd.Series,
    prob_info: Dict[str, float],
    threshold: float,
    log_file,
    signal_log_path: Path,
    pending_evaluations: List[Dict[str, Any]],
    evaluation_window_minutes: int,
    dry_run: bool,
    min_notional: float,
) -> None:
    current_time = pd.to_datetime(signal_row["datetime"])
    current_price = float(signal_row["close"])

    if current_time.tzinfo is None:
        current_time_utc = current_time.tz_localize("UTC")
    else:
        current_time_utc = current_time.tz_convert("UTC")
    is_high_volume_hours = 8 <= current_time_utc.hour < 22

    market_data = MarketData(
        current_price=current_price,
        volume_15m=float(signal_row["volume"]),
        avg_volume_24h=float(signal_row.get("volume_ma", signal_row["volume"])),
        volatility_1h=float(signal_row.get("volatility_1h", signal_row.get("volatility", 0.0))),
        price_change_4h=float(signal_row.get("price_change_4h", 0.0)),
        trend_1h=str(signal_row.get("trend_1h", "NEUTRAL")),
        is_high_volume_hours=is_high_volume_hours,
        recent_news_events=False,
    )

    direction = "UP" if prob_info["prob_up"] >= threshold else "DOWN"
    confidence = prob_info["confidence"]

    evaluation_deadline = current_time.to_pydatetime() + timedelta(minutes=evaluation_window_minutes)
    log_entry_id = f"{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    base_payload = {
        "signal_timestamp": current_time.isoformat(),
        "decision_threshold": threshold,
        "prob_up": prob_info["prob_up"],
        "prob_down": prob_info["prob_down"],
        "confidence": confidence,
        "direction": direction,
        "price": current_price,
        "volume_15m": float(signal_row["volume"]),
        "avg_volume_24h": float(signal_row.get("volume_ma", signal_row["volume"])),
        "volatility_1h": float(signal_row.get("volatility_1h", signal_row.get("volatility", 0.0))),
        "is_high_volume_hours": is_high_volume_hours,
        "log_id": log_entry_id,
        "evaluation_status": "pending",
        "evaluation_deadline": evaluation_deadline.isoformat(),
    }

    pending_record = {
        "log_id": log_entry_id,
        "direction": direction,
        "entry_price": current_price,
        "deadline": evaluation_deadline,
    }
    pending_registered = False

    def log_and_register(payload: Dict[str, Any]) -> None:
        nonlocal pending_registered
        log_signal_decision(signal_log_path, payload)
        if not pending_registered:
            register_pending_evaluation(pending_evaluations, pending_record.copy())
            pending_registered = True

    if confidence < system.min_confidence:
        payload = {
            **base_payload,
            "status": "rejected",
            "stage": "pre_system_threshold",
            "reason": "confidence_below_min_confidence",
            "min_confidence": system.min_confidence,
        }
        system.last_signal_decision = payload.copy()
        log_and_register(payload)
        return

    updates = system.update_positions(current_price, current_time)
    for update in updates:
        position_id = update["position_id"]
        qty_record = position_records.get(position_id)
        qty_to_close = 0.0
        if qty_record:
            qty_to_close = qty_record["entry_qty"] * update["close_percentage"]
        if api and not dry_run and qty_to_close > 0:
            api.sell(quantity=qty_to_close)
        if qty_record:
            qty_record["entry_qty"] -= qty_to_close
            if qty_record["entry_qty"] <= 1e-8:
                position_records.pop(position_id, None)

        log_event(log_file, "EXIT", {
            "timestamp": current_time,
            "price": current_price,
            **update,
        })

    position = system.process_signal(
        direction,
        confidence,
        market_data,
        signal_time=current_time.to_pydatetime(),
    )
    decision = getattr(system, "last_signal_decision", None)
    payload: Dict[str, Any] = {}
    if isinstance(decision, dict):
        payload.update(decision)
        if "timestamp" in payload:
            payload["decision_timestamp"] = payload.pop("timestamp")

    payload.update(base_payload)

    if position:
        if direction == "DOWN":
            system._close_position(position, current_price, current_time, "SPOT_SHORT_UNSUPPORTED")
            payload.update({
                "status": "rejected",
                "reason": "spot_short_not_supported",
            })
            log_and_register(payload)
            return

        qty = compute_order_quantity(system, position, current_price, api, min_notional, direction)
        if qty <= 0:
            system._close_position(position, current_price, current_time, "EXECUTION_SKIPPED")
            payload.update({
                "status": "rejected",
                "reason": "quantity_below_minimum_or_balance",
            })
            log_and_register(payload)
            return

        executed_qty = qty
        if api and not dry_run:
            order = api.buy(quantity=qty)
            executed_qty = order.quantity
        log_event(log_file, "ENTRY", {
            "timestamp": current_time,
            "price": current_price,
            "direction": direction,
            "confidence": confidence,
            "prob_up": prob_info["prob_up"],
            "position_size_pct": position.position_size_pct,
            "position_id": position.signal_id,
            "quantity": executed_qty,
            "dry_run": dry_run,
        })
        position_records[position.signal_id] = {
            "direction": direction,
            "entry_qty": executed_qty,
        }
        payload.update({
            "status": "accepted",
            "position_id": position.signal_id,
            "position_size_pct": position.position_size_pct,
            "quantity": executed_qty,
            "dry_run": dry_run,
        })
    else:
        payload.setdefault("status", "rejected")
        payload.setdefault("reason", "rejected_without_detail")

    log_and_register(payload)


def write_summary(summary_path: Path, system: ComprehensiveTradingSystem) -> None:
    summary = system.get_performance_summary()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=float)


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()
    dry_run = not args.live if not args.dry_run else True
    if args.live:
        dry_run = False

    config = load_trading_config(args.config)
    system = ComprehensiveTradingSystem(initial_capital=20000.0, config=config)

    results_dir = args.log.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    signal_log_path = args.signal_log_dir / f"live_signal_log_{run_timestamp}.txt"
    args.signal_log_dir.mkdir(parents=True, exist_ok=True)

    testnet_flag = dry_run or os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    client = get_client(testnet=testnet_flag)
    raw_df = fetch_initial_data(client, symbol, args.interval, args.lookback_days)

    upsert_initial_data(raw_df, symbol, args.interval)

    _purge_gru_backtest_artifacts(symbol)
    cutoff = datetime.utcnow()
    try:
        retrain_info = retrain_gru_model_until(cutoff, symbol, args.interval)
        print(
            f"🤖 GRU retrained with {retrain_info['samples']} sequences "
            f"(features: {retrain_info['available_features']}, sequence={retrain_info['sequence_length']})"
        )
    except Exception as exc:
        print(f"⚠️ GRU retraining failed: {exc}")
        raise

    if not sync_signal_model_artifacts(symbol):
        raise RuntimeError("Failed to sync GRU artifacts for live trading.")

    model = GRUSignalModel(model_dir=get_gru_signal_dir(symbol))
    feature_engine = EnhancedFeatureEngine(performance_mode=True)
    raw_df = enrich_base_features(raw_df)

    last_processed_ts = None
    decision_threshold = model.best_threshold
    iteration = 0
    pending_evaluations: List[Dict[str, Any]] = []
    position_records: Dict[str, Dict[str, Any]] = {}

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log.open("a") if args.log else None
    summary_path = results_dir / "live_trading_summary.json"

    api = None
    if not dry_run:
        api = BinanceSpotTradingAPI(symbol=symbol, testnet=testnet_flag)
        print("🚀 Live spot trading enabled – orders will be sent to Binance.")
    else:
        print("🧪 Dry-run mode: no orders will be sent (testnet data only).")

    try:
        while True:
            iteration += 1
            if args.iterations and iteration > args.iterations:
                break

            if last_processed_ts is None and not raw_df.empty:
                last_processed_ts = int(raw_df["open_time"].iloc[-1])

            new_start_ms = (last_processed_ts or int(raw_df["open_time"].iloc[-1])) + 1
            new_klines, delayed_fetch = fetch_new_klines(
                client, symbol, args.interval, new_start_ms
            )

            if not new_klines.empty:
                now_ms = int(time.time() * 1000)
                new_klines = new_klines[new_klines["close_time"] <= now_ms]
                if new_klines.empty:
                    time.sleep(args.sleep)
                    continue
                upsert_initial_data(new_klines, symbol, args.interval)
                raw_df = pd.concat([raw_df, new_klines], ignore_index=True)
                raw_df = enrich_base_features(raw_df)
                raw_df = raw_df.iloc[-MAX_HISTORY_ROWS:]

                if delayed_fetch:
                    last_processed_ts = int(raw_df["open_time"].iloc[-1])
                    for row in new_klines.itertuples(index=False):
                        ts = pd.to_datetime(getattr(row, "datetime")).isoformat()
                        skip_payload = {
                            "signal_timestamp": ts,
                            "price": float(getattr(row, "close", np.nan)),
                            "status": "skipped",
                            "stage": "delayed_kline_fetch",
                            "reason": "delayed_kline_fetch",
                        }
                        log_signal_decision(signal_log_path, skip_payload)
                    print("⚠️  Delayed kline fetch resolved after retries. Skipping signal processing for this batch.")
                    time.sleep(args.sleep)
                    continue
            else:
                if delayed_fetch:
                    skip_payload = {
                        "signal_timestamp": datetime.utcnow().isoformat() + "Z",
                        "status": "skipped",
                        "stage": "delayed_kline_fetch",
                        "reason": "fetch_failed_after_retries",
                    }
                    log_signal_decision(signal_log_path, skip_payload)
                    print("⚠️  Kline fetch failed after retries; will try again next cycle.")
                    time.sleep(args.sleep)
                    continue

            feature_df = prepare_feature_dataframe(raw_df.copy(), feature_engine)
            predictions = model.predict_series(feature_df, limit=500)
            if not predictions:
                print("等待更多数据以满足序列长度要求...")
                time.sleep(args.sleep)
                continue

            processed = 0
            for idx, prob_info in predictions.items():
                row = feature_df.loc[idx]
                timestamp_dt = pd.to_datetime(row["datetime"]) if "datetime" in row else pd.to_datetime(row["open_time"], unit="ms")
                timestamp_ms = int(timestamp_dt.value // 1_000_000)
                if last_processed_ts and timestamp_ms <= last_processed_ts:
                    continue
                process_new_signal(
                    system,
                    api,
                    position_records,
                    row,
                    prob_info,
                    decision_threshold,
                    log_file,
                    signal_log_path,
                    pending_evaluations,
                    EVALUATION_WINDOW_MINUTES,
                    dry_run,
                    args.min_notional,
                )
                last_processed_ts = timestamp_ms
                processed += 1

            evaluate_pending_signals(pending_evaluations, raw_df, signal_log_path)
            write_summary(summary_path, system)

            metrics = system.get_performance_summary()
            print(f"迭代 {iteration}: Equity={metrics['portfolio_value']:.2f} WinRate={metrics['win_rate']*100:.2f}% Trades={metrics['total_trades']}")

            if processed == 0 and predictions:
                latest_idx = max(predictions.keys())
                info = predictions[latest_idx]
                latest_row = feature_df.loc[latest_idx]
                latest_price = float(latest_row["close"])
                print(
                    f"  ↳ 最新信号: time={latest_row['datetime']} price={latest_price:.2f} "
                    f"prob_up={info['prob_up']:.3f} confidence={info['confidence']:.3f}"
                )

            if args.iterations and iteration >= args.iterations:
                break

            time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("🛑 Interrupted by user. Exiting gracefully.")
    finally:
        write_summary(summary_path, system)
        if log_file:
            log_file.close()
        if api and not dry_run:
            try:
                api.cancel_all_orders()
            except Exception as exc:
                print(f"⚠️ Failed to cancel orders during shutdown: {exc}")


if __name__ == "__main__":
    main()
