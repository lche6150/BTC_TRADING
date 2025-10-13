#!/usr/bin/env python3
"""Paper-trading loop that streams live Binance data and simulates trades locally."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Ensure repository modules are importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.trading.confidence_trading_system import ComprehensiveTradingSystem, MarketData
from src.data.features import EnhancedFeatureEngine
from src.data.database import TradingDatabase
from live_trader.model_loader import GRUSignalModel
from trading_backtest import MODEL_FEATURES, retrain_gru_model_until

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "15m"
DEFAULT_LOOKBACK_DAYS = 20
MAX_HISTORY_ROWS = 6000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper trading using cached GRU signal model")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Trading symbol (default BTCUSDT)")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Kline interval (default 15m)")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help="Initial historical days to bootstrap features")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of polling iterations (0 for infinite loop)")
    parser.add_argument("--sleep", type=int, default=60,
                        help="Seconds to sleep between polling cycles (default 60)")
    parser.add_argument("--config", type=Path, default=Path("best_config.json"),
                        help="Trading system configuration JSON")
    parser.add_argument("--output", type=Path, default=Path("results/live_trading/paper_trades.log"),
                        help="Path to append paper-trade events")
    parser.add_argument("--testnet", action="store_true", help="Use Binance testnet")
    return parser.parse_args()


def load_trading_config(path: Path) -> Dict[str, any]:
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
    df['returns'] = df['close'].pct_change()
    df['volume_ma'] = df['volume'].rolling(24).mean()
    df['volatility'] = df['returns'].rolling(24).std()
    df['price_change_1h'] = df['close'].pct_change(4)
    df['price_change_4h'] = df['close'].pct_change(16)
    df['sma_short'] = df['close'].rolling(12).mean()
    df['sma_long'] = df['close'].rolling(48).mean()
    df['trend_1h'] = np.where(
        df['sma_short'] > df['sma_long'], 'BULLISH',
        np.where(df['sma_short'] < df['sma_long'], 'BEARISH', 'NEUTRAL')
    )
    return df


def prepare_feature_dataframe(df: pd.DataFrame, feature_engine: EnhancedFeatureEngine) -> pd.DataFrame:
    engineered = feature_engine.create_all_features(df, include_multiframe=True, min_periods=100)
    required_cols = [c for c in MODEL_FEATURES if c in engineered.columns]
    engineered = engineered.dropna(subset=required_cols + ['datetime', 'close', 'volume'])
    engineered = engineered.reset_index(drop=True)
    engineered['datetime'] = pd.to_datetime(engineered['datetime'])
    return engineered


def get_client(testnet: bool) -> Client:
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")
    client = Client(api_key=api_key, api_secret=api_secret, testnet=testnet)
    return client


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
    """
    Fetch recent klines with basic retry handling for transient network issues.

    Returns dataframe and a boolean flag indicating whether the call required retries.
    """
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


def log_event(log_file, event_type: str, payload: Dict[str, any]) -> None:
    if log_file is None:
        return
    entry = {"type": event_type, **payload}
    log_file.write(json.dumps(entry, default=str) + "\n")
    log_file.flush()


def log_signal_decision(signal_log_file, payload: Dict[str, any]) -> None:
    if signal_log_file is None:
        return
    signal_log_file.write(json.dumps(payload, default=str) + "\n")
    signal_log_file.flush()


def sync_signal_model_artifacts() -> bool:
    """
    Ensure paper trader model cache (models/gru_signal) exists by copying
    the latest backtest artifacts when available.
    """
    source_dir = Path("models/gru_backtest")
    target_dir = Path("models/gru_signal")

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
    """Insert fetched historical klines into the SQLite database for training reuse."""
    if df.empty:
        return

    records = []
    for row in df.itertuples(index=False):
        records.append({
            'timestamp': int(row.open_time),
            'open': float(row.open),
            'high': float(row.high),
            'low': float(row.low),
            'close': float(row.close),
            'volume': float(row.volume),
            'close_time': int(row.close_time),
            'quote_asset_volume': float(row.quote_asset_volume),
            'number_of_trades': int(row.number_of_trades),
            'taker_buy_base_asset_volume': float(row.taker_buy_base),
            'taker_buy_quote_asset_volume': float(row.taker_buy_quote),
        })

    if not records:
        return

    db = TradingDatabase()
    db.insert_ohlcv_data(records, symbol, interval)


def process_new_signal(system: ComprehensiveTradingSystem,
                        signal_row: pd.Series,
                        prob_info: Dict[str, float],
                        threshold: float,
                        log_file,
                        signal_log_file) -> None:
    current_time = pd.to_datetime(signal_row['datetime'])
    current_price = float(signal_row['close'])

    if current_time.tzinfo is None:
        current_time_utc = current_time.tz_localize('UTC')
    else:
        current_time_utc = current_time.tz_convert('UTC')
    is_high_volume_hours = 8 <= current_time_utc.hour < 22

    market_data = MarketData(
        current_price=current_price,
        volume_15m=float(signal_row['volume']),
        avg_volume_24h=float(signal_row.get('volume_ma', signal_row['volume'])),
        volatility_1h=float(signal_row.get('volatility_1h', signal_row.get('volatility', 0.0))),
        price_change_4h=float(signal_row.get('price_change_4h', 0.0)),
        trend_1h=str(signal_row.get('trend_1h', 'NEUTRAL')),
        is_high_volume_hours=is_high_volume_hours,
        recent_news_events=False,
    )

    direction = 'UP' if prob_info['prob_up'] >= threshold else 'DOWN'
    confidence = prob_info['confidence']

    base_payload = {
        "signal_timestamp": current_time.isoformat(),
        "decision_threshold": threshold,
        "prob_up": prob_info['prob_up'],
        "prob_down": prob_info['prob_down'],
        "confidence": confidence,
        "direction": direction,
        "price": current_price,
        "volume_15m": float(signal_row['volume']),
        "avg_volume_24h": float(signal_row.get('volume_ma', signal_row['volume'])),
        "volatility_1h": float(signal_row.get('volatility_1h', signal_row.get('volatility', 0.0))),
        "is_high_volume_hours": is_high_volume_hours,
    }

    if confidence < system.min_confidence:
        payload = {
            **base_payload,
            "status": "rejected",
            "stage": "pre_system_threshold",
            "reason": "confidence_below_min_confidence",
            "min_confidence": system.min_confidence,
        }
        system.last_signal_decision = payload.copy()
        log_signal_decision(signal_log_file, payload)
        return

    # Update existing positions first with the latest price
    updates = system.update_positions(current_price, current_time)
    for update in updates:
        print(f"[EXIT] {update}")
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
    payload = {}
    if isinstance(decision, dict):
        payload.update(decision)
        if "timestamp" in payload:
            payload["decision_timestamp"] = payload.pop("timestamp")

    payload.update(base_payload)

    if position:
        print(f"[ENTRY] {position.signal_id} {direction} size={position.position_size_pct:.2%} price={current_price:.2f}")
        log_event(log_file, "ENTRY", {
            "timestamp": current_time,
            "price": current_price,
            "direction": direction,
            "confidence": confidence,
            "prob_up": prob_info['prob_up'],
            "position_size_pct": position.position_size_pct,
            "position_id": position.signal_id,
        })
        payload.update({
            "status": "accepted",
            "position_id": position.signal_id,
            "position_size_pct": position.position_size_pct,
        })
    else:
        payload.setdefault("status", "rejected")
        payload.setdefault("reason", "rejected_without_detail")

    log_signal_decision(signal_log_file, payload)


def main() -> None:
    args = parse_args()

    config = load_trading_config(args.config)
    system = ComprehensiveTradingSystem(initial_capital=20000.0, config=config)

    results_dir = args.output.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    signal_log_path = results_dir / f"paper_log_{run_timestamp}.txt"
    signal_log_file = signal_log_path.open('a')
    print(f"📝 Signal log: {signal_log_path}")

    client = get_client(args.testnet)
    raw_df = fetch_initial_data(client, args.symbol, args.interval, args.lookback_days)

    # Store fetched klines in local database so retraining can access the latest data
    upsert_initial_data(raw_df, args.symbol, args.interval)

    # Retrain GRU using all data available up to now before starting live loop
    cutoff = datetime.utcnow()
    try:
        retrain_info = retrain_gru_model_until(cutoff, args.symbol, args.interval)
        print(
            f"🤖 GRU retrained with {retrain_info['samples']} sequences "
            f"(features: {retrain_info['available_features']}, sequence={retrain_info['sequence_length']})"
        )
    except Exception as exc:
        print(f"⚠️ GRU retraining skipped due to: {exc}\n   Using previously cached model artifacts.")

    if sync_signal_model_artifacts():
        print("📦 Updated paper trader model cache (models/gru_signal) with latest artifacts.")
    else:
        signal_model_path = Path("models/gru_signal/model.keras")
        if not signal_model_path.exists():
            print("⚠️ Paper trader model cache not found. Run a backtest or copy artifacts to models/gru_signal.")

    model = GRUSignalModel()
    feature_engine = EnhancedFeatureEngine(performance_mode=True)
    raw_df = enrich_base_features(raw_df)

    last_processed_ts = None
    decision_threshold = model.best_threshold
    iteration = 0

    log_file = args.output.open('a') if args.output else None

    try:
        while True:
            iteration += 1

            if args.iterations and iteration > args.iterations:
                break

            if last_processed_ts is None and not raw_df.empty:
                last_processed_ts = int(raw_df['open_time'].iloc[-1])

            new_start_ms = (last_processed_ts or int(raw_df['open_time'].iloc[-1])) + 1
            new_klines, delayed_fetch = fetch_new_klines(
                client, args.symbol, args.interval, new_start_ms
            )

            if not new_klines.empty:
                now_ms = int(time.time() * 1000)
                new_klines = new_klines[new_klines['close_time'] <= now_ms]
                if new_klines.empty:
                    time.sleep(args.sleep)
                    continue
                upsert_initial_data(new_klines, args.symbol, args.interval)
                raw_df = pd.concat([raw_df, new_klines], ignore_index=True)
                raw_df = enrich_base_features(raw_df)
                raw_df = raw_df.iloc[-MAX_HISTORY_ROWS:]

                if delayed_fetch:
                    last_processed_ts = int(raw_df['open_time'].iloc[-1])
                    print("⚠️  Delayed kline fetch resolved after retries. "
                          "Skipping signal processing for this batch.")
                    time.sleep(args.sleep)
                    continue
            else:
                if delayed_fetch:
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
                timestamp_dt = pd.to_datetime(row['datetime']) if 'datetime' in row else pd.to_datetime(row['open_time'], unit='ms')
                timestamp_ms = int(timestamp_dt.value // 1_000_000)
                if last_processed_ts and timestamp_ms <= last_processed_ts:
                    continue
                process_new_signal(system, row, prob_info, decision_threshold, log_file, signal_log_file)
                last_processed_ts = timestamp_ms
                processed += 1

            metrics = system.get_performance_summary()
            print(f"迭代 {iteration}: Equity={metrics['portfolio_value']:.2f} WinRate={metrics['win_rate']*100:.2f}% Trades={metrics['total_trades']}")

            if processed == 0 and predictions:
                latest_idx = max(predictions.keys())
                info = predictions[latest_idx]
                latest_row = feature_df.loc[latest_idx]
                latest_price = float(latest_row['close'])
                print(
                    f"  ↳ 最新信号: time={latest_row['datetime']} price={latest_price:.2f} "
                    f"prob_up={info['prob_up']:.3f} confidence={info['confidence']:.3f}"
                )

            if args.iterations and iteration >= args.iterations:
                break

            time.sleep(args.sleep)
    finally:
        if log_file:
            log_file.close()
        if signal_log_file:
            signal_log_file.close()

    summary_path = results_dir / "paper_trading_summary.json"
    with summary_path.open('w') as f:
        json.dump(system.get_performance_summary(), f, indent=2, default=float)
    print(f"🏁 Paper trading completed. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
