#!/usr/bin/env python3
"""
Compare backtest and paper-trader style simulations over the same window, model, and config.

Usage:
    PYTHONPATH=. python scripts/compare_backtest_paper.py \
        --start 2025-10-20 --end 2025-11-01 \
        --config best_config.json \
        --artifacts models/gru_signal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd

from trading_backtest import (
    DEFAULT_SYMBOL,
    MODEL_FEATURES,
    create_trading_log,
    load_backtest_data,
    generate_signals_with_tracking,
    run_trading_simulation_with_tracking,
)
from src.trading.confidence_trading_system import ComprehensiveTradingSystem, MarketData
from src.utils.artifacts import get_gru_signal_dir
from src.data.features import EnhancedFeatureEngine
from live_trader.model_loader import GRUSignalModel



def run_offline_paper_simulation(
    data: pd.DataFrame,
    decision_threshold: float,
    system_config: Dict | None,
    start_dt: pd.Timestamp,
    artifacts_path: Path,
    initial_capital: float = 20000.0,
) -> Tuple[ComprehensiveTradingSystem, List[Dict], List[Dict]]:
    """Stream through the dataframe bar-by-bar to mirror the paper trader workflow."""
    system = ComprehensiveTradingSystem(initial_capital=initial_capital, config=system_config)
    feature_engine = EnhancedFeatureEngine(performance_mode=True)
    model = GRUSignalModel(model_dir=artifacts_path)

    trades_log: List[Dict] = []
    signal_log: List[Dict] = []

    feature_df = feature_engine.create_all_features(data.copy(), include_multiframe=True, min_periods=100)
    required_cols = [c for c in MODEL_FEATURES if c in feature_df.columns]
    if not required_cols:
        return system, trades_log, signal_log

    feature_df = feature_df.dropna(subset=required_cols + ["datetime", "close", "volume"]).reset_index(drop=True)
    feature_df["datetime"] = pd.to_datetime(feature_df["datetime"])

    predictions = model.predict_series(feature_df, limit=len(feature_df))
    if not predictions:
        return system, trades_log, signal_log

    for idx, feature_row in feature_df.iterrows():
        prob_info = predictions.get(idx)
        if not prob_info:
            continue

        signal_time = pd.to_datetime(feature_row["datetime"])
        if signal_time < start_dt:
            continue

        current_price = float(feature_row["close"])

        updates = system.update_positions(current_price, signal_time)
        for update in updates:
            trades_log.append({
                "timestamp": signal_time,
                "action": "CLOSE",
                "reason": update["exit_reason"],
                "price": current_price,
                "pnl": update["realized_pnl"],
                "exit_fee": update.get("exit_fee", 0.0),
                "position_id": update["position_id"],
            })

        prob_up = float(prob_info["prob_up"])
        confidence = float(prob_info["confidence"])
        direction = "UP" if prob_up >= decision_threshold else "DOWN"

        signal_record = {
            "timestamp": signal_time,
            "price": float(feature_row["close"]),
            "prob_up": prob_up,
            "prob_down": 1.0 - prob_up,
            "confidence": confidence,
            "direction": direction,
        }

        if confidence < system.min_confidence:
            signal_record.update({
                "status": "rejected",
                "stage": "pre_system_threshold",
                "reason": "confidence_below_min_confidence",
                "min_confidence": system.min_confidence,
            })
            signal_log.append(signal_record)
            continue

        current_time_utc = signal_time.tz_localize("UTC") if signal_time.tzinfo is None else signal_time.tz_convert("UTC")
        market_data = MarketData(
            current_price=float(feature_row["close"]),
            volume_15m=float(feature_row["volume"]),
            avg_volume_24h=float(feature_row.get("volume_ma", feature_row["volume"])),
            volatility_1h=float(feature_row.get("volatility", 0.02)),
            price_change_4h=float(feature_row.get("price_change_4h", 0.0)),
            trend_1h=str(feature_row.get("trend_1h", "NEUTRAL")),
            is_high_volume_hours=8 <= current_time_utc.hour < 22,
            recent_news_events=False,
        )

        position = system.process_signal(
            direction=direction,
            base_confidence=confidence,
            market_data=market_data,
            signal_time=signal_time.to_pydatetime(),
        )
        decision = getattr(system, "last_signal_decision", None)

        if position:
            signal_record.update({
                "status": "accepted",
                "stage": "position_opened",
                "position_id": position.signal_id,
            })
            if isinstance(decision, dict):
                signal_record.update({k: v for k, v in decision.items() if k not in {"status", "timestamp"}})
            trades_log.append({
                "timestamp": signal_time,
                "action": "OPEN",
                "direction": direction,
                "price": float(feature_row["close"]),
                "confidence": confidence,
                "prob_up": prob_up,
                "position_size_pct": position.position_size_pct,
                "position_value": position.position_value,
                "entry_fee": position.entry_fee,
                "position_id": position.signal_id,
                "stop_loss": getattr(position, "stop_loss", None),
                "stop_pct": getattr(position, "stop_pct", None),
                "target1_price": getattr(position, "target_1", None),
                "target1_pct": getattr(position, "target1_pct", None),
                "target2_price": getattr(position, "target_2", None),
                "target2_pct": getattr(position, "target2_pct", None),
            })
        else:
            signal_record.update({
                "status": "rejected",
                "stage": decision.get("stage") if isinstance(decision, dict) else None,
                "reason": decision.get("reason") if isinstance(decision, dict) else None,
            })

        signal_log.append(signal_record)
    if system.active_positions:
        final_time = data["datetime"].iloc[-1] + pd.Timedelta(minutes=15)
        final_price = data["close"].iloc[-1]
        for position in system.active_positions.copy():
            system._close_position(position, final_price, final_time, "SESSION_END")
            trades_log.append({
                "timestamp": final_time,
                "action": "CLOSE",
                "reason": "SESSION_END",
                "price": final_price,
                "pnl": position.realized_pnl,
                "exit_fee": 0.0,
                "position_id": position.signal_id,
            })

    return system, trades_log, signal_log


def compare_summaries(backtest_summary: Dict, paper_summary: Dict) -> Tuple[bool, Dict]:
    """
    Compare key metrics with a small tolerance.
    """
    keys = [
        "portfolio_value",
        "realized_pnl",
        "total_trades",
        "winning_trades",
        "win_rate",
    ]
    diffs = {}
    match = True

    for key in keys:
        bt_val = backtest_summary.get(key)
        paper_val = paper_summary.get(key)
        if isinstance(bt_val, float) and isinstance(paper_val, float):
            if not np.isclose(bt_val, paper_val, atol=1e-6):
                match = False
                diffs[key] = (bt_val, paper_val)
        else:
            if bt_val != paper_val:
                match = False
                diffs[key] = (bt_val, paper_val)
    return match, diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare backtest vs paper-style replay")
    parser.add_argument("--start", required=True, help="Start date (UTC) e.g. 2025-10-20")
    parser.add_argument("--end", required=True, help="End date (UTC) e.g. 2025-11-01")
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help="Trading symbol (default BTCUSDT)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("best_config.json"),
        help="Trading system config JSON",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Directory containing model.keras/scaler.pkl/meta.json (defaults to symbol-specific path)",
    )
    parser.add_argument(
        "--paper-lookback-days",
        type=int,
        default=5,
        help="Number of days of additional history to preload for the paper-style replay",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store comparison artifacts (defaults to results/compare_backtest_paper/...)",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    artifacts_path = args.artifacts if args.artifacts else get_gru_signal_dir(symbol)
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_path}")
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with args.config.open("r") as f:
        system_config = json.load(f)

    comparison_start_dt = pd.to_datetime(args.start)
    lookback_start_dt = (comparison_start_dt - pd.Timedelta(days=args.paper_lookback_days)).strftime("%Y-%m-%d")

    # Load market data once with the requested lookback so both paths share identical history
    full_data = load_backtest_data(lookback_start_dt, args.end, symbol=symbol)
    comparison_mask = full_data["datetime"] >= comparison_start_dt
    comparison_data = full_data.loc[comparison_mask].reset_index(drop=True)

    # Generate predictions for the backtest path using the provided artifacts (no retraining)
    merged_data, model_metrics = generate_signals_with_tracking(
        full_data.copy(),
        args.start,
        args.end,
        symbol=symbol,
        artifacts_path=artifacts_path,
    )
    merged_data = merged_data.loc[comparison_mask].reset_index(drop=True)

    out_dir = (
        args.output_dir
        if args.output_dir
        else Path("results/compare_backtest_paper")
        / symbol.lower()
        / f"{args.start.replace('-', '')}_to_{args.end.replace('-', '')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    backtest_dir = out_dir / "backtest"
    paper_dir = out_dir / "paper"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Run the standard backtest simulation
    backtest_system, backtest_trades_log, backtest_signals = run_trading_simulation_with_tracking(
        merged_data.copy(),
        system_config=system_config,
    )
    backtest_summary = backtest_system.get_performance_summary()

    model = GRUSignalModel(model_dir=artifacts_path)
    decision_threshold = model.best_threshold

    paper_system, paper_trades_log, paper_signals = run_offline_paper_simulation(
        full_data.copy(),
        decision_threshold=decision_threshold,
        system_config=system_config,
        start_dt=comparison_start_dt,
        artifacts_path=artifacts_path,
    )
    paper_summary = paper_system.get_performance_summary()

    match, diffs = compare_summaries(backtest_summary, paper_summary)
    print("Backtest summary:", json.dumps(backtest_summary, indent=2))
    print("Paper-style summary:", json.dumps(paper_summary, indent=2))

    def filter_records(records: List[Dict]) -> List[Dict]:
        filtered = []
        for record in records:
            ts = record.get("timestamp")
            if ts is None:
                continue
            if not isinstance(ts, pd.Timestamp):
                ts = pd.to_datetime(ts)
            if ts >= comparison_start_dt:
                filtered.append(record)
        return filtered

    paper_trades_log = filter_records(paper_trades_log)
    paper_signals = filter_records(paper_signals)

    def save_trading_artifacts(log_dir: Path, trades_log: List[Dict], signals: List[Dict], data_frame: pd.DataFrame, prefix: str):
        trade_pairs: Dict[str, Dict[str, List[Dict]]] = {}
        for trade in trades_log:
            pid = trade.get("position_id")
            if not pid:
                continue
            bucket = trade_pairs.setdefault(pid, {"open": None, "closes": []})
            if trade["action"] == "OPEN":
                bucket["open"] = trade
            elif trade["action"] == "CLOSE":
                bucket["closes"].append(trade)
        log_path = log_dir / f"{prefix}_trading_log.txt"
        create_trading_log(trade_pairs, data_frame, args.start, args.end, save_path=log_path)

        signal_path = log_dir / f"{prefix}_signal_log.jsonl"
        with signal_path.open("w") as f:
            for record in signals:
                serializable = {}
                for key, value in record.items():
                    if isinstance(value, pd.Timestamp):
                        serializable[key] = value.isoformat()
                    else:
                        serializable[key] = value
                f.write(json.dumps(serializable, default=str) + "\n")

    save_trading_artifacts(backtest_dir, backtest_trades_log, backtest_signals, comparison_data, "backtest")
    save_trading_artifacts(paper_dir, paper_trades_log, paper_signals, comparison_data, "paper")

    def save_summary(path: Path, summary: Dict):
        with path.open("w") as f:
            json.dump(summary, f, indent=2, default=float)

    save_summary(backtest_dir / "summary.json", backtest_summary)
    save_summary(paper_dir / "summary.json", paper_summary)

    if match:
        print("✅ Summaries match. Backtest and paper replay are consistent.")
    else:
        print("❌ Summaries differ:")
        for key, (bt_val, paper_val) in diffs.items():
            print(f"  {key}: backtest={bt_val}, paper={paper_val}")

    comparison_path = out_dir / "comparison.json"
    comparison_payload = {
        "match": match,
        "differences": {k: {"backtest": bt, "paper": pv} for k, (bt, pv) in diffs.items()},
    }
    with comparison_path.open("w") as f:
        json.dump(comparison_payload, f, indent=2, default=float)


if __name__ == "__main__":
    main()
