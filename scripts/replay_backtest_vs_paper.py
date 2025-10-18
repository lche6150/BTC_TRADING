#!/usr/bin/env python3
"""
Replay the paper-trader processing loop against the same feature feed used in a backtest,
so we can verify that both engines produce identical trade decisions.

Usage:
    PYTHONPATH=. python scripts/replay_backtest_vs_paper.py \
        --features results/replay_feed_20250101_20251231.parquet \
        --backtest-log results/backtests/20250101_to_20251231/trading_log.txt \
        --config best_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from live_trader.model_loader import GRUSignalModel
from src.trading.confidence_trading_system import ComprehensiveTradingSystem, MarketData


def load_backtest_trades(backtest_log: Path) -> pd.DataFrame:
    """Parse the backtest trading log into a normalized DataFrame of entry/exit events."""
    if not backtest_log.exists():
        raise FileNotFoundError(f"Backtest log not found: {backtest_log}")

    lines = backtest_log.read_text().splitlines()
    trade_records = []

    current_trade = None
    base_year = None
    pending_entry = None

    import re

    date_re = re.compile(r"(\d{2}/\d{2}) (\d{2}:\d{2})")

    for line in lines:
        if line.startswith("TRADING LOG -"):
            try:
                date_part = line.split("-")[1].split(" (")[0].strip().split(" - ")[0]
                base_year = pd.to_datetime(date_part).year
            except Exception:
                base_year = None
            continue

        if line.startswith("Trade "):
            current_trade = line.split(":")[0].split()[1]
            pending_entry = None
            continue

        if "Entry Time" in line:
            match = date_re.search(line)
            if not match:
                continue
            month_day, time_str = match.groups()
            year = base_year or 2025
            timestamp = pd.to_datetime(f"{year}-{month_day.replace('/', '-')} {time_str}", format="%Y-%m-%d %H:%M", utc=True)
            pending_entry = {
                "timestamp": timestamp,
                "action": "ENTRY",
                "direction": None,
                "confidence": None,
                "reason": None,
                "exit_reason": None,
                "pnl": None,
            }
            continue

        if "Direction" in line and pending_entry is not None:
            pending_entry["direction"] = line.split(":")[1].strip()
            continue

        if "Confidence" in line and pending_entry is not None:
            value = line.split(":")[1].strip().replace("%", "")
            try:
                pending_entry["confidence"] = float(value) / 100.0
            except ValueError:
                pending_entry["confidence"] = None
            continue

        if line.strip().startswith("Exit"):
            match = date_re.search(line)
            if not match:
                continue
            month_day, time_str = match.groups()
            year = base_year or 2025
            timestamp = pd.to_datetime(f"{year}-{month_day.replace('/', '-')} {time_str}", format="%Y-%m-%d %H:%M", utc=True)
            parts = line.split("|")
            reason = parts[1].split(":", 1)[1].strip() if len(parts) > 1 else ""
            pnl = float(parts[2].split("$")[1].replace(",", "")) if len(parts) > 2 else 0.0

            if pending_entry is not None:
                trade_records.append(pending_entry)
                pending_entry = None

            trade_records.append(
                {
                    "timestamp": timestamp,
                    "action": "EXIT",
                    "direction": None,
                    "confidence": None,
                    "reason": None,
                    "exit_reason": reason,
                    "pnl": pnl,
                }
            )

    if pending_entry is not None:
        trade_records.append(pending_entry)

    return pd.DataFrame(trade_records).sort_values("timestamp").reset_index(drop=True)


def load_backtest_signal_log(signal_log: Path) -> pd.DataFrame:
    """Parse backtest signal_log.jsonl if available."""
    if not signal_log or not signal_log.exists():
        return pd.DataFrame()

    records = []
    with signal_log.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_key = obj.get("signal_timestamp") or obj.get("timestamp")
            if not ts_key:
                continue
            timestamp = pd.to_datetime(ts_key, utc=True)
            status = obj.get("status")
            if not status:
                status = "ACCEPTED" if obj.get("executed") else "REJECTED"
            records.append(
                {
                    "timestamp": timestamp,
                    "status_backtest": str(status).upper(),
                    "reason_backtest": obj.get("reason"),
                    "direction_backtest": obj.get("direction"),
                    "confidence_backtest": obj.get("confidence"),
                }
            )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)


def replay_signals(features_path: Path, config_path: Path, output_log: Path):
    """Run the ComprehensiveTradingSystem using the same feature rows. Returns (decisions_df, entries_df, exits_df)."""
    with config_path.open() as f:
        config = json.load(f)

    system = ComprehensiveTradingSystem(initial_capital=20000.0, config=config)
    model = GRUSignalModel()
    threshold = model.best_threshold

    df = pd.read_parquet(features_path).sort_values("datetime").reset_index(drop=True)
    preds = model.predict_series(df, limit=len(df))
    if not preds:
        raise RuntimeError("Unable to generate predictions for replay feed.")
    pred_df = (
        pd.DataFrame.from_dict(preds, orient="index")
        .rename_axis("index")
        .reset_index()
    )

    merged = df.reset_index().merge(pred_df, on="index", how="inner")
    merged["timestamp"] = pd.to_datetime(merged["datetime"], utc=True)

    decisions = []
    entries = []
    exits = []
    last_processed_ts = None

    for row in merged.itertuples():
        timestamp = row.timestamp
        if last_processed_ts and timestamp <= last_processed_ts:
            continue

        market_data = MarketData(
            current_price=row.close,
            volume_15m=row.volume,
            avg_volume_24h=row.volume_ma if not np.isnan(row.volume_ma) else row.volume,
            volatility_1h=row.volatility if not np.isnan(row.volatility) else 0.02,
            price_change_4h=row.price_change_4h if not np.isnan(row.price_change_4h) else 0.0,
            trend_1h=row.trend_1h,
            is_high_volume_hours=8 <= timestamp.hour < 22,
            recent_news_events=False,
        )

        updates = system.update_positions(row.close, timestamp)
        for update in updates:
            exits.append(
                {
                    "timestamp": timestamp,
                    "exit_reason_replay": update["exit_reason"],
                    "pnl_replay": update["realized_pnl"],
                    "position_id": update.get("position_id"),
                }
            )

        direction = "UP" if row.prob_up >= threshold else "DOWN"
        confidence = row.confidence

        position = system.process_signal(direction, confidence, market_data, signal_time=timestamp)
        decision = getattr(system, "last_signal_decision", {})
        status = decision.get("status", "rejected").upper()

        decisions.append(
            {
                "timestamp": timestamp,
                "status_replay": status,
                "direction_replay": direction,
                "confidence_replay": confidence,
                "reason_replay": decision.get("reason"),
                "prob_up": row.prob_up,
                "prob_down": getattr(row, "prob_down", 1.0 - row.prob_up),
            }
        )

        if position:
            position.entry_time = position.entry_time.replace(tzinfo=timestamp.tzinfo) if position.entry_time.tzinfo is None else position.entry_time
            entries.append(
                {
                    "timestamp": timestamp,
                    "entry_direction_replay": direction,
                    "entry_confidence_replay": confidence,
                    "entry_price_replay": row.close,
                    "position_id": position.signal_id,
                }
            )

        last_processed_ts = timestamp

    summary = system.get_performance_summary()
    with output_log.open("w") as f:
        json.dump(summary, f, indent=2)
    return pd.DataFrame(decisions), pd.DataFrame(entries), pd.DataFrame(exits)


def main():
    parser = argparse.ArgumentParser(description="Replay paper trader on saved feature feed")
    parser.add_argument("--features", type=Path, required=True, help="Path to feature parquet file")
    parser.add_argument("--backtest-log", type=Path, required=True, help="Backtest trading_log.txt")
    parser.add_argument("--config", type=Path, default=Path("best_config.json"), help="Config JSON")
    parser.add_argument("--backtest-signals", type=Path, help="Optional backtest signal_log.jsonl for status comparison")
    parser.add_argument("--output", type=Path, default=Path("results/replay_results"), help="Output directory")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    replay_decisions, replay_entries, replay_exits = replay_signals(
        args.features, args.config, args.output / "replay_summary.json"
    )
    backtest_entries = load_backtest_trades(args.backtest_log)
    backtest_signals = load_backtest_signal_log(args.backtest_signals) if args.backtest_signals else pd.DataFrame()

    # Build replay frame
    replay_frame = replay_decisions.copy()
    if not replay_frame.empty:
        replay_frame["action_replay"] = replay_frame.pop("status_replay")
        replay_frame["exit_reason_replay"] = pd.NA
    else:
        replay_frame = pd.DataFrame(columns=["timestamp", "action_replay", "direction_replay", "confidence_replay", "reason_replay", "exit_reason_replay"])

    if not replay_exits.empty:
        exit_rows = replay_exits.copy()
        exit_rows["action_replay"] = "EXIT"
        exit_rows["direction_replay"] = pd.NA
        exit_rows["confidence_replay"] = pd.NA
        exit_rows["reason_replay"] = pd.NA
        exit_rows["exit_reason_replay"] = exit_rows.pop("exit_reason_replay")
        replay_frame = pd.concat([replay_frame, exit_rows], ignore_index=True)

    replay_cols = ["timestamp", "action_replay", "direction_replay", "confidence_replay", "reason_replay", "exit_reason_replay"]
    replay_frame = replay_frame.reindex(columns=replay_cols)
    replay_frame = replay_frame.sort_values("timestamp")

    # Build backtest frame
    if not backtest_signals.empty:
        backtest_frame = backtest_signals.rename(
            columns={
                "status_backtest": "action_backtest",
                "direction_backtest": "direction_backtest",
                "confidence_backtest": "confidence_backtest",
                "reason_backtest": "reason_backtest",
            }
        )
    else:
        backtest_frame = pd.DataFrame(columns=["timestamp", "action_backtest", "direction_backtest", "confidence_backtest", "reason_backtest"])

    backtest_frame["exit_reason_backtest"] = pd.NA
    if not backtest_entries.empty:
        exit_rows = backtest_entries[backtest_entries["action"] == "EXIT"].copy()
        exit_rows["action_backtest"] = "EXIT"
        exit_rows["direction_backtest"] = pd.NA
        exit_rows["confidence_backtest"] = pd.NA
        exit_rows.rename(columns={"exit_reason": "exit_reason_backtest"}, inplace=True)
        exit_rows = exit_rows[["timestamp", "action_backtest", "direction_backtest", "confidence_backtest", "exit_reason_backtest"]]
        frames = [df for df in (backtest_frame, exit_rows) if not df.empty]
        backtest_frame = pd.concat(frames, ignore_index=True) if frames else backtest_frame

    backtest_cols = ["timestamp", "action_backtest", "direction_backtest", "confidence_backtest", "reason_backtest", "exit_reason_backtest"]
    backtest_frame = backtest_frame.reindex(columns=backtest_cols)
    backtest_frame = backtest_frame.sort_values("timestamp")

    comparison = replay_frame.merge(backtest_frame, on="timestamp", how="outer")
    comparison.sort_values("timestamp").to_csv(args.output / "replay_vs_backtest.csv", index=False)
    print(f"Comparison saved to {args.output / 'replay_vs_backtest.csv'}")

    # Summaries
    replay_pnls = replay_exits["pnl_replay"] if not replay_exits.empty and "pnl_replay" in replay_exits else pd.Series(dtype=float)
    replay_summary = {
        "actions": replay_frame["action_replay"].value_counts(dropna=True).to_dict(),
        "total_exits": int(replay_frame["action_replay"].eq("EXIT").sum()),
        "total_realized_pnl": float(replay_pnls.sum()) if not replay_pnls.empty else 0.0,
        "win_rate": float((replay_pnls > 0).mean()) if not replay_pnls.empty else None,
    }
    with (args.output / "replay_event_summary.json").open("w") as f:
        json.dump(replay_summary, f, indent=2)

    exit_pnls = backtest_entries.loc[backtest_entries["action"] == "EXIT", "pnl"] if not backtest_entries.empty else pd.Series(dtype=float)
    backtest_summary = {
        "actions": backtest_frame["action_backtest"].value_counts(dropna=True).to_dict(),
        "total_exits": int(backtest_frame["action_backtest"].eq("EXIT").sum()),
        "total_realized_pnl": float(exit_pnls.sum()) if not exit_pnls.empty else 0.0,
        "win_rate": float((exit_pnls > 0).mean()) if not exit_pnls.empty else None,
    }
    with (args.output / "backtest_event_summary.json").open("w") as f:
        json.dump(backtest_summary, f, indent=2)

    # Write replay trade log
    trade_frames = []
    if not replay_entries.empty:
        entry_log = replay_entries.rename(
            columns={
                "entry_direction_replay": "direction",
                "entry_confidence_replay": "confidence",
                "entry_price_replay": "price",
            }
        )
        entry_log = entry_log.assign(action="ENTRY", exit_reason=pd.NA, pnl=pd.NA)
        trade_frames.append(entry_log[["timestamp", "action", "direction", "confidence", "price", "exit_reason", "pnl"]])
    if not replay_exits.empty:
        exit_log = replay_exits.rename(columns={"exit_reason_replay": "exit_reason", "pnl_replay": "pnl"})
        exit_log = exit_log.assign(action="EXIT", direction=pd.NA, confidence=pd.NA, price=pd.NA)
        trade_frames.append(exit_log[["timestamp", "action", "direction", "confidence", "price", "exit_reason", "pnl"]])
    if trade_frames:
        replay_trade_log = pd.concat(trade_frames, ignore_index=True).sort_values("timestamp")
        replay_trade_log.to_csv(args.output / "replay_trade_log.csv", index=False)

    # Write backtest trade log (entries, exits, and signal statuses)
    backtest_trade_frames = []
    if not backtest_entries.empty:
        entry_log = backtest_entries[backtest_entries["action"] == "ENTRY"].copy()
        if not entry_log.empty:
            entry_log = entry_log.assign(reason=pd.NA, exit_reason=pd.NA, pnl=pd.NA)[
                ["timestamp", "action", "direction", "confidence", "reason", "exit_reason", "pnl"]
            ]
            backtest_trade_frames.append(entry_log)

        exit_log = backtest_entries[backtest_entries["action"] == "EXIT"].copy()
        if not exit_log.empty:
            exit_log = exit_log.assign(direction=pd.NA, confidence=pd.NA, reason=pd.NA)[
                ["timestamp", "action", "direction", "confidence", "reason", "exit_reason", "pnl"]
            ]
            backtest_trade_frames.append(exit_log)

    if not backtest_signals.empty:
        status_log = backtest_signals.rename(
            columns={
                "status_backtest": "action",
                "direction_backtest": "direction",
                "confidence_backtest": "confidence",
                "reason_backtest": "reason",
            }
        )
        status_log = status_log.assign(exit_reason=pd.NA, pnl=pd.NA)[
            ["timestamp", "action", "direction", "confidence", "reason", "exit_reason", "pnl"]
        ]
        backtest_trade_frames.append(status_log)

    if backtest_trade_frames:
        backtest_trade_log = pd.concat(backtest_trade_frames, ignore_index=True).sort_values("timestamp")
        backtest_trade_log.to_csv(args.output / "backtest_trade_log.csv", index=False)


if __name__ == "__main__":
    main()
