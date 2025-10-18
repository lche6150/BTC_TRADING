#!/usr/bin/env python3
"""
Walk-forward evaluation harness that compares the default trading configuration
against best_config.json across sequential out-of-sample windows.

Example:
    PYTHONPATH=. python scripts/walkforward_evaluation.py \
        --start 2023-01-01 --end 2025-09-30 --window-days 90 --step-days 90
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRADING_SCRIPT = REPO_ROOT / "trading_backtest.py"
BEST_CONFIG = REPO_ROOT / "best_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation for trading configs")
    parser.add_argument("--start", required=True, help="Start date (UTC) for first evaluation window, e.g. 2023-01-01")
    parser.add_argument("--end", required=True, help="End date (UTC) for final evaluation window, e.g. 2025-09-30")
    parser.add_argument(
        "--window-days",
        type=int,
        default=90,
        help="Length of each evaluation window in days (default: 90)",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=90,
        help="Forward step between windows in days (default: 90)",
    )
    parser.add_argument(
        "--results-dir",
        default=REPO_ROOT / "results" / "walkforward",
        type=Path,
        help="Directory to store per-fold outputs (default: results/walkforward)",
    )
    return parser.parse_args()


def run_backtest(start_date: str, end_date: str, figure_path: Path, log_path: Path, config_path: Path | None) -> dict:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(TRADING_SCRIPT),
        "--start",
        start_date,
        "--end",
        end_date,
        "--figure",
        str(figure_path),
        "--log",
        str(log_path),
    ]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    print(f"\n➡️  Running backtest: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)

    metrics_path = figure_path.with_name(f"{figure_path.stem}_metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected metrics file not found: {metrics_path}")

    with metrics_path.open("r") as f:
        metrics = json.load(f)

    metrics["metrics_path"] = str(metrics_path)
    metrics["figure_path"] = str(figure_path)
    metrics["log_path"] = str(log_path)
    return metrics


def main() -> None:
    args = parse_args()

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")
    window = timedelta(days=args.window_days)
    step = timedelta(days=args.step_days)

    if end_ts <= start_ts:
        raise ValueError("End date must be after start date")

    results_dir = Path(args.results_dir)
    default_dir = results_dir / "default"
    best_dir = results_dir / "best_config"
    summary_rows = []

    fold_idx = 0
    current_start = start_ts

    while current_start < end_ts:
        test_end = min(current_start + window - timedelta(days=1), end_ts)
        if test_end <= current_start:
            break

        fold_idx += 1
        fold_label = f"fold_{fold_idx:02d}"
        print("\n" + "=" * 80)
        print(f"📆 Fold {fold_idx}: {current_start.date()} → {test_end.date()}")
        print("=" * 80)

        # Reset model caches so each fold retrains on the appropriate history.
        for cache_dir in [REPO_ROOT / "models" / "gru_backtest", REPO_ROOT / "models" / "gru_signal"]:
            shutil.rmtree(cache_dir, ignore_errors=True)

        # Default configuration
        default_figure = default_dir / fold_label / "Figure_1.png"
        default_log = default_dir / fold_label / "trading_log.txt"
        default_metrics = run_backtest(
            start_date=current_start.date().isoformat(),
            end_date=test_end.date().isoformat(),
            figure_path=default_figure,
            log_path=default_log,
            config_path=None,
        )
        default_metrics.update(
            {
                "fold": fold_idx,
                "config": "default",
                "test_start": current_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
            }
        )
        summary_rows.append(default_metrics)

        # Best configuration (reuse cached GRU from default run)
        best_figure = best_dir / fold_label / "Figure_1.png"
        best_log = best_dir / fold_label / "trading_log.txt"
        best_metrics = run_backtest(
            start_date=current_start.date().isoformat(),
            end_date=test_end.date().isoformat(),
            figure_path=best_figure,
            log_path=best_log,
            config_path=BEST_CONFIG,
        )
        best_metrics.update(
            {
                "fold": fold_idx,
                "config": "best_config",
                "test_start": current_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
            }
        )
        summary_rows.append(best_metrics)

        current_start += step

    if not summary_rows:
        print("No folds processed – check the supplied dates and window configuration.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / "walkforward_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📄 Walk-forward summary saved to {summary_path}")

    comparison = (
        summary_df.groupby("config")[["realized_pnl", "win_rate", "total_trades", "accuracy"]]
        .agg(["sum", "mean"])
        .round(4)
    )
    print("\n📊 Aggregated results by config:")
    print(comparison)

    pnl_by_config = summary_df.groupby("config")["realized_pnl"].sum()
    winning_config = pnl_by_config.idxmax()
    print(
        f"\n🏁 Overall winner based on cumulative realized P&L: "
        f"{winning_config} (${pnl_by_config[winning_config]:,.2f})"
    )


if __name__ == "__main__":
    main()
