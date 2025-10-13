#!/usr/bin/env python3
"""
Hyperparameter Grid Search for the trading system.

Iterates over all combinations supplied in the parameter grid,
executes the trading simulation for each, and logs performance metrics.

Usage:
    python scripts/hyperparameter_grid_search.py \
        --start-date 2025-09-01 --end-date 2025-09-30 \
        --output results/hyperparameter_tests/grid_search_log.txt

    # Optional: provide a JSON file with parameter lists
    python scripts/hyperparameter_grid_search.py --grid-config grid.json

The JSON should map parameter names (dot-separated for nested keys)
into lists of values, e.g.
{
  "min_confidence": [0.6, 0.65],
  "position_sizes.high": [0.04, 0.05]
}

⚠️ The number of runs grows multiplicatively with each list length.
Plan your grids carefully to avoid runaway searches.
"""

import argparse
import itertools
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from trading_backtest import (
    load_backtest_data,
    generate_signals_with_tracking,
    run_trading_simulation_with_tracking,
    _train_gru_and_predict,
)
from src.trading.confidence_trading_system import ComprehensiveTradingSystem

DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
    "min_confidence": [0.60, 0.65],
    "position_sizes.low": [0.015, 0.020],
    "position_sizes.medium": [0.025, 0.030],
    "position_sizes.high": [0.040, 0.050],
    "target1_multiplier": [1.5, 2.0],
    "target2_multiplier": [3.0, 3.5],
    "time_limit_max_hours": [24.0, 36.0],
}


def set_nested(config: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split('.')
    cur = config
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def load_param_grid(path: Path) -> Dict[str, List[Any]]:
    with path.open('r') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Grid configuration must be a JSON object mapping parameter names to lists")
    for key, values in data.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Parameter '{key}' must map to a non-empty list of values")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter grid search for the trading system")
    parser.add_argument('--start-date', default='2025-08-01', help='Start date (UTC) for backtest window')
    parser.add_argument('--end-date', default='2025-09-30', help='End date (UTC) for backtest window')
    parser.add_argument('--initial-capital', type=float, default=20000.0, help='Starting capital per run')
    parser.add_argument('--grid-config', type=Path, help='Path to JSON file containing parameter grid')
    parser.add_argument('--output', type=Path,
                        default=Path('results/hyperparameter_tests/grid_search_log.txt'),
                        help='Path to append grid search log')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite log file instead of appending')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress for each run')

    args = parser.parse_args()

    param_grid = load_param_grid(args.grid_config) if args.grid_config else DEFAULT_PARAM_GRID

    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    total_runs = 1
    for values in value_lists:
        total_runs *= len(values)

    if total_runs == 0:
        print("No parameter combinations to evaluate.")
        return

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = 'w' if args.overwrite else 'a'
    log_file = output_path.open(mode)

    log_file.write(f"# Hyperparameter grid search started {datetime.utcnow().isoformat()}Z\n")
    log_file.write(f"# Start: {args.start_date}, End: {args.end_date}, Runs: {total_runs}\n")
    log_file.flush()

    # Load data once and reuse across runs (columns remain constant)
    data = load_backtest_data(args.start_date, args.end_date)
    predictions_df, model_metrics = _train_gru_and_predict(args.start_date, args.end_date)
    data, _ = generate_signals_with_tracking(
        data.copy(),
        args.start_date,
        args.end_date,
        predictions=predictions_df,
        cached_metrics=model_metrics,
    )

    default_config = ComprehensiveTradingSystem.default_config()

    for run_idx, values in enumerate(itertools.product(*value_lists), start=1):
        config_override: Dict[str, Any] = {}
        for key, value in zip(keys, values):
            set_nested(config_override, key, value)

        system_config = deepcopy(default_config)
        ComprehensiveTradingSystem._deep_update(system_config, config_override)

        if args.verbose:
            print(f"Run {run_idx}/{total_runs}: {config_override}")

        trading_system, trades_log, _ = run_trading_simulation_with_tracking(
            data,
            system_config=system_config,
            initial_capital=args.initial_capital,
        )

        metrics = trading_system.get_performance_summary()

        entry = {
            "run": run_idx,
            "parameters": config_override,
            "metrics": metrics,
            "trade_events": len(trades_log),
        }

        log_file.write(json.dumps(entry, default=float) + "\n")
        log_file.flush()

    log_file.write(f"# Hyperparameter grid search completed {datetime.utcnow().isoformat()}Z\n\n")
    log_file.close()
    print(f"Grid search complete. Logged {total_runs} runs to {output_path}")


if __name__ == '__main__':
    main()
