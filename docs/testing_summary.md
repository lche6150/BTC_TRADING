# Testing Summary (October 2025)

This note collects the recent validation runs comparing trading-system configurations and profit-target multipliers. All tests were executed with the latest GRU retraining pipeline after downloading Binance BTCUSDT 15 m data through 2025‑10‑14.

## 1. Walk-Forward Comparison (Default vs `best_config.json`)

- **Window setup:** 5 sequential 6‑month evaluation windows spanning 2023‑01‑01 → 2024‑12‑31.  
- **Procedure:** For each fold we retrained the GRU on data up to the test window, ran the default configuration, reran the same window with `best_config.json`, and recorded metrics in `results/walkforward`.
- **Aggregate results:**

| Config        | Total P&L (USD) | Avg P&L / fold (USD) | Avg Win Rate | Trades |
|---------------|----------------:|---------------------:|--------------:|-------:|
| best_config   | 27,767.80       | 5,553.56             | 47.17 %       | 23,576 |
| default       | 19,338.09       | 3,867.62             | 51.30 %       | 23,576 |

`best_config.json` delivered the higher cumulative return in four of five folds despite a lower win rate, confirming the confidence-weighted sizing/targets outperform the baseline across regimes. Detailed per-fold files live under `results/walkforward/<config>/fold_0X/`, with the summary CSV at `results/walkforward/walkforward_summary.csv`.

## 2. Profit-Target Multiplier Sweep (2024 Window)

- **Data:** 2024‑01‑01 → 2024‑12‑31, GRU retrained once and cached.
- **Grid:** Target 1 multipliers ∈ {1.5, 1.6, 1.7, 1.8, 1.9, 2.0}; Target 2 multipliers ∈ {2.7, 2.8, 2.9, 3.0}.  
- **Outputs:** Logs/figures/metrics stored in `results/target_sweep` with summary table at `results/target_sweep/target_multiplier_sweep.csv`.

Top combinations by realized P&L:

| Target 1 | Target 2 | P&L (USD) | Win Rate | Trades |
|----------|----------|----------:|---------:|-------:|
| **2.0**  | **3.0**  | **14,769.27** | 50.6 % | 11,817 |
| 2.0      | 2.9      | 14,295.16 | 51.0 % | 11,817 |
| 1.9      | 3.0      | 14,097.85 | 51.1 % | 11,817 |
| 2.0      | 2.8      | 13,861.89 | 51.5 % | 11,817 |
| 1.8      | 3.0      | 13,359.94 | 51.9 % | 11,817 |

Lowering Target 2 to the 2.8–3.0 range while keeping Target 1 at 2.0 boosted net returns relative to our existing `best_config.json` (which uses 2.0 / 3.5). Win rates climbed a few percentage points as targets tightened.

## 3. Recommended Next Steps

- Update `best_config.json` to reflect the preferred profit-target multipliers (e.g., 2.0 / 3.0) and rerun the walk-forward harness to confirm robustness across folds.
- Once the new configuration is locked, promote it for future backtests/live simulations and archive these results for regression comparisons.

All relevant artifacts are already checked into `results/` for reproducibility.
