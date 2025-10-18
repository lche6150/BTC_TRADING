# Trading System Execution Flow

This document summarizes how the mid-frequency BTC/USDT trading system turns a model prediction into trades, referencing the implementation in `src/trading/confidence_trading_system.py` and the orchestration helper `trading_backtest.py`.

## 1. Signal Generation

1. Raw market data are enhanced with momentum, volatility, and trend features (`trading_backtest.py`).
2. A scoring function (or the GRU classifier in production) produces:
   - **Direction** (`UP` / `DOWN`).
   - **Base confidence** (0.55–0.95), shaped by recent/long-term returns and volatility.
3. For each bar, the system packages these alongside contextual market data (`MarketData` dataclass):
   - Current price and volumes.
   - 1 h volatility, 4 h price change, 1 h trend label.
   - Whether the bar is within “high-volume” hours and if a news event flag is active.

## 2. Entry Pipeline (`ComprehensiveTradingSystem.process_signal`)

The trading engine receives the raw direction and base confidence. A trade is only created if all steps succeed:

1. **Global guards** – ` _validate_trading_conditions`
   - Trading is halted if daily/weekly drawdown limits are hit.
   - Prevents opening positions when risk controls are active (e.g. portfolio heat above cap).

2. **Context filters** – `_apply_entry_filters`
   - 15 m volume must be ≥ `volume_filter_ratio` (0.8) × the 24 h average; thin-liquidity bars are skipped.
   - 1 h volatility must be below `volatility_threshold * entry_volatility_multiplier` (2.5 × 1.5 ≈ 3.75 %). Anything above the cap is rejected out of hand.

3. **Confidence adjustment** – `_calculate_adjusted_confidence_from_data`
   - Adds bonuses (e.g. trend alignment, > 2× volume) and penalties (off-hours, elevated volatility, news flag).
   - Result must stay ≥ 0.60 (global minimum) to proceed.

4. **Position sizing** – `_calculate_position_size`
   - Confidence buckets map to 1.5 %, 2.5 %, or 4 % notional exposures.
   - Drawdown controls reduce these percentages automatically (−25 % after 5 % drawdown, −50 % after 10 %).

5. **Risk validation** – `_validate_position_risk`
   - Estimates the new trade’s risk using the dynamic stop percentage (see below) and ensures combined heat stays ≤ 6 % of capital.

If every check passes, `_create_position` instantiates a `TradingPosition` and the trade enters the active book.

## 3. Position Construction

Every position stores the entry snapshot plus dynamic safeguards:

- **Volatility-aware stop & targets**
  - `stop_pct` = `clip(|volatility_1h|, stop_volatility_low=0.003, stop_volatility_high=0.02)` → a 0.3 % to 2 % band.
  - Targets use the config multipliers (`target1_multiplier`, `target2_multiplier`; defaults 1.75 and 3.0, live config 2.0 and 3.5). Target 1 always closes 50 % of size; Target 2 closes the remainder.
  - `break_even_multiplier = 2.0`, `trailing_trigger_multiplier = 3.5`, and `trailing_offset_multiplier = 0.75` inherit from the same stop percentage.

- **State tracking**
  - Records realized/unrealized P&L, run-up/run-down, remaining position size, and “win credit” (used when partial profits are taken).

## 4. Ongoing Management (`update_positions`)

On every bar:

1. **Stop-loss evaluation** – `_check_stop_loss`
   - Main stop based on volatility-adjusted price.
   - Break-even stop activates once price moves ≥ break-even trigger.
   - Trailing stop engages after a larger run-up and ratchets based on `trailing_offset_pct`.

2. **Profit targets** – `_check_profit_targets`
   - Target 1 (≥ threshold, position still full) closes 50 %.
   - Target 2 closes the remainder.

3. **Time-based rules** – `_check_exit_conditions`
   - After 6 h with |unrealized| ≤ 0.5 × stop_pct, the position is now closed outright (`TIME_LIMIT_FLAT`) to avoid churning partial exits.
   - After 18 h with unrealized gains still below stop_pct, the position is closed (`TIME_LIMIT_SLOW_PROGRESS`).
   - A hard cap (`time_limit_max_hours`, default 30 h) force-closes any stragglers.

4. **Order execution** – `_close_position`
   - Applies per-exit percentage, updates capital/P&L, logs the reason.
   - Positive partials contribute fractional “win credit” so win rate reflects total profit taken across clips.

5. **End-of-session sweep** (backtest harness)
   - Remaining positions are updated one final time and closed with `SESSION_END` to avoid stale exposure at the end of the evaluation window.

## 5. Output & Metrics

`trading_backtest.py` mirrors the live loop:

- Every signal is logged with `status`, `stage`, and `reason` (matching the paper-trader diagnostics).
- The trade log and Figure 1 chart are generated from the same open/close stream the live loop writes into `paper_trades.log`.
- Metrics from `get_performance_summary()` (P&L, win rate, drawdown flags, active count) are exported alongside the signal log so backtests and replays line up.

## Summary

Both backtest and paper trader now share an identical confidence- and volatility-aware engine: multistage signal filters, adaptive sizing tied to drawdown state, portfolio-heat enforcement, dynamic stops/targets, and strict time hygiene. The richer logging makes it easy to trace why signals were accepted or rejected, and the validation-based threshold keeps the GRU’s performance honest between historical and live runs.
