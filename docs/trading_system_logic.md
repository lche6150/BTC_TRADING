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
   - Requires at least 0.8× 24 h average volume (so signals with nearly zero liquidity are skipped).
   - Rejects bars with extreme volatility (> 1.5× baseline threshold).

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
  - `stop_pct` = `max(0.20 %, 4 × volatility_1h)`
  - `target1_pct` = `1.75 × stop_pct` (close 50 %), `target2_pct` = `3 × stop_pct` (close balance)
  - Break-even and trailing thresholds scale with the same base values (`break_even_trigger_pct`, `trailing_trigger_pct`).

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
   - ≥ 6 h with minimal P&L → close 50 %.
   - ≥ 18 h with sub-stop_pct gains → close 75 %.
   - ≥ 30 h hard limit → force close.

4. **Order execution** – `_close_position`
   - Applies per-exit percentage, updates capital/P&L, logs the reason.
   - Positive partials contribute fractional “win credit” so win rate reflects total profit taken across clips.

5. **End-of-session sweep** (backtest harness)
   - Remaining positions are updated one final time and closed with `SESSION_END` to avoid stale exposure at the end of the evaluation window.

## 5. Output & Metrics

`trading_backtest.py` collects every open/close event to:

- Generate the detailed trade log (`create_trading_log`).
- Render the 3-panel visualization (entries/exits, signal confidence, cumulative P&L).
- Export JSON metrics from `get_performance_summary()` showing realized P&L, win rate, drawdown controls, and open positions.

## Summary

The system fuses model confidence, market context, and volatility-aware risk management to produce trades. Entry decisions are conservative (multi-stage filters), position size is driven by confidence and drawdown status, and exits blend price targets, adaptive stops, and time-based hygiene. Together, these rules aim to translate predictive accuracy into stable trading performance while keeping risk within predefined thresholds.
