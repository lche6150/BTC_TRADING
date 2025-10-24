#!/usr/bin/env python3
"""
Comprehensive Trading System - trading_logic.md Implementation

Model: 70% directional accuracy (UP/DOWN predictions)
Capital: $20,000 USD
Timeframe: 15-minute signals

Key Features:
- Confidence-based position sizing (1-3% based on >60% confidence)
- Staged profit targets (0.2%→50% close, 0.3%→100% close)
- Time-based exits (4h/12h/24h limits)
- Volume/volatility filters with signal quality bonuses
- Comprehensive risk management (daily/weekly limits, drawdown controls)
- Multi-timeframe trend alignment
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import time
import json
from dataclasses import dataclass
from enum import Enum
from copy import deepcopy

class PositionStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL_CLOSED = "PARTIAL_CLOSED"  # After first profit target
    CLOSED = "CLOSED"
    STOPPED_OUT = "STOPPED_OUT"
    TIME_EXPIRED = "TIME_EXPIRED"

@dataclass
class MarketData:
    """Current market conditions for signal processing"""
    current_price: float
    volume_15m: float
    avg_volume_24h: float
    volatility_1h: float
    price_change_4h: float
    trend_1h: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    is_high_volume_hours: bool  # 8:00-22:00 UTC
    recent_news_events: bool  # Major events within 2h

class TradingPosition:
    """Individual trading position with full lifecycle management"""

    def __init__(self, signal_id: str, timestamp: datetime, direction: str,
                 base_confidence: float, market_data: MarketData,
                 entry_price: float, position_size_pct: float,
                 risk_config: Optional[Dict[str, float]] = None):

        # Signal identification
        self.signal_id = signal_id
        self.timestamp = timestamp
        self.direction = direction  # 'UP' or 'DOWN'
        self.base_confidence = base_confidence
        self.adjusted_confidence = base_confidence
        self.market_data = market_data

        # Position sizing and entry
        self.position_size_pct = position_size_pct
        self.entry_price = entry_price
        self.position_value = 0.0  # Will be set based on portfolio value

        config = risk_config or {}
        vol_low = config.get("stop_volatility_low", 0.003)
        vol_high = config.get("stop_volatility_high", 0.02)
        self.volatility_pct = float(np.clip(abs(market_data.volatility_1h), vol_low, vol_high))
        self.stop_pct = self.volatility_pct
        self.target1_pct = self.stop_pct * config.get("target1_multiplier", 1.75)
        self.target2_pct = self.stop_pct * config.get("target2_multiplier", 3.0)
        self.break_even_trigger_pct = self.stop_pct * config.get("break_even_multiplier", 2.0)
        self.trailing_trigger_pct = self.stop_pct * config.get("trailing_trigger_multiplier", 3.5)
        self.trailing_offset_pct = self.stop_pct * config.get("trailing_offset_multiplier", 0.75)

        # Risk management levels (converted to prices)
        self.stop_loss = self._calculate_stop_loss()
        self.target_1 = self._calculate_target_1()
        self.target_2 = self._calculate_target_2()
        self.break_even_stop = entry_price  # Updated when break-even trigger hit
        self.trailing_stop: Optional[float] = None

        # Position tracking
        self.status = PositionStatus.FILLED
        self.entry_time = timestamp
        self.exit_time = None
        self.remaining_size_pct = 1.0  # 100% initially

        # Performance tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.max_favorable = 0.0  # Best unrealized gain
        self.max_adverse = 0.0    # Worst unrealized loss
        self.win_credit = 0.0
        self.total_fees_paid = 0.0
        self.entry_fee = 0.0

    def _calculate_stop_loss(self) -> float:
        """Initial stop based on volatility-adjusted percentage"""
        if self.direction == 'UP':
            return self.entry_price * (1.0 - self.stop_pct)
        else:
            return self.entry_price * (1.0 + self.stop_pct)

    def _calculate_target_1(self) -> float:
        """Target 1: scale with volatility (close 50%)"""
        if self.direction == 'UP':
            return self.entry_price * (1.0 + self.target1_pct)
        else:
            return self.entry_price * (1.0 - self.target1_pct)

    def _calculate_target_2(self) -> float:
        """Target 2: scale with volatility (close remaining)"""
        if self.direction == 'UP':
            return self.entry_price * (1.0 + self.target2_pct)
        else:
            return self.entry_price * (1.0 - self.target2_pct)

    def update_stops_after_1pct_gain(self):
        """Move stop to break-even after 1% gain"""
        self.break_even_stop = self.entry_price

    def update_trailing_stop_after_2pct_gain(self, current_price: float):
        """Activate trailing stop after 2% gain, trail by 1%"""
        if self.direction == 'UP':
            self.trailing_stop = current_price * 0.99  # Trail by 1%
        else:
            self.trailing_stop = current_price * 1.01  # Trail by 1%

    def get_hold_time_hours(self, reference_time: Optional[datetime] = None) -> float:
        """Calculate position hold time in hours"""
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds() / 3600

        reference_time = reference_time or datetime.utcnow()
        return (reference_time - self.entry_time).total_seconds() / 3600

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate current unrealized P&L percentage"""
        if self.direction == 'UP':
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        self.unrealized_pnl = pnl_pct * self.remaining_size_pct

        # Update max favorable/adverse
        if pnl_pct > self.max_favorable:
            self.max_favorable = pnl_pct
        if pnl_pct < self.max_adverse:
            self.max_adverse = pnl_pct

        return self.unrealized_pnl

class ComprehensiveTradingSystem:
    """
    Complete trading system implementing trading_logic.md specifications

    Model: 70% directional accuracy with confidence-based sizing
    Capital: $20,000 USD default
    Risk Management: Strict position limits and time-based exits
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "min_confidence": 0.60,
        "position_sizes": {"low": 0.015, "medium": 0.025, "high": 0.04},
        "max_portfolio_heat": 0.06,
        "daily_loss_limit": 0.03,
        "weekly_loss_limit": 0.08,
        "max_simultaneous_positions": 999,
        "volume_bonus_multiplier": 2.0,
        "volume_penalty_threshold": 1.2,
        "low_volume_penalty_weight": 0.5,
        "volume_filter_ratio": 0.8,
        "entry_volatility_multiplier": 1.5,
        "volatility_threshold": 2.5,
        "confidence_bonus": 0.10,
        "confidence_penalty": 0.10,
        "off_hours_penalty_weight": 0.5,
        "high_vol_penalty_scaling": 1.0,
        "stop_volatility_low": 0.003,
        "stop_volatility_high": 0.02,
        "target1_multiplier": 1.75,
        "target2_multiplier": 3.0,
        "break_even_multiplier": 2.0,
        "trailing_trigger_multiplier": 3.5,
        "trailing_offset_multiplier": 0.75,
        "time_limit_flat_hours": 6.0,
        "time_limit_flat_tolerance_multiplier": 0.5,
        "time_limit_slow_hours": 18.0,
        "time_limit_slow_threshold_multiplier": 1.0,
        "time_limit_max_hours": 30.0,
        "maker_fee_rate": 0.001,
        "taker_fee_rate": 0.001,
        "stop_reentry_cooldown_minutes": 30,
    }

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return deepcopy(cls.DEFAULT_CONFIG)

    @staticmethod
    def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                ComprehensiveTradingSystem._deep_update(base[key], value)
            else:
                base[key] = value
        return base

    def __init__(self, initial_capital: float = 20000.0, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)

        self.config = self.default_config()
        if config:
            self._deep_update(self.config, config)

        # Portfolio management
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_start_capital = initial_capital
        self.weekly_start_capital = initial_capital

        # Position tracking
        self.active_positions: List[TradingPosition] = []
        self.closed_positions: List[TradingPosition] = []
        self.position_counter = 0

        # Configured limits and sizing
        self.max_simultaneous_positions = self.config["max_simultaneous_positions"]
        self.daily_loss_limit = self.config["daily_loss_limit"]
        self.weekly_loss_limit = self.config["weekly_loss_limit"]
        self.max_portfolio_heat = self.config["max_portfolio_heat"]

        self.min_confidence = self.config["min_confidence"]
        self.position_sizes = self.config["position_sizes"].copy()

        # Risk control states
        self.drawdown_5pct_active = False
        self.drawdown_10pct_active = False
        self.trading_halted = False
        self.halt_reason = ""

        # Performance tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_trade_count = 0
        self.signal_history = []

        # Signal quality parameters
        self.volume_bonus_multiplier = self.config["volume_bonus_multiplier"]
        self.volume_penalty_threshold = self.config["volume_penalty_threshold"]
        self.low_volume_penalty_weight = self.config["low_volume_penalty_weight"]
        self.volume_filter_ratio = self.config["volume_filter_ratio"]
        self.entry_volatility_multiplier = self.config["entry_volatility_multiplier"]
        self.volatility_threshold = self.config["volatility_threshold"]
        self.confidence_bonus = self.config["confidence_bonus"]
        self.confidence_penalty = self.config["confidence_penalty"]
        self.off_hours_penalty_weight = self.config["off_hours_penalty_weight"]
        self.high_vol_penalty_scaling = self.config["high_vol_penalty_scaling"]

        # Risk parameterization
        self.stop_volatility_low = self.config["stop_volatility_low"]
        self.stop_volatility_high = self.config["stop_volatility_high"]
        self.target1_multiplier = self.config["target1_multiplier"]
        self.target2_multiplier = self.config["target2_multiplier"]
        self.break_even_multiplier = self.config["break_even_multiplier"]
        self.trailing_trigger_multiplier = self.config["trailing_trigger_multiplier"]
        self.trailing_offset_multiplier = self.config["trailing_offset_multiplier"]

        self.maker_fee_rate = self.config["maker_fee_rate"]
        self.taker_fee_rate = self.config["taker_fee_rate"]

        cooldown_minutes = max(0, float(self.config.get("stop_reentry_cooldown_minutes", 0)))
        self.stop_reentry_cooldown = timedelta(minutes=cooldown_minutes)
        self.last_stop_loss_time = {"UP": None, "DOWN": None}

        # Time management
        self.time_limit_flat_hours = self.config["time_limit_flat_hours"]
        self.time_limit_flat_tolerance_multiplier = self.config["time_limit_flat_tolerance_multiplier"]
        self.time_limit_slow_hours = self.config["time_limit_slow_hours"]
        self.time_limit_slow_threshold_multiplier = self.config["time_limit_slow_threshold_multiplier"]
        self.time_limit_max_hours = self.config["time_limit_max_hours"]

        # Signal diagnostics
        self.last_signal_decision: Optional[Dict[str, Any]] = None

        self.logger.info(f"Trading system initialized with ${initial_capital:,.2f} capital")

    def process_signal(
        self,
        direction: str,
        base_confidence: float,
        market_data: MarketData,
        signal_time: Optional[datetime] = None,
    ) -> Optional[TradingPosition]:
        """
        Main signal processing pipeline implementing trading_logic.md entry logic.

        Returns TradingPosition if trade should be executed, None otherwise.
        Diagnostics for the most recent signal are stored on `last_signal_decision`.
        """

        signal_time = signal_time or datetime.utcnow()

        decision_context: Dict[str, Any] = {
            "status": "pending",
            "stage": "initialize",
            "direction": direction,
            "base_confidence": float(base_confidence),
            "market_price": float(market_data.current_price),
            "timestamp": signal_time.isoformat(),
        }
        self.last_signal_decision = decision_context.copy()

        try:
            # 1. Pre-trade validation
            valid, reason, details = self._validate_trading_conditions()
            if not valid:
                decision_context.update({
                    "status": "rejected",
                    "stage": "pre_trade_checks",
                    "reason": reason or "pre_trade_checks_failed",
                })
                if details:
                    decision_context["details"] = details
                self.last_signal_decision = decision_context
                return None

            # 2. Apply signal filters (volume, volatility, time, trend)
            filters_ok, filter_reason, filter_details = self._apply_entry_filters(market_data)
            if not filters_ok:
                decision_context.update({
                    "status": "rejected",
                    "stage": "entry_filters",
                    "reason": filter_reason or "entry_filters_failed",
                })
                if filter_details:
                    decision_context["details"] = filter_details
                self.last_signal_decision = decision_context
                return None

            # 3. Calculate adjusted confidence (bonuses/penalties)
            adjusted_confidence = self._calculate_adjusted_confidence_from_data(
                base_confidence, market_data)
            decision_context["adjusted_confidence"] = float(adjusted_confidence)

            # 4. Check minimum confidence threshold
            if adjusted_confidence < self.min_confidence:
                self.logger.debug(
                    "Signal rejected: confidence %.3f < %.3f",
                    adjusted_confidence,
                    self.min_confidence,
                )
                decision_context.update({
                    "status": "rejected",
                    "stage": "confidence_threshold",
                    "reason": "adjusted_confidence_below_min",
                    "min_confidence": self.min_confidence,
                })
                self.last_signal_decision = decision_context
                return None

            # 5. Determine position size based on confidence
            position_size_pct = self._calculate_position_size(adjusted_confidence)
            decision_context["position_size_pct"] = float(position_size_pct)

            # 6. Final risk checks
            estimated_stop_pct = float(np.clip(abs(market_data.volatility_1h), 0.003, 0.02))
            risk_ok, risk_reason, risk_details = self._validate_position_risk(
                position_size_pct, market_data.current_price, estimated_stop_pct
            )
            if not risk_ok:
                decision_context.update({
                    "status": "rejected",
                    "stage": "risk_validation",
                    "reason": risk_reason or "position_risk_check_failed",
                    "estimated_stop_pct": estimated_stop_pct,
                })
                if risk_details:
                    decision_context.setdefault("details", {}).update(risk_details)
                self.last_signal_decision = decision_context
                return None

            # 7. Create and execute position
            position = self._create_position(
                direction, adjusted_confidence, market_data,
                market_data.current_price, position_size_pct
            )

            self.active_positions.append(position)
            self.total_trades += 1
            self.daily_trade_count += 1

            decision_context.update({
                "status": "accepted",
                "stage": "position_opened",
                "reason": "position_opened",
                "position_id": position.signal_id,
            })
            self.last_signal_decision = decision_context

            self.logger.info(
                "Position opened: %s $%s (%.1f%%) confidence=%.1f%%",
                direction,
                f"{position.position_value:,.2f}",
                position_size_pct * 100,
                adjusted_confidence * 100,
            )

            return position

        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            decision_context.update({
                "status": "rejected",
                "stage": "exception",
                "reason": "exception",
                "details": {"error": str(e)},
            })
            self.last_signal_decision = decision_context
            return None

    def _validate_trading_conditions(self) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Pre-trade validation checks."""

        # Check if trading is halted
        if self.trading_halted:
            self.logger.debug(f"Trading halted: {self.halt_reason}")
            return False, "trading_halted", {"halt_reason": self.halt_reason}

        # Check maximum simultaneous positions
        if len(self.active_positions) >= self.max_simultaneous_positions:
            self.logger.debug(f"Max positions reached: {len(self.active_positions)}")
            return False, "max_positions_reached", {"active_positions": len(self.active_positions)}

        # Check daily loss limit
        daily_loss_pct = (self.current_capital - self.daily_start_capital) / self.daily_start_capital
        if daily_loss_pct <= -self.daily_loss_limit:
            self.logger.warning(f"Daily loss limit reached: {daily_loss_pct:.2%}")
            return False, "daily_loss_limit", {"daily_loss_pct": daily_loss_pct}

        # Check weekly loss limit
        weekly_loss_pct = (self.current_capital - self.weekly_start_capital) / self.weekly_start_capital
        if weekly_loss_pct <= -self.weekly_loss_limit:
            self.logger.warning(f"Weekly loss limit reached: {weekly_loss_pct:.2%}")
            return False, "weekly_loss_limit", {"weekly_loss_pct": weekly_loss_pct}

        return True, None, {}

    def _apply_entry_filters(self, market_data: MarketData) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Apply entry filters from trading_logic.md."""

        diagnostics: Dict[str, Any] = {
            "volume_15m": market_data.volume_15m,
            "avg_volume_24h": market_data.avg_volume_24h,
            "volatility_1h": market_data.volatility_1h,
            "is_high_volume_hours": market_data.is_high_volume_hours,
        }

        # Volume Filter: allow >= configured ratio of average, but note ratio for confidence penalties
        volume_ratio = market_data.volume_15m / max(market_data.avg_volume_24h, 1e-9)
        diagnostics["volume_ratio"] = volume_ratio
        if volume_ratio < self.volume_filter_ratio:
            self.logger.debug(
                "Volume filter failed: %.2f < %.2f",
                volume_ratio,
                self.volume_filter_ratio,
            )
            return False, "volume_ratio_below_threshold", diagnostics

        # Volatility guardrail: hard stop only for extreme regimes
        volatility_cap = self.volatility_threshold * self.entry_volatility_multiplier
        diagnostics["volatility_cap"] = volatility_cap
        if market_data.volatility_1h > volatility_cap:
            self.logger.debug(
                "Volatility filter failed: %.2f > %.2f",
                market_data.volatility_1h,
                volatility_cap,
            )
            return False, "volatility_above_cap", diagnostics

        # Trading window: only allow signals during high-volume hours (08:00-22:00 UTC)
        # if not market_data.is_high_volume_hours:
        #     self.logger.debug("Time window filter failed: outside 08:00-22:00 UTC")
        #     return False, "outside_trading_window", diagnostics

        return True, None, diagnostics

    def _calculate_adjusted_confidence_from_data(self, base_confidence: float,
                                     market_data: MarketData) -> float:
        """Calculate adjusted confidence with bonuses and penalties"""

        adjusted = base_confidence

        # BONUSES (+10% each)
        volume_ratio = market_data.volume_15m / max(market_data.avg_volume_24h, 1e-9)

        # Volume bonus: >2x average volume
        if volume_ratio >= self.volume_bonus_multiplier:
            adjusted += self.confidence_bonus
            self.logger.debug(f"Volume bonus applied: {volume_ratio:.2f}x average")
        elif volume_ratio < self.volume_penalty_threshold:
            adjusted -= self.confidence_penalty * self.low_volume_penalty_weight
            self.logger.debug(f"Volume penalty applied: {volume_ratio:.2f}x average")

        # Multi-timeframe alignment bonus
        # This would require checking if 15m signal aligns with 1h trend
        # For now, simplified check based on trend_1h
        if market_data.trend_1h != 'NEUTRAL':
            adjusted += self.confidence_bonus
            self.logger.debug(f"Trend alignment bonus: {market_data.trend_1h}")

        # PENALTIES (-10% each)

        # Major news events within 2 hours
        if market_data.recent_news_events:
            adjusted -= self.confidence_penalty
            self.logger.debug("News event penalty applied")

        # Recent high volatility (>5% moves in past 4 hours)
        if abs(market_data.price_change_4h) > 0.05:
            adjusted -= self.confidence_penalty
            self.logger.debug(f"High volatility penalty: {market_data.price_change_4h:.2%}")

        # Hour-of-day penalty
        if not market_data.is_high_volume_hours:
            adjusted -= self.confidence_penalty * self.off_hours_penalty_weight
            self.logger.debug("Off-hours penalty applied")

        # Elevated short-term volatility penalty
        if market_data.volatility_1h > self.volatility_threshold:
            severity = min(1.0, (market_data.volatility_1h - self.volatility_threshold) / self.volatility_threshold)
            adjusted -= self.confidence_penalty * severity * self.high_vol_penalty_scaling
            self.logger.debug(
                f"Intraday volatility penalty applied: {market_data.volatility_1h:.2f}"
            )

        # Ensure confidence stays within [0, 1] bounds
        adjusted = max(0.0, min(1.0, adjusted))

        return adjusted

    def _calculate_position_size(self, confidence: float) -> float:
        """Determine position size based on confidence level"""

        # Apply drawdown adjustments first
        base_sizes = self.position_sizes.copy()

        if self.drawdown_10pct_active:
            # 10% drawdown: reduce sizes by 50%
            for key in base_sizes:
                base_sizes[key] *= 0.5
        elif self.drawdown_5pct_active:
            # 5% drawdown: reduce sizes by 25%
            for key in base_sizes:
                base_sizes[key] *= 0.75

        # Confidence-based sizing from trading_logic.md
        if confidence > 0.80:
            return base_sizes['high']    # 3% for high confidence
        elif confidence >= 0.60:
            return base_sizes['medium']  # 2% for medium confidence
        else:
            return base_sizes['low']     # 1% for low confidence

    def _validate_position_risk(self, position_size_pct: float, current_price: float,
                                estimated_stop_pct: float) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Final risk validation before position creation."""

        # Calculate total portfolio heat with new position
        current_heat = self._calculate_portfolio_heat()
        new_position_risk = position_size_pct * estimated_stop_pct
        projected_heat = current_heat + new_position_risk

        diagnostics = {
            "current_heat": current_heat,
            "new_position_risk": new_position_risk,
            "projected_heat": projected_heat,
            "max_portfolio_heat": self.max_portfolio_heat,
            "position_size_pct": position_size_pct,
            "estimated_stop_pct": estimated_stop_pct,
        }

        if projected_heat > self.max_portfolio_heat:
            self.logger.debug(
                "Portfolio heat limit: %.2f%% > %.2f%%",
                projected_heat * 100,
                self.max_portfolio_heat * 100,
            )
            return False, "portfolio_heat_limit", diagnostics

        return True, None, diagnostics

    def _calculate_portfolio_heat(self) -> float:
        """Calculate current total risk exposure"""
        total_risk = 0.0

        for position in self.active_positions:
            # Risk is distance to stop loss
            if position.direction == 'UP':
                risk_pct = (position.entry_price - position.stop_loss) / position.entry_price
            else:
                risk_pct = (position.stop_loss - position.entry_price) / position.entry_price

            position_risk = position.position_size_pct * risk_pct * position.remaining_size_pct
            total_risk += position_risk

        return total_risk

    def _create_position(
        self,
        direction: str,
        confidence: float,
        market_data: MarketData,
        entry_price: float,
        position_size_pct: float,
        entry_time: Optional[datetime] = None,
    ) -> TradingPosition:
        """Create new trading position"""

        entry_time = entry_time or datetime.utcnow()

        self.position_counter += 1
        signal_id = f"TRADE_{self.position_counter:04d}_{entry_time.strftime('%Y%m%d_%H%M%S')}"

        position = TradingPosition(
            signal_id=signal_id,
            timestamp=entry_time,
            direction=direction,
            base_confidence=confidence,
            market_data=market_data,
            entry_price=entry_price,
            position_size_pct=position_size_pct,
            risk_config={
                "stop_volatility_low": self.stop_volatility_low,
                "stop_volatility_high": self.stop_volatility_high,
                "target1_multiplier": self.target1_multiplier,
                "target2_multiplier": self.target2_multiplier,
                "break_even_multiplier": self.break_even_multiplier,
                "trailing_trigger_multiplier": self.trailing_trigger_multiplier,
                "trailing_offset_multiplier": self.trailing_offset_multiplier,
            }
        )

        # Set position value based on current capital
        position.position_value = self.current_capital * position_size_pct

        # Deduct entry fee immediately
        entry_fee = position.position_value * self.taker_fee_rate
        position.entry_fee = entry_fee
        position.total_fees_paid += entry_fee
        self.current_capital -= entry_fee
        self.daily_pnl -= entry_fee

        return position

    def update_positions(self, current_price: float, current_time: datetime = None) -> List[Dict]:
        """
        Update all active positions - check exits, stops, targets
        Returns list of position updates/closes
        """
        if current_time is None:
            current_time = datetime.utcnow()

        updates = []

        for position in self.active_positions.copy():
            # Update unrealized P&L
            position.calculate_unrealized_pnl(current_price)

            # Check various exit conditions
            exit_reason = self._check_exit_conditions(position, current_price, current_time)

            if exit_reason:
                update = self._close_position(position, current_price, current_time, exit_reason)
                updates.append(update)

        return updates

    def _check_exit_conditions(self, position: TradingPosition, current_price: float,
                              current_time: datetime) -> Optional[str]:
        """Check all exit conditions for a position"""

        hold_time_hours = position.get_hold_time_hours(current_time)

        # 1. STOP LOSS CHECK
        if self._check_stop_loss(position, current_price):
            return "STOP_LOSS"

        # 2. PROFIT TARGET CHECKS
        target_hit = self._check_profit_targets(position, current_price)
        if target_hit:
            return target_hit

        # 3. TIME-BASED EXITS (from trading_logic.md)

        if hold_time_hours >= self.time_limit_max_hours:
            return "TIME_LIMIT_MAX_HOLD"

        if hold_time_hours >= self.time_limit_slow_hours:
            if position.unrealized_pnl < position.stop_pct * self.time_limit_slow_threshold_multiplier:
                return "TIME_LIMIT_SLOW_PROGRESS"

        if hold_time_hours >= self.time_limit_flat_hours:
            if abs(position.unrealized_pnl) <= position.stop_pct * self.time_limit_flat_tolerance_multiplier:
                return "TIME_LIMIT_FLAT"

        return None

    def _check_stop_loss(self, position: TradingPosition, current_price: float) -> bool:
        """Check if stop loss should be triggered"""

        # Initial stop loss
        if position.direction == 'UP':
            if current_price <= position.stop_loss:
                return True
        else:  # DOWN/SHORT
            if current_price >= position.stop_loss:
                return True

        # Break-even stop (after sufficient run-up)
        if position.max_favorable >= position.break_even_trigger_pct:
            if position.direction == 'UP':
                if current_price <= position.entry_price:
                    return True
            else:
                if current_price >= position.entry_price:
                    return True

        # Trailing stop (after larger run-up)
        if position.max_favorable >= position.trailing_trigger_pct:
            if position.direction == 'UP':
                trailing_candidate = current_price * (1.0 - position.trailing_offset_pct)
                if position.trailing_stop is None or trailing_candidate > position.trailing_stop:
                    position.trailing_stop = trailing_candidate
                if position.trailing_stop is not None and current_price <= position.trailing_stop:
                    return True
            else:
                trailing_candidate = current_price * (1.0 + position.trailing_offset_pct)
                if position.trailing_stop is None or trailing_candidate < position.trailing_stop:
                    position.trailing_stop = trailing_candidate
                if position.trailing_stop is not None and current_price >= position.trailing_stop:
                    return True

        return False

    def _check_profit_targets(self, position: TradingPosition, current_price: float) -> Optional[str]:
        """Check if profit targets should be triggered"""

        if position.direction == 'UP':
            if current_price >= position.target_2 and position.remaining_size_pct > 0.0:
                return "PROFIT_TARGET_2"

            if current_price >= position.target_1 and position.remaining_size_pct >= 0.999:
                return "PROFIT_TARGET_1"

        else:  # DOWN/SHORT
            if current_price <= position.target_2 and position.remaining_size_pct > 0.0:
                return "PROFIT_TARGET_2"

            if current_price <= position.target_1 and position.remaining_size_pct >= 0.999:
                return "PROFIT_TARGET_1"

        return None

    def _close_position(self, position: TradingPosition, current_price: float,
                       current_time: datetime, exit_reason: str) -> Dict:
        """Close position (fully or partially) and update tracking"""

        # Determine close percentage based on exit reason
        close_percentage = self._get_close_percentage(exit_reason, position)

        # Calculate realized P&L for the closed portion
        if position.direction == 'UP':
            pnl_pct = (current_price - position.entry_price) / position.entry_price
            exit_notional_multiplier = 1.0 + pnl_pct
        else:
            pnl_pct = (position.entry_price - current_price) / position.entry_price
            exit_notional_multiplier = 1.0 - pnl_pct

        closed_value = position.position_value * position.remaining_size_pct * close_percentage
        exit_notional = max(0.0, closed_value * exit_notional_multiplier)
        exit_fee = exit_notional * self.taker_fee_rate
        realized_pnl = closed_value * pnl_pct - exit_fee

        # Update position
        position.realized_pnl += realized_pnl
        position.remaining_size_pct *= (1.0 - close_percentage)
        position.total_fees_paid += exit_fee

        # Update portfolio capital
        self.current_capital += realized_pnl
        self.daily_pnl += realized_pnl

        # Track performance
        if realized_pnl > 0:
            credit = min(close_percentage, 1.0 - position.win_credit)
            position.win_credit += credit
            self.winning_trades += credit

        # Update position status
        if position.remaining_size_pct <= 0.01:  # Effectively closed
            position.status = self._get_final_status(exit_reason)
            position.exit_time = current_time
            self.active_positions.remove(position)
            self.closed_positions.append(position)
        else:
            position.status = PositionStatus.PARTIAL_CLOSED

        update_info = {
            'position_id': position.signal_id,
            'exit_reason': exit_reason,
            'close_percentage': close_percentage,
            'exit_price': current_price,
            'realized_pnl': realized_pnl,
            'remaining_size': position.remaining_size_pct,
            'hold_time_hours': position.get_hold_time_hours(current_time),
            'exit_fee': exit_fee,
            'position_value': position.position_value,
            'position_size_pct': position.position_size_pct,
        }

        self.logger.info(f"Position update: {position.signal_id} {exit_reason} "
                        f"closed {close_percentage:.0%} P&L=${realized_pnl:.2f}")

        return update_info

    def _get_close_percentage(self, exit_reason: str, position: TradingPosition) -> float:
        """Determine what percentage of position to close"""

        if exit_reason == "PROFIT_TARGET_1":
            return 0.5  # Close 50% at first target
        if exit_reason == "PROFIT_TARGET_2":
            return 1.0  # Close remaining at second target
        if exit_reason == "TIME_LIMIT_FLAT":
            # Previously scaled out in multiple 50% steps which compounded fees.
            # Close the full remainder once the flat hold limit is reached.
            return 1.0
        if exit_reason == "TIME_LIMIT_SLOW_PROGRESS":
            return 0.75
        if exit_reason == "TIME_LIMIT_MAX_HOLD":
            return 1.0
        return 1.0  # Default: full close

    def _get_final_status(self, exit_reason: str) -> PositionStatus:
        """Determine final position status"""
        if "STOP" in exit_reason:
            return PositionStatus.STOPPED_OUT
        elif "TIME" in exit_reason:
            return PositionStatus.TIME_EXPIRED
        else:
            return PositionStatus.CLOSED

    def check_drawdown_controls(self):
        """Check and apply drawdown controls from trading_logic.md"""

        # Calculate current drawdown from peak
        peak_capital = max(self.current_capital, self.initial_capital)
        current_drawdown = (peak_capital - self.current_capital) / peak_capital

        # 15% account drawdown: Stop trading, strategy review
        if current_drawdown >= 0.15:
            self.trading_halted = True
            self.halt_reason = f"15% drawdown reached: {current_drawdown:.2%}"
            self.logger.critical(f"TRADING HALTED: {self.halt_reason}")

        # 10% account drawdown: Reduce position sizes by 50%
        elif current_drawdown >= 0.10:
            if not self.drawdown_10pct_active:
                self.drawdown_10pct_active = True
                self.drawdown_5pct_active = False
                self.logger.warning(f"10% drawdown control activated: {current_drawdown:.2%}")

        # 5% account drawdown: Reduce position sizes by 25%
        elif current_drawdown >= 0.05:
            if not self.drawdown_5pct_active:
                self.drawdown_5pct_active = True
                self.logger.warning(f"5% drawdown control activated: {current_drawdown:.2%}")

        # Reset controls if drawdown recovers
        elif current_drawdown < 0.03:
            if self.drawdown_5pct_active or self.drawdown_10pct_active:
                self.drawdown_5pct_active = False
                self.drawdown_10pct_active = False
                self.logger.info("Drawdown controls reset - normal position sizing restored")

    def get_performance_summary(self) -> Dict:
        """Generate comprehensive performance summary"""

        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions)

        # Calculate metrics
        net_realized_pnl = self.current_capital - self.initial_capital
        total_return = net_realized_pnl / self.initial_capital
        win_rate = self.winning_trades / max(self.total_trades, 1)

        # Risk metrics
        current_drawdown = max(0, (max(self.current_capital, self.initial_capital) - self.current_capital) / max(self.current_capital, self.initial_capital))

        return {
            'portfolio_value': self.current_capital,
            'total_return': total_return,
            'realized_pnl': net_realized_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'active_positions': len(self.active_positions),
            'daily_pnl': self.daily_pnl,
            'current_drawdown': current_drawdown,
            'risk_controls': {
                'drawdown_5pct': self.drawdown_5pct_active,
                'drawdown_10pct': self.drawdown_10pct_active,
                'trading_halted': self.trading_halted
            },
            'daily_trade_count': self.daily_trade_count
        }

    def reset_daily_limits(self):
        """Reset daily tracking (call at start of each trading day)"""
        self.daily_start_capital = self.current_capital
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.logger.info(f"Daily limits reset - Starting capital: ${self.current_capital:,.2f}")

    def reset_weekly_limits(self):
        """Reset weekly tracking (call at start of each trading week)"""
        self.weekly_start_capital = self.current_capital
        self.weekly_pnl = 0.0
        self.logger.info(f"Weekly limits reset - Starting capital: ${self.current_capital:,.2f}")


# Example usage and testing functions
def create_sample_market_data(price: float = 50000.0) -> MarketData:
    """Create sample market data for testing"""
    return MarketData(
        current_price=price,
        volume_15m=100.0,
        avg_volume_24h=50.0,  # 2x volume ratio
        volatility_1h=2.0,    # Below 3.0 threshold
        price_change_4h=0.02, # 2% change
        trend_1h='BULLISH',
        is_high_volume_hours=True,
        recent_news_events=False
    )


if __name__ == "__main__":
    # Basic test
    system = ComprehensiveTradingSystem(20000.0)
    market_data = create_sample_market_data(50000.0)

    print("🧪 Testing Trading System")
    print(f"Initial capital: ${system.current_capital:,.2f}")

    position = system.process_signal('UP', 0.75, market_data)
    if position:
        print(f"✅ Position created: {position.signal_id}")
    else:
        print("❌ Position rejected")

    summary = system.get_performance_summary()
    print(f"Portfolio value: ${summary['portfolio_value']:,.2f}")
