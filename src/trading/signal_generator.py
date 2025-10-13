#!/usr/bin/env python3
"""
Signal Generator - Generate trading signals from model predictions
Optimized for Binance trading with fee consideration and confidence filtering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import json

class TradingSignalGenerator:
    """
    Generate trading signals based on model predictions with confidence filtering
    and Binance fee optimization
    """

    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)

        # Default configuration optimized for Binance
        self.config = {
            # Signal generation parameters
            'confidence_threshold': 0.65,  # Only trade when model is confident
            'min_price_change': 0.002,     # Minimum 0.2% expected move to cover fees
            'signal_timeout_minutes': 90,   # Signal validity period

            # Binance fee structure (spot trading)
            'maker_fee': 0.001,            # 0.1% maker fee
            'taker_fee': 0.001,            # 0.1% taker fee
            'slippage': 0.0005,            # 0.05% expected slippage

            # Risk management
            'max_position_size': 0.02,      # Max 2% of portfolio per trade
            'stop_loss_pct': 0.015,         # 1.5% stop loss
            'take_profit_pct': 0.025,       # 2.5% take profit (1.67:1 R/R)

            # Signal filtering
            'min_volume_threshold': 1000000,  # Minimum $1M volume for liquidity
            'volatility_filter': True,      # Filter high volatility periods
            'trend_alignment': True,        # Align with broader trend
        }

        if config:
            self.config.update(config)

        self.current_signals = []
        self.signal_history = []

        self.logger.info(f"TradingSignalGenerator initialized with config: {self.config}")

    def generate_signal(self,
                       model_prediction: float,
                       prediction_confidence: float,
                       market_data: Dict,
                       current_time: datetime = None) -> Optional[Dict]:
        """
        Generate trading signal based on model prediction and market conditions

        Args:
            model_prediction: Model output (0=down, 1=up)
            prediction_confidence: Model confidence score (0-1)
            market_data: Current market conditions
            current_time: Signal generation time

        Returns:
            Trading signal dict or None if no signal
        """
        if current_time is None:
            current_time = datetime.now()

        try:
            # Apply confidence filter
            if prediction_confidence < self.config['confidence_threshold']:
                self.logger.debug(f"Signal filtered: confidence {prediction_confidence:.3f} < threshold {self.config['confidence_threshold']}")
                return None

            # Apply market condition filters
            if not self._validate_market_conditions(market_data):
                return None

            # Calculate minimum profit required to cover fees
            min_profit_required = self._calculate_min_profit_requirement()

            # Generate signal based on prediction
            signal_type = 'LONG' if model_prediction > 0.5 else 'SHORT'

            # Calculate position sizing
            position_size = self._calculate_position_size(
                prediction_confidence,
                market_data.get('volatility', 0.02)
            )

            # Calculate entry/exit levels
            entry_levels = self._calculate_entry_exit_levels(
                market_data['current_price'],
                signal_type,
                prediction_confidence
            )

            signal = {
                'signal_id': f"{signal_type}_{current_time.strftime('%Y%m%d_%H%M%S')}",
                'timestamp': current_time,
                'signal_type': signal_type,
                'confidence': prediction_confidence,
                'position_size_pct': position_size,
                'min_profit_required': min_profit_required,

                # Entry/Exit levels
                'entry_price': entry_levels['entry_price'],
                'stop_loss': entry_levels['stop_loss'],
                'take_profit': entry_levels['take_profit'],
                'timeout': current_time + timedelta(minutes=self.config['signal_timeout_minutes']),

                # Market context
                'market_price': market_data['current_price'],
                'market_volume': market_data.get('volume', 0),
                'volatility': market_data.get('volatility', 0),

                # Risk metrics
                'risk_reward_ratio': entry_levels['risk_reward_ratio'],
                'expected_return': entry_levels['expected_return'],
                'max_loss': entry_levels['max_loss'],

                # Status
                'status': 'PENDING',
                'filled': False,
                'reason': f"Model confidence: {prediction_confidence:.3f}"
            }

            # Validate signal profitability
            if signal['expected_return'] < min_profit_required:
                self.logger.debug(f"Signal filtered: expected return {signal['expected_return']:.3f} < required {min_profit_required:.3f}")
                return None

            self.current_signals.append(signal)
            self.logger.info(f"Generated {signal_type} signal: {signal['signal_id']} (confidence: {prediction_confidence:.3f})")

            return signal

        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None

    def _validate_market_conditions(self, market_data: Dict) -> bool:
        """Validate market conditions are suitable for trading"""

        # Check minimum volume for liquidity
        if market_data.get('volume', 0) < self.config['min_volume_threshold']:
            self.logger.debug("Market filtered: insufficient volume")
            return False

        # Check volatility filter
        if self.config['volatility_filter']:
            volatility = market_data.get('volatility', 0)
            if volatility > 0.05:  # > 5% volatility
                self.logger.debug(f"Market filtered: high volatility {volatility:.3f}")
                return False

        # Check for price gaps or anomalies
        current_price = market_data.get('current_price', 0)
        if current_price <= 0:
            self.logger.debug("Market filtered: invalid price")
            return False

        return True

    def _calculate_min_profit_requirement(self) -> float:
        """Calculate minimum profit required to cover all trading costs"""

        # Total cost = entry fee + exit fee + slippage
        total_cost = (self.config['taker_fee'] * 2) + (self.config['slippage'] * 2)

        # Add 50% buffer for safety
        min_profit = total_cost * 1.5

        return max(min_profit, self.config['min_price_change'])

    def _calculate_position_size(self, confidence: float, volatility: float) -> float:
        """Calculate position size based on confidence and volatility"""

        # Base size from confidence
        base_size = self.config['max_position_size'] * confidence

        # Adjust for volatility (reduce size in high volatility)
        volatility_adjustment = max(0.5, 1 - (volatility * 10))

        # Final position size
        position_size = base_size * volatility_adjustment

        return min(position_size, self.config['max_position_size'])

    def _calculate_entry_exit_levels(self,
                                   current_price: float,
                                   signal_type: str,
                                   confidence: float) -> Dict:
        """Calculate optimal entry, stop loss, and take profit levels"""

        # Adjust levels based on confidence
        confidence_multiplier = 0.5 + (confidence * 0.5)  # Scale from 0.5 to 1.0

        if signal_type == 'LONG':
            # Long position levels
            entry_price = current_price * (1 + self.config['slippage'])  # Account for slippage
            stop_loss = entry_price * (1 - self.config['stop_loss_pct'])
            take_profit = entry_price * (1 + self.config['take_profit_pct'] * confidence_multiplier)

        else:  # SHORT
            # Short position levels
            entry_price = current_price * (1 - self.config['slippage'])  # Account for slippage
            stop_loss = entry_price * (1 + self.config['stop_loss_pct'])
            take_profit = entry_price * (1 - self.config['take_profit_pct'] * confidence_multiplier)

        # Calculate risk metrics
        max_loss = abs(entry_price - stop_loss) / entry_price
        expected_return = abs(take_profit - entry_price) / entry_price
        risk_reward_ratio = expected_return / max_loss if max_loss > 0 else 0

        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'max_loss': max_loss,
            'expected_return': expected_return,
            'risk_reward_ratio': risk_reward_ratio
        }

    def update_signal_status(self, signal_id: str, status: str, **kwargs):
        """Update signal status (FILLED, CANCELLED, COMPLETED, etc.)"""

        for signal in self.current_signals:
            if signal['signal_id'] == signal_id:
                signal['status'] = status
                signal.update(kwargs)

                if status in ['COMPLETED', 'CANCELLED', 'STOPPED']:
                    # Move to history
                    self.signal_history.append(signal)
                    self.current_signals.remove(signal)

                self.logger.info(f"Signal {signal_id} status updated to {status}")
                break

    def cleanup_expired_signals(self, current_time: datetime = None):
        """Remove expired signals that haven't been filled"""

        if current_time is None:
            current_time = datetime.now()

        expired_signals = []
        for signal in self.current_signals:
            if current_time > signal['timeout'] and signal['status'] == 'PENDING':
                expired_signals.append(signal)

        for signal in expired_signals:
            self.update_signal_status(signal['signal_id'], 'EXPIRED')
            self.logger.info(f"Signal {signal['signal_id']} expired")

    def get_active_signals(self) -> List[Dict]:
        """Get all active signals"""
        return [s for s in self.current_signals if s['status'] in ['PENDING', 'FILLED']]

    def get_signal_performance(self) -> Dict:
        """Calculate signal performance metrics"""

        completed_signals = [s for s in self.signal_history if s['status'] == 'COMPLETED']

        if not completed_signals:
            return {'total_signals': 0, 'win_rate': 0, 'avg_return': 0}

        total_signals = len(completed_signals)
        winning_signals = len([s for s in completed_signals if s.get('actual_return', 0) > 0])

        total_return = sum(s.get('actual_return', 0) for s in completed_signals)
        avg_return = total_return / total_signals

        win_rate = winning_signals / total_signals

        return {
            'total_signals': total_signals,
            'winning_signals': winning_signals,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'sharpe_ratio': self._calculate_sharpe_ratio(completed_signals)
        }

    def _calculate_sharpe_ratio(self, signals: List[Dict]) -> float:
        """Calculate Sharpe ratio for signal performance"""

        returns = [s.get('actual_return', 0) for s in signals]

        if len(returns) < 2:
            return 0

        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0

        # Assuming 365 trading days per year
        sharpe_ratio = (avg_return * 365) / (std_return * np.sqrt(365))

        return sharpe_ratio

    def save_signals_to_file(self, filepath: str):
        """Save signal history to JSON file"""

        data = {
            'config': self.config,
            'active_signals': self.current_signals,
            'signal_history': self.signal_history,
            'performance': self.get_signal_performance(),
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info(f"Signals saved to {filepath}")

    def load_signals_from_file(self, filepath: str):
        """Load signal history from JSON file"""

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.config.update(data.get('config', {}))
            self.current_signals = data.get('active_signals', [])
            self.signal_history = data.get('signal_history', [])

            self.logger.info(f"Signals loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading signals: {e}")


class PositionManager:
    """
    Manage trading positions with Binance fee optimization
    """

    def __init__(self, initial_balance: float = 1000.0):
        self.logger = logging.getLogger(__name__)
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}
        self.trade_history = []

        # Binance fee structure
        self.fees = {
            'maker': 0.001,    # 0.1%
            'taker': 0.001,    # 0.1%
            'withdrawal': 0.0005  # Network fee approximation
        }

        self.logger.info(f"PositionManager initialized with balance: ${initial_balance}")

    def calculate_position_value(self, price: float, quantity: float, side: str) -> Dict:
        """Calculate position value accounting for fees"""

        notional_value = price * quantity

        if side == 'BUY':
            fee = notional_value * self.fees['taker']
            total_cost = notional_value + fee
        else:  # SELL
            fee = notional_value * self.fees['taker']
            total_received = notional_value - fee
            total_cost = -total_received

        return {
            'notional_value': notional_value,
            'fee': fee,
            'total_cost': total_cost,
            'effective_price': total_cost / quantity if quantity != 0 else 0
        }

    def open_position(self, signal: Dict, current_price: float) -> bool:
        """Open a new position based on signal"""

        try:
            # Calculate position size in dollars
            position_value = self.current_balance * signal['position_size_pct']

            # Calculate quantity
            quantity = position_value / current_price

            # Calculate costs
            costs = self.calculate_position_value(current_price, quantity, 'BUY')

            # Check if we have enough balance
            if costs['total_cost'] > self.current_balance:
                self.logger.warning(f"Insufficient balance for position: required ${costs['total_cost']:.2f}, available ${self.current_balance:.2f}")
                return False

            # Create position
            position = {
                'signal_id': signal['signal_id'],
                'type': signal['signal_type'],
                'entry_price': costs['effective_price'],
                'quantity': quantity,
                'entry_time': datetime.now(),
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'initial_value': costs['total_cost'],
                'entry_fee': costs['fee'],
                'status': 'OPEN'
            }

            # Update balance and positions
            self.current_balance -= costs['total_cost']
            self.positions[signal['signal_id']] = position

            self.logger.info(f"Opened {signal['signal_type']} position: {signal['signal_id']} "
                           f"(qty: {quantity:.6f}, price: ${costs['effective_price']:.2f})")

            return True

        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
            return False

    def close_position(self, signal_id: str, current_price: float, reason: str = 'MANUAL') -> Dict:
        """Close a position and calculate P&L"""

        if signal_id not in self.positions:
            self.logger.error(f"Position {signal_id} not found")
            return None

        position = self.positions[signal_id]

        try:
            # Calculate exit costs
            exit_costs = self.calculate_position_value(current_price, position['quantity'], 'SELL')

            # Calculate P&L
            gross_pnl = exit_costs['total_cost'] + position['initial_value']  # exit_costs is negative
            total_fees = position['entry_fee'] + exit_costs['fee']
            net_pnl = gross_pnl - total_fees

            # Calculate returns
            return_pct = net_pnl / position['initial_value']

            # Update balance
            self.current_balance += abs(exit_costs['total_cost'])

            # Create trade record
            trade = {
                'signal_id': signal_id,
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': exit_costs['effective_price'],
                'quantity': position['quantity'],
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'gross_pnl': gross_pnl,
                'total_fees': total_fees,
                'net_pnl': net_pnl,
                'return_pct': return_pct,
                'reason': reason,
                'initial_value': position['initial_value']
            }

            # Update position status
            position['status'] = 'CLOSED'
            position['exit_price'] = exit_costs['effective_price']
            position['exit_time'] = datetime.now()
            position['net_pnl'] = net_pnl
            position['return_pct'] = return_pct

            # Move to history
            self.trade_history.append(trade)
            del self.positions[signal_id]

            self.logger.info(f"Closed position {signal_id}: P&L ${net_pnl:.2f} ({return_pct*100:.2f}%) - {reason}")

            return trade

        except Exception as e:
            self.logger.error(f"Error closing position {signal_id}: {e}")
            return None

    def update_positions(self, current_price: float):
        """Update all open positions and check for stop loss/take profit"""

        positions_to_close = []

        for signal_id, position in self.positions.items():

            if position['type'] == 'LONG':
                # Check stop loss
                if current_price <= position['stop_loss']:
                    positions_to_close.append((signal_id, 'STOP_LOSS'))
                # Check take profit
                elif current_price >= position['take_profit']:
                    positions_to_close.append((signal_id, 'TAKE_PROFIT'))

            else:  # SHORT
                # Check stop loss
                if current_price >= position['stop_loss']:
                    positions_to_close.append((signal_id, 'STOP_LOSS'))
                # Check take profit
                elif current_price <= position['take_profit']:
                    positions_to_close.append((signal_id, 'TAKE_PROFIT'))

        # Close triggered positions
        closed_trades = []
        for signal_id, reason in positions_to_close:
            trade = self.close_position(signal_id, current_price, reason)
            if trade:
                closed_trades.append(trade)

        return closed_trades

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""

        total_position_value = sum(pos['initial_value'] for pos in self.positions.values())

        # Calculate performance metrics
        completed_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t['net_pnl'] > 0])

        total_pnl = sum(t['net_pnl'] for t in self.trade_history)
        total_fees = sum(t['total_fees'] for t in self.trade_history)

        win_rate = winning_trades / completed_trades if completed_trades > 0 else 0

        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'position_value': total_position_value,
            'total_portfolio_value': self.current_balance + total_position_value,
            'total_pnl': total_pnl,
            'total_fees_paid': total_fees,
            'total_return_pct': (self.current_balance + total_position_value - self.initial_balance) / self.initial_balance,
            'open_positions': len(self.positions),
            'completed_trades': completed_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'positions': list(self.positions.values()),
            'recent_trades': self.trade_history[-10:]  # Last 10 trades
        }