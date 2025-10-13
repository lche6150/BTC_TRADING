# 70% Accuracy Model - Trading Strategy

## Core Strategy Framework

**Model**: 70% directional accuracy (UP/DOWN predictions)  
**Capital**: $20,000 USD  
**Timeframe**: 15-minute signals  
**Pairs**: BTC/USDT, ETH/USDT

---

## Position Sizing Rules

### Base Position Sizes

- **Conservative**: 1-2% of capital ($200-400)
- **Moderate**: 2-3% of capital ($400-600)
- **Aggressive**: 3-5% of capital ($600-1,000)

### Confidence-Based Sizing

| Signal Confidence | Position Size | Risk Per Trade |
| ----------------- | ------------- | -------------- |
| High (>80%)       | 3% of capital | 1.0% max loss  |
| Medium (60-80%)   | 2% of capital | 0.8% max loss  |
| Low (55-60%)      | 1% of capital | 0.5% max loss  |

---

## Entry Logic

### Long Entry Conditions

1. **Model Signal**: UP prediction with >55% confidence
2. **Volume Filter**: Current 15m volume > 1.5x average
3. **Volatility Check**: Price volatility < 3 standard deviations
4. **Time Window**: Trade during 8:00-22:00 UTC (high volume hours)
5. **Trend Alignment**: 1-hour trend not strongly bearish

### Short Entry Conditions

1. **Model Signal**: DOWN prediction with >55% confidence
2. **Volume Filter**: Current 15m volume > 1.5x average
3. **Volatility Check**: Price volatility < 3 standard deviations
4. **Time Window**: Trade during 8:00-22:00 UTC
5. **Trend Alignment**: 1-hour trend not strongly bullish

### Signal Quality Enhancement

**Add +10% confidence bonus for**:

- Volume >2x average
- Multi-timeframe alignment (15m + 1h same direction)
- Price near key support/resistance levels

**Subtract -10% confidence penalty for**:

- Major news events within 2 hours
- Conflicting higher timeframe signals
- Recent high volatility (>5% moves in past 4 hours)

---

## Exit Strategy

### Profit Targets

- **Target 1**: 2% gain → Close 50% of position
- **Target 2**: 4% gain → Close remaining 50%
- **Maximum Hold**: 24 hours regardless of P&L

### Stop Loss Rules

- **Initial Stop**: 1.5% from entry price
- **Break-Even Stop**: Move to entry price after 1% gain
- **Trailing Stop**: After 2% gain, trail by 1%

### Time-Based Exits

- **4 Hours**: If position flat (±0.2%), close 50%
- **12 Hours**: If position <1% gain, close 75%
- **24 Hours**: Force close all positions

---

## Risk Management

### Position Limits

- **Maximum simultaneous positions**: 3
- **Daily loss limit**: 3% of capital ($600)
- **Weekly loss limit**: 8% of capital ($1,600)
- **Maximum portfolio heat**: 6% total risk exposure

### Drawdown Controls

- **5% account drawdown**: Reduce position sizes by 25%
- **10% account drawdown**: Reduce position sizes by 50%
- **15% account drawdown**: Stop trading, strategy review

### Emergency Exits

- **Flash crash**: >10% move in 1 hour → Close all positions
- **Model confidence drop**: Signal drops below 40% → Close position
- **Exchange issues**: Unable to monitor >2 hours → Close all

---

## Trading Logic Flow

### Signal Processing

1. **Receive model prediction** (UP/DOWN + confidence %)
2. **Apply filters** (volume, volatility, time, trend)
3. **Calculate adjusted confidence** (bonuses/penalties)
4. **Determine position size** based on final confidence
5. **Execute trade** if confidence >55%

### Position Management

1. **Set initial stop loss** at 1.5% from entry
2. **Monitor for profit targets** (2% and 4%)
3. **Adjust stops** (break-even after 1% gain)
4. **Apply trailing stops** (after 2% gain)
5. **Force close** at 24-hour limit

### Daily Routine

1. **Pre-market**: Check overnight news, set daily limits
2. **Active hours**: Monitor signals every 15 minutes
3. **Position tracking**: Update stops and targets
4. **End of day**: Calculate P&L, plan next session

---

## Performance Targets

### Expected Metrics (70% Accuracy Model)

- **Monthly return**: 3-5%
- **Win rate**: >60% (allowing for model variance)
- **Sharpe ratio**: >1.5
- **Maximum drawdown**: <10%
- **Average hold time**: 8-16 hours

### Trade Frequency

- **Quality signals**: 2-5 per day per pair
- **Executed trades**: 60-100 per month
- **Average trades per week**: 15-25

---

## Strategy Rules Summary

### ALWAYS Do

- Risk only 1% of capital per trade
- Use stop losses on every position
- Follow the confidence-based position sizing
- Close positions at 24-hour limit
- Respect daily and weekly loss limits

### NEVER Do

- Trade without stop loss
- Risk more than 3% position size
- Hold positions longer than 24 hours
- Trade during major news events
- Override the model signals with discretion

### Performance Review

- **Daily**: Calculate win rate, P&L, rule compliance
- **Weekly**: Assess strategy effectiveness vs targets
- **Monthly**: Full performance analysis and adjustments

---

**Key Success Factor**: Strict discipline in following these rules, especially during drawdown periods. The 70% accuracy edge only works with proper risk management and consistent execution.
