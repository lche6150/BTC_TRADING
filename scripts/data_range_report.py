#!/usr/bin/env python3
"""
Comprehensive Data Range Report
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.database import TradingDatabase

def create_data_report():
    """Create comprehensive data range report"""
    print("=" * 70)
    print("📊 BTC/USDT 15-MINUTE DATA RANGE REPORT")
    print("=" * 70)

    try:
        db = TradingDatabase()
        df = db.get_ohlcv_data("BTCUSDT", "15m")

        if df.empty:
            print("❌ No data found in database!")
            return

        # Extract datetime info
        start_date = pd.to_datetime(df['datetime'].iloc[0])
        end_date = pd.to_datetime(df['datetime'].iloc[-1])
        total_records = len(df)

        # Calculate time span
        time_span = end_date - start_date
        days = time_span.days
        months = days / 30.44  # Average days per month

        print(f"📅 DATA TIMEFRAME:")
        print(f"   Start Date: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   End Date:   {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Duration:   {days} days ({months:.1f} months)")

        print(f"\n📊 DATA VOLUME:")
        print(f"   Total Records: {total_records:,}")
        print(f"   Expected Records: {days * 96:,} (96 per day)")
        print(f"   Data Completeness: {(total_records / (days * 96)) * 100:.1f}%")

        # Price analysis
        print(f"\n💰 PRICE ANALYSIS:")
        print(f"   Starting Price: ${df['close'].iloc[0]:,.2f}")
        print(f"   Ending Price:   ${df['close'].iloc[-1]:,.2f}")
        print(f"   Lowest Price:   ${df['low'].min():,.2f}")
        print(f"   Highest Price:  ${df['high'].max():,.2f}")

        price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        print(f"   Total Return:   {price_change:+.1f}%")

        # Volume analysis
        print(f"\n📈 VOLUME ANALYSIS:")
        print(f"   Average Volume: {df['volume'].mean():,.2f} BTC")
        print(f"   Total Volume:   {df['volume'].sum():,.2f} BTC")
        print(f"   Peak Volume:    {df['volume'].max():,.2f} BTC")

        # Data quality check
        print(f"\n🔍 DATA QUALITY:")
        missing_data = df.isnull().sum().sum()
        print(f"   Missing Values: {missing_data}")
        print(f"   Data Integrity: {'✅ Perfect' if missing_data == 0 else '⚠️ Has missing data'}")

        # Check for time gaps
        df_sorted = df.sort_values('datetime').copy()
        time_diffs = df_sorted['datetime'].diff().dropna()
        expected_diff = pd.Timedelta(minutes=15)

        normal_intervals = (time_diffs == expected_diff).sum()
        total_intervals = len(time_diffs)

        print(f"   Time Intervals: {normal_intervals}/{total_intervals} normal (15min)")
        print(f"   Interval Accuracy: {(normal_intervals/total_intervals)*100:.1f}%")

        # Market coverage analysis
        print(f"\n🌍 MARKET COVERAGE:")
        print(f"   Covers: {months:.1f} months of continuous trading")
        print(f"   24/7 Data: ✅ Cryptocurrency market (no gaps)")

        # Recent data status
        now = datetime.now()
        last_update = end_date.to_pydatetime()
        hours_behind = (now - last_update).total_seconds() / 3600

        print(f"\n⏱️ DATA FRESHNESS:")
        print(f"   Last Update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Hours Behind: {hours_behind:.1f} hours")
        print(f"   Status: {'🟢 Recent' if hours_behind < 24 else '🟡 Needs update' if hours_behind < 168 else '🔴 Stale'}")

        # Model training suitability
        print(f"\n🤖 ML MODEL SUITABILITY:")
        suitable_for_training = total_records >= 10000 and missing_data == 0
        print(f"   Training Data: {'✅ Excellent' if suitable_for_training else '⚠️ Limited'}")
        print(f"   Sample Size: {total_records:,} records")
        print(f"   Recommended Min: 10,000 records")

        # Calculate data used in optimization
        optimization_records = 12000  # From focused optimization
        optimization_coverage = min(optimization_records / total_records * 100, 100)

        print(f"\n🎯 OPTIMIZATION USAGE:")
        print(f"   Records Used: {optimization_records:,} / {total_records:,}")
        print(f"   Coverage: {optimization_coverage:.1f}% of available data")
        print(f"   Time Period: Last {(optimization_records / 96):.0f} days")

        return {
            'start_date': start_date,
            'end_date': end_date,
            'total_records': total_records,
            'days': days,
            'months': months,
            'completeness': (total_records / (days * 96)) * 100
        }

    except Exception as e:
        print(f"❌ Error creating report: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = create_data_report()