#!/usr/bin/env python3
"""
Historical Data Downloader for BTC/USDT Trading System
Downloads 1+ year of 15-minute interval data as required by project specifications

Usage:
    python download_historical_data.py [--symbol BTCUSDT] [--interval 15m] [--days 365]
"""

import asyncio
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.binance_client import BinanceAPIClient
from src.data.database import TradingDatabase
from src.utils.config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_download.log', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

class HistoricalDataDownloader:
    """Manages comprehensive historical data download and storage"""

    def __init__(self):
        self.client = BinanceAPIClient()
        self.db = TradingDatabase()
        self.logger = logging.getLogger(__name__)

    async def download_historical_data(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "15m",
        days_back: int = 365,
        force_redownload: bool = False
    ) -> bool:
        """
        Download comprehensive historical data

        Args:
            symbol: Trading pair to download
            interval: Time interval for data
            days_back: Number of days to go back
            force_redownload: Whether to redownload existing data

        Returns:
            Success status
        """
        print("=" * 60)
        print(f"🚀 HISTORICAL DATA DOWNLOAD STARTED")
        print("=" * 60)
        print(f"Symbol: {symbol}")
        print(f"Interval: {interval}")
        print(f"Days back: {days_back}")
        print(f"Target data points: ~{(days_back * 24 * 60) // self._get_interval_minutes(interval):,}")

        # Check existing data
        min_date, max_date, count = self.db.get_data_range(symbol, interval)
        print(f"\n📊 EXISTING DATA STATUS:")
        print(f"Current records: {count:,}")
        if min_date and max_date:
            print(f"Date range: {min_date.strftime('%Y-%m-%d %H:%M')} to {max_date.strftime('%Y-%m-%d %H:%M')}")
            days_covered = (max_date - min_date).days
            print(f"Days covered: {days_covered}")

        # Determine what data to download
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)

        if not force_redownload and count > 0:
            # Smart incremental download
            if max_date and max_date < end_time - timedelta(hours=1):
                print(f"\n🔄 INCREMENTAL DOWNLOAD:")
                print(f"Updating data from {max_date.strftime('%Y-%m-%d %H:%M')} to present")
                start_time = max_date
            else:
                print(f"\n✅ DATA IS UP TO DATE")
                print(f"Latest data: {max_date.strftime('%Y-%m-%d %H:%M')}")
                return True

        # Download data
        success = await self._download_data_range(symbol, interval, start_time, end_time)

        if success:
            # Update statistics
            min_date, max_date, final_count = self.db.get_data_range(symbol, interval)
            print(f"\n🎉 DOWNLOAD COMPLETED SUCCESSFULLY!")
            print(f"Total records: {final_count:,}")
            print(f"Final date range: {min_date.strftime('%Y-%m-%d %H:%M')} to {max_date.strftime('%Y-%m-%d %H:%M')}")

            # Check for gaps
            await self._check_data_quality(symbol, interval)

        return success

    async def _download_data_range(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> bool:
        """Download data for a specific time range"""
        try:
            print(f"\n⬇️  DOWNLOADING DATA...")
            print(f"Period: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")

            total_days = (end_time - start_time).days
            current_start = start_time
            chunk_size_days = 30  # Download in 30-day chunks
            total_inserted = 0

            progress_bar_length = 40
            completed_chunks = 0
            total_chunks = (total_days + chunk_size_days - 1) // chunk_size_days

            while current_start < end_time:
                current_end = min(current_start + timedelta(days=chunk_size_days), end_time)

                # Update progress bar
                completed_chunks += 1
                progress = completed_chunks / total_chunks
                filled_length = int(progress_bar_length * progress)
                bar = '█' * filled_length + '-' * (progress_bar_length - filled_length)
                print(f'\rProgress: |{bar}| {progress*100:.1f}% ({completed_chunks}/{total_chunks} chunks)', end='', flush=True)

                # Fetch data chunk
                chunk_data = await self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=current_start.strftime('%Y-%m-%d %H:%M:%S'),
                    end_str=current_end.strftime('%Y-%m-%d %H:%M:%S'),
                    limit=1000
                )

                if chunk_data:
                    # Store in database
                    inserted_count = self.db.insert_ohlcv_data(chunk_data, symbol, interval)
                    total_inserted += inserted_count

                    # Log this chunk
                    if chunk_data:
                        start_ts = chunk_data[0]['timestamp']
                        end_ts = chunk_data[-1]['timestamp']
                        self.db.log_data_collection(symbol, interval, start_ts, end_ts, len(chunk_data))

                current_start = current_end
                # Small delay to respect API limits
                await asyncio.sleep(0.1)

            print(f"\n✅ Downloaded {total_inserted:,} new records")
            return True

        except Exception as e:
            self.logger.error(f"Error downloading data: {e}")
            print(f"❌ Error downloading data: {e}")
            return False

    async def _check_data_quality(self, symbol: str, interval: str):
        """Check data quality and identify gaps"""
        print(f"\n🔍 DATA QUALITY CHECK:")

        # Check for gaps
        interval_minutes = self._get_interval_minutes(interval)
        gaps = self.db.get_missing_data_gaps(symbol, interval, interval_minutes)

        if gaps:
            print(f"⚠️  Found {len(gaps)} data gaps:")
            for i, (start_gap, end_gap) in enumerate(gaps[:5]):  # Show first 5 gaps
                duration = end_gap - start_gap
                print(f"   Gap {i+1}: {start_gap.strftime('%Y-%m-%d %H:%M')} to {end_gap.strftime('%Y-%m-%d %H:%M')} ({duration})")
            if len(gaps) > 5:
                print(f"   ... and {len(gaps) - 5} more gaps")
        else:
            print("✅ No significant data gaps found")

        # Get overall statistics
        stats = self.db.get_database_stats()
        print(f"\n📈 DATABASE STATISTICS:")
        print(f"Database size: {stats['database_size_mb']:.2f} MB")
        for stat in stats['symbol_stats']:
            if stat['symbol'] == symbol and stat['interval'] == interval:
                print(f"Records for {symbol} {interval}: {stat['count']:,}")
                if stat['start_date'] and stat['end_date']:
                    total_days = (stat['end_date'] - stat['start_date']).days
                    expected_records = (total_days * 24 * 60) // interval_minutes
                    coverage = (stat['count'] / expected_records) * 100 if expected_records > 0 else 0
                    print(f"Data coverage: {coverage:.1f}% ({total_days} days)")

    def _get_interval_minutes(self, interval: str) -> int:
        """Convert interval string to minutes"""
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval.endswith('d'):
            return int(interval[:-1]) * 24 * 60
        else:
            return 15  # Default

    async def fill_data_gaps(self, symbol: str, interval: str, max_gap_hours: int = 24):
        """Fill identified data gaps"""
        print(f"\n🔧 FILLING DATA GAPS (gaps > {max_gap_hours} hours)...")

        interval_minutes = self._get_interval_minutes(interval)
        gaps = self.db.get_missing_data_gaps(symbol, interval, interval_minutes)

        filled_gaps = 0
        for gap_start, gap_end in gaps:
            gap_duration = gap_end - gap_start
            if gap_duration.total_seconds() / 3600 > max_gap_hours:  # Only fill large gaps
                print(f"Filling gap: {gap_start.strftime('%Y-%m-%d %H:%M')} to {gap_end.strftime('%Y-%m-%d %H:%M')}")

                success = await self._download_data_range(symbol, interval, gap_start, gap_end)
                if success:
                    filled_gaps += 1

                await asyncio.sleep(0.5)  # Extra delay for gap filling

        print(f"✅ Filled {filled_gaps} data gaps")

async def main():
    """Main download function"""
    parser = argparse.ArgumentParser(description='Download historical crypto trading data')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--interval', default='15m', help='Time interval (default: 15m)')
    parser.add_argument('--days', type=int, default=365, help='Days to go back (default: 365)')
    parser.add_argument('--force', action='store_true', help='Force redownload existing data')
    parser.add_argument('--fill-gaps', action='store_true', help='Fill data gaps after download')

    args = parser.parse_args()

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    downloader = HistoricalDataDownloader()

    # Check API client status
    status = downloader.client.get_client_status()
    if not status['initialized']:
        print("❌ ERROR: Binance API client not initialized.")
        print("Please set up your API keys in .env file:")
        print("BINANCE_API_KEY=your_key")
        print("BINANCE_SECRET_KEY=your_secret")
        print("BINANCE_TESTNET=true")
        return False

    print(f"🔑 API Status: {'Testnet' if status['testnet'] else 'Live'} mode")

    try:
        # Download historical data
        success = await downloader.download_historical_data(
            symbol=args.symbol,
            interval=args.interval,
            days_back=args.days,
            force_redownload=args.force
        )

        if success and args.fill_gaps:
            await downloader.fill_data_gaps(args.symbol, args.interval)

        if success:
            print(f"\n🎯 SUCCESS: Historical data download completed!")
            print(f"Next step: Run feature engineering pipeline")
            return True
        else:
            print(f"\n❌ FAILED: Data download encountered errors")
            return False

    except KeyboardInterrupt:
        print(f"\n⏹️  Download interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 BTC/USDT Historical Data Downloader")
    print("=" * 60)
    success = asyncio.run(main())
    sys.exit(0 if success else 1)