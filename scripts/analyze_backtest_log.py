#!/usr/bin/env python3
"""
Analyze a backtest folder by parsing its trading_log.txt and saving daily/weekly stats.

Usage:
    python scripts/analyze_backtest_log.py --subdir 20251010_to_20251022

By default the script looks inside results/backtests/<subdir>/trading_log.txt,
computes daily PnL/win-rate summaries (and weekly if the range spans enough days),
and writes the report to results/backtests/<subdir>/performance_summary.txt.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re

HEADER_PATTERN = re.compile(
    r"TRADING LOG - (?P<start>[A-Za-z]{3,}\s+\d{1,2},\s+\d{4}) - (?P<end>[A-Za-z]{3,}\s+\d{1,2},\s+\d{4})"
)
ENTRY_PATTERN = re.compile(r"Entry Time:\s*(\d{2}/\d{2})\s+(\d{2}:\d{2})")
EXIT_PATTERN = re.compile(r"Exit\s+\d+:\s*(\d{2}/\d{2})\s+(\d{2}:\d{2})")
NET_PNL_PATTERN = re.compile(r"Net P&L \(all fees\):\s*\$([+\-]?[0-9,\.]+)")


@dataclass
class TradeRecord:
    trade_id: int
    exit_time: datetime
    net_pnl: float


def parse_header_dates(line: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    match = HEADER_PATTERN.search(line or "")
    if not match:
        return None, None
    start = datetime.strptime(match.group("start"), "%b %d, %Y")
    end = datetime.strptime(match.group("end"), "%b %d, %Y")
    return start, end


def resolve_timestamp(
    month_day: str,
    time_str: str,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
) -> datetime:
    month, day = map(int, month_day.split("/"))
    hour, minute = map(int, time_str.split(":"))

    base_year = start_date.year if start_date else datetime.utcnow().year
    candidate = datetime(base_year, month, day, hour, minute)

    if start_date and end_date:
        # Allow for logs that straddle year boundaries; adjust if out of range
        tolerance = timedelta(days=180)
        if candidate < start_date - tolerance:
            candidate = candidate.replace(year=candidate.year + 1)
        if candidate > end_date + tolerance:
            candidate = candidate.replace(year=candidate.year - 1)

        if not (start_date <= candidate <= end_date):
            alt = datetime(end_date.year, month, day, hour, minute)
            if start_date <= alt <= end_date:
                candidate = alt

    return candidate


def parse_trading_log(log_path: Path) -> List[TradeRecord]:
    if not log_path.exists():
        raise FileNotFoundError(f"Trading log not found: {log_path}")

    lines = log_path.read_text().splitlines()
    if not lines:
        return []

    start_date, end_date = parse_header_dates(lines[0])

    records: List[TradeRecord] = []
    current_trade_id: Optional[int] = None
    current_exit: Optional[datetime] = None
    current_net: Optional[float] = None

    def finalize():
        nonlocal current_trade_id, current_exit, current_net
        if current_trade_id is not None and current_exit is not None and current_net is not None:
            records.append(TradeRecord(current_trade_id, current_exit, current_net))
        current_trade_id = None
        current_exit = None
        current_net = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("SUMMARY:"):
            finalize()
            break

        if line.startswith("Trade "):
            finalize()
            try:
                current_trade_id = int(line.split()[1])
            except (IndexError, ValueError):
                current_trade_id = len(records) + 1
            continue

        if current_trade_id is None:
            continue

        exit_match = EXIT_PATTERN.search(line)
        if exit_match:
            month_day, time_str = exit_match.groups()
            current_exit = resolve_timestamp(month_day, time_str, start_date, end_date)
            continue

        pnl_match = NET_PNL_PATTERN.search(line)
        if pnl_match:
            value = pnl_match.group(1).replace(",", "")
            try:
                current_net = float(value)
            except ValueError:
                current_net = 0.0
            continue

    finalize()
    return records


def summarize_daily(trades: Iterable[TradeRecord]) -> List[Tuple[datetime.date, Dict[str, float]]]:
    daily: Dict[datetime.date, Dict[str, float]] = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0.0})
    for trade in trades:
        bucket = daily[trade.exit_time.date()]
        bucket["trades"] += 1
        if trade.net_pnl > 0:
            bucket["wins"] += 1
        bucket["total_pnl"] += trade.net_pnl

    ordered = sorted(daily.items())
    for _, stats in ordered:
        stats["win_rate"] = stats["wins"] / stats["trades"] if stats["trades"] else 0.0
    return ordered


def summarize_weekly(trades: Iterable[TradeRecord]) -> List[Tuple[Tuple[datetime.date, datetime.date], Dict[str, float]]]:
    weekly: Dict[Tuple[datetime.date, datetime.date], Dict[str, float]] = defaultdict(
        lambda: {"trades": 0, "wins": 0, "total_pnl": 0.0}
    )
    for trade in trades:
        date = trade.exit_time.date()
        week_start = date - timedelta(days=date.weekday())  # Monday
        week_end = week_start + timedelta(days=6)
        bucket = weekly[(week_start, week_end)]
        bucket["trades"] += 1
        if trade.net_pnl > 0:
            bucket["wins"] += 1
        bucket["total_pnl"] += trade.net_pnl

    ordered = sorted(weekly.items())
    for _, stats in ordered:
        stats["win_rate"] = stats["wins"] / stats["trades"] if stats["trades"] else 0.0
    return ordered


def format_daily_section(daily_stats: List[Tuple[datetime.date, Dict[str, float]]]) -> str:
    if not daily_stats:
        return "Daily Performance:\n  No trades found.\n"

    lines = ["Daily Performance:"]
    for day, stats in daily_stats:
        lines.append(
            f"  {day}: trades={stats['trades']}, wins={stats['wins']}, "
            f"win_rate={stats['win_rate']*100:.1f}%, net_pnl={stats['total_pnl']:.2f}"
        )
    return "\n".join(lines) + "\n"


def format_weekly_section(weekly_stats: List[Tuple[Tuple[datetime.date, datetime.date], Dict[str, float]]]) -> str:
    if not weekly_stats:
        return ""
    lines = ["Weekly Performance (week starting Monday):"]
    for (start, end), stats in weekly_stats:
        lines.append(
            f"  {start} → {end}: trades={stats['trades']}, wins={stats['wins']}, "
            f"win_rate={stats['win_rate']*100:.1f}%, net_pnl={stats['total_pnl']:.2f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize backtest results from trading_log.txt.")
    parser.add_argument(
        "--subdir",
        required=True,
        help="Backtest subfolder name under results/backtests (e.g., 20251010_to_20251022).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/backtests"),
        help="Base directory that contains backtest subfolders (default: results/backtests).",
    )
    parser.add_argument(
        "--min-days-for-weekly",
        type=int,
        default=30,
        help="Minimum span (days) required to include weekly stats (default: 30).",
    )
    parser.add_argument(
        "--output-name",
        default="performance_summary.txt",
        help="Filename for the saved summary (default: performance_summary.txt).",
    )
    args = parser.parse_args()

    backtest_dir = args.base_dir / args.subdir
    log_path = backtest_dir / "trading_log.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"Expected trading log at {log_path}")

    trades = parse_trading_log(log_path)
    if not trades:
        output = "No trades parsed from trading_log.txt.\n"
    else:
        daily_stats = summarize_daily(trades)
        sections = [format_daily_section(daily_stats)]

        span_days = (max(t.exit_time for t in trades) - min(t.exit_time for t in trades)).days
        if span_days >= args.min_days_for_weekly:
            weekly_stats = summarize_weekly(trades)
            if weekly_stats:
                sections.append(format_weekly_section(weekly_stats))

        output = "\n".join(sections).rstrip() + "\n"

    backtest_dir.mkdir(parents=True, exist_ok=True)
    summary_path = backtest_dir / args.output_name
    summary_path.write_text(output)


if __name__ == "__main__":
    main()
