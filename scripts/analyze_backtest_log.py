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

import numpy as np
import re

HEADER_PATTERN = re.compile(
    r"TRADING LOG - (?P<start>[A-Za-z]{3,}\s+\d{1,2},\s+\d{4}) - (?P<end>[A-Za-z]{3,}\s+\d{1,2},\s+\d{4})"
)
ENTRY_PATTERN = re.compile(r"Entry Time:\s*(\d{2}/\d{2})\s+(\d{2}:\d{2})")
EXIT_PATTERN = re.compile(r"Exit\s+\d+:\s*(\d{2}/\d{2})\s+(\d{2}:\d{2})")
NET_PNL_PATTERN = re.compile(r"Net P&L \(all fees\):\s*\$([+\-]?[0-9,\.]+)")
TRADE_HEADER_PATTERN = re.compile(r"^Trade\s+(\d+)\b")
CONFIDENCE_PATTERN = re.compile(r"Confidence:\s*([0-9]+\.?[0-9]*)%")


@dataclass
class TradeRecord:
    trade_id: int
    entry_time: datetime
    exit_time: datetime
    net_pnl: float
    hold_minutes: float
    confidence: Optional[float]


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
    current_entry: Optional[datetime] = None
    current_exit: Optional[datetime] = None
    current_net: Optional[float] = None
    current_confidence: Optional[float] = None

    def finalize() -> None:
        nonlocal current_trade_id, current_entry, current_exit, current_net, current_confidence
        if (
            current_trade_id is not None
            and current_entry is not None
            and current_exit is not None
            and current_net is not None
        ):
            hold_minutes = max(
                0.0,
                (current_exit - current_entry).total_seconds() / 60.0,
            )
            records.append(
                TradeRecord(
                    current_trade_id,
                    current_entry,
                    current_exit,
                    current_net,
                    hold_minutes,
                    current_confidence,
                )
            )
        current_trade_id = None
        current_entry = None
        current_exit = None
        current_net = None
        current_confidence = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("SUMMARY:"):
            finalize()
            break

        header_match = TRADE_HEADER_PATTERN.match(line)
        if header_match:
            finalize()
            current_trade_id = int(header_match.group(1))
            continue

        if current_trade_id is None:
            continue

        match = ENTRY_PATTERN.search(line)
        if match:
            month_day, time_str = match.groups()
            current_entry = resolve_timestamp(month_day, time_str, start_date, end_date)
            continue

        match = EXIT_PATTERN.search(line)
        if match:
            month_day, time_str = match.groups()
            current_exit = resolve_timestamp(month_day, time_str, start_date, end_date)
            continue

        match = CONFIDENCE_PATTERN.search(line)
        if match:
            try:
                current_confidence = float(match.group(1)) / 100.0
            except ValueError:
                current_confidence = None
            continue

        match = NET_PNL_PATTERN.search(line)
        if match:
            try:
                current_net = float(match.group(1).replace(",", ""))
            except ValueError:
                current_net = 0.0

    finalize()
    return records


def summarize_daily(trades: Iterable[TradeRecord]) -> List[Tuple[datetime.date, Dict[str, float]]]:
    daily: Dict[datetime.date, Dict[str, float]] = defaultdict(
        lambda: {
            "trades": 0,
            "wins": 0,
            "total_pnl": 0.0,
            "total_hold_minutes": 0.0,
            "confidence_buckets": defaultdict(lambda: {"wins": 0, "count": 0, "total_pnl": 0.0}),
        }
    )
    for trade in trades:
        bucket = daily[trade.exit_time.date()]
        bucket["trades"] += 1
        if trade.net_pnl > 0:
            bucket["wins"] += 1
        bucket["total_pnl"] += trade.net_pnl
        bucket["total_hold_minutes"] += trade.hold_minutes
        add_confidence_sample(bucket["confidence_buckets"], trade.confidence, trade.net_pnl)

    ordered = sorted(daily.items())
    for _, stats in ordered:
        stats["win_rate"] = stats["wins"] / stats["trades"] if stats["trades"] else 0.0
        stats["average_hold_minutes"] = (
            stats["total_hold_minutes"] / stats["trades"] if stats["trades"] else 0.0
        )
        stats["confidence_buckets"] = summarize_confidence_bucket(stats["confidence_buckets"])
    return ordered


def summarize_weekly(trades: Iterable[TradeRecord]) -> List[Tuple[Tuple[datetime.date, datetime.date], Dict[str, float]]]:
    weekly: Dict[Tuple[datetime.date, datetime.date], Dict[str, float]] = defaultdict(
        lambda: {
            "trades": 0,
            "wins": 0,
            "total_pnl": 0.0,
            "total_hold_minutes": 0.0,
            "confidence_buckets": defaultdict(lambda: {"wins": 0, "count": 0, "total_pnl": 0.0}),
        }
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
        bucket["total_hold_minutes"] += trade.hold_minutes
        add_confidence_sample(bucket["confidence_buckets"], trade.confidence, trade.net_pnl)

    ordered = sorted(weekly.items())
    for _, stats in ordered:
        stats["win_rate"] = stats["wins"] / stats["trades"] if stats["trades"] else 0.0
        stats["average_hold_minutes"] = (
            stats["total_hold_minutes"] / stats["trades"] if stats["trades"] else 0.0
        )
        stats["confidence_buckets"] = summarize_confidence_bucket(stats["confidence_buckets"])
    return ordered


def summarize_monthly(trades: Iterable[TradeRecord]) -> List[Tuple[Tuple[int, int], Dict[str, float]]]:
    monthly: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(
        lambda: {
            "trades": 0,
            "wins": 0,
            "total_pnl": 0.0,
            "total_hold_minutes": 0.0,
            "confidence_buckets": defaultdict(lambda: {"wins": 0, "count": 0, "total_pnl": 0.0}),
        }
    )
    for trade in trades:
        key = (trade.exit_time.year, trade.exit_time.month)
        bucket = monthly[key]
        bucket["trades"] += 1
        if trade.net_pnl > 0:
            bucket["wins"] += 1
        bucket["total_pnl"] += trade.net_pnl
        bucket["total_hold_minutes"] += trade.hold_minutes
        add_confidence_sample(bucket["confidence_buckets"], trade.confidence, trade.net_pnl)

    ordered = sorted(monthly.items())
    for _, stats in ordered:
        stats["win_rate"] = stats["wins"] / stats["trades"] if stats["trades"] else 0.0
        stats["average_hold_minutes"] = (
            stats["total_hold_minutes"] / stats["trades"] if stats["trades"] else 0.0
        )
        stats["confidence_buckets"] = summarize_confidence_bucket(stats["confidence_buckets"])
    return ordered


def add_confidence_sample(bucket_map, confidence: Optional[float], net_pnl: float) -> None:
    if confidence is None or not np.isfinite(confidence):
        return
    if confidence < 0.6:
        key = "<60%"
    elif confidence < 0.70:
        key = "60-70%"
    elif confidence < 0.75:
        key = "70-75%"
    elif confidence < 0.76:
        key = "75-76%"
    elif confidence < 0.77:
        key = "76-77%"
    elif confidence < 0.78:
        key = "77-78%"
    elif confidence < 0.79:
        key = "78-79%"
    elif confidence < 0.8:
        key = "79-80%"
    elif confidence < 0.81:
        key = "80-81%"
    elif confidence < 0.82:
        key = "81-82%"
    elif confidence < 0.83:
        key = "82-83%"
    elif confidence < 0.84:
        key = "83-84%"
    elif confidence < 0.85:
        key = "84-85%"
    elif confidence < 0.9:
        key = "85-90%"
    else:
        key = "90-100%"
    entry = bucket_map[key]
    entry["count"] += 1
    if net_pnl > 0:
        entry["wins"] += 1
    entry["total_pnl"] += net_pnl


def summarize_confidence_bucket(bucket_map):
    summary = {}
    for key, stats in bucket_map.items():
        count = stats["count"]
        wins = stats["wins"]
        summary[key] = {
            "trades": count,
            "wins": wins,
            "win_rate": wins / count if count else 0.0,
            "total_pnl": stats.get("total_pnl", 0.0),
        }
    return summary


def summarize_overall_confidence(trades: Iterable[TradeRecord]):
    bucket_map = defaultdict(lambda: {"wins": 0, "count": 0, "total_pnl": 0.0})
    for trade in trades:
        add_confidence_sample(bucket_map, trade.confidence, trade.net_pnl)
    return summarize_confidence_bucket(bucket_map)


def format_daily_section(daily_stats: List[Tuple[datetime.date, Dict[str, float]]]) -> str:
    if not daily_stats:
        return "Daily Performance:\n  No trades found.\n"

    lines = ["Daily Performance:"]
    for day, stats in daily_stats:
        lines.append(
            f"  {day}: trades={stats['trades']}, wins={stats['wins']}, "
            f"win_rate={stats['win_rate']*100:.1f}%, net_pnl={stats['total_pnl']:.2f}, "
            f"avg_hold={stats['average_hold_minutes']:.1f}m"
        )
    return "\n".join(lines) + "\n"


def format_weekly_section(weekly_stats: List[Tuple[Tuple[datetime.date, datetime.date], Dict[str, float]]]) -> str:
    if not weekly_stats:
        return ""
    lines = ["Weekly Performance (week starting Monday):"]
    for (start, end), stats in weekly_stats:
        lines.append(
            f"  {start} → {end}: trades={stats['trades']}, wins={stats['wins']}, "
            f"win_rate={stats['win_rate']*100:.1f}%, net_pnl={stats['total_pnl']:.2f}, "
            f"avg_hold={stats['average_hold_minutes']:.1f}m"
        )
    return "\n".join(lines) + "\n"


def format_monthly_section(monthly_stats: List[Tuple[Tuple[int, int], Dict[str, float]]]) -> str:
    if not monthly_stats:
        return ""
    lines = ["Monthly Performance:"]
    for (year, month), stats in monthly_stats:
        lines.append(
            f"  {year}-{month:02d}: trades={stats['trades']}, wins={stats['wins']}, "
            f"win_rate={stats['win_rate']*100:.1f}%, net_pnl={stats['total_pnl']:.2f}, "
            f"avg_hold={stats['average_hold_minutes']:.1f}m"
        )
    return "\n".join(lines) + "\n"


def format_confidence_section(bucket_summary: Dict[str, Dict[str, float]]) -> str:
    if not bucket_summary:
        return ""
    order = ["60-70%", "70-80%", "80-90%", "90-100%"]
    # Include other keys if present (e.g., "<60%")
    extra_keys = [k for k in sorted(bucket_summary.keys()) if k not in order]
    ordered_keys = [k for k in order if k in bucket_summary] + extra_keys
    lines = ["Confidence Buckets:"]
    for key in ordered_keys:
        stats = bucket_summary[key]
        lines.append(
            f"  {key}: trades={stats['trades']}, wins={stats['wins']}, "
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

        monthly_stats = summarize_monthly(trades)
        if monthly_stats:
            sections.append(format_monthly_section(monthly_stats))

        confidence_summary = summarize_overall_confidence(trades)
        confidence_section = format_confidence_section(confidence_summary)
        if confidence_section:
            sections.append(confidence_section)

        output = "\n".join(sections).rstrip() + "\n"

    backtest_dir.mkdir(parents=True, exist_ok=True)
    summary_path = backtest_dir / args.output_name
    summary_path.write_text(output)


if __name__ == "__main__":
    main()
