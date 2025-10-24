#!/usr/bin/env python3
"""
Sum realized PnL in a paper trading log starting from a given line.

Usage:
    python scripts/sum_paper_pnl.py --file results/live_trading/paper_trades.log --start-line 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator


# script to sum realized pnl from paper_trades.log

def iter_log(path: Path, start_line: int) -> Iterator[dict]:
    with path.open() as fh:
        for idx, line in enumerate(fh, start=1):
            if idx < start_line:
                continue
            if line.strip() == "":
                break
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Sum realized PnL from a paper trades log.")
    parser.add_argument("--file", required=True, type=Path, help="Path to paper_trades.log")
    parser.add_argument("--start-line", type=int, default=1, help="1-based line number to start summing from")
    args = parser.parse_args()

    total_pnl = 0.0
    exit_count = 0
    entry_count = 0
    closed_count = 0
    positions: Dict[str, Dict[str, float]] = {}

    for record in iter_log(args.file, args.start_line):
        if record.get("type") == "EXIT":
            total_pnl += float(record.get("realized_pnl", 0.0))
            exit_count += 1
            pos_id = record.get("position_id")
            remaining = float(record.get("remaining_size", 0.0))
            if pos_id and pos_id in positions:
                if remaining <= 1e-6:
                    closed_count += 1
                    positions.pop(pos_id, None)
                else:
                    positions[pos_id]["remaining_size"] = remaining
                    positions[pos_id]["last_exit_price"] = float(record.get("exit_price", record.get("price", 0.0)))
        elif record.get("type") == "ENTRY":
            pos_id = record.get("position_id")
            if pos_id:
                entry_count += 1
                positions[pos_id] = {
                    "direction": record.get("direction"),
                    "entry_price": float(record.get("price", 0.0)),
                    "remaining_size": 1.0,
                }

    open_positions = positions

    print(f"Summed realized PnL from line {args.start_line}: {total_pnl:.2f} across {exit_count} exits")
    print(f"Entries processed: {entry_count}")
    print(f"Closed trades: {closed_count}")
    print(f"Open trades: {len(open_positions)}")

    if open_positions:
        print("\nOpen position details:")
        for pos_id, info in open_positions.items():
            direction = info.get("direction")
            entry_price = info.get("entry_price")
            remaining = info.get("remaining_size")
            print(f"  {pos_id}: direction={direction}, entry_price={entry_price:.2f}, remaining_size={remaining:.4f}")


if __name__ == "__main__":
    main()
