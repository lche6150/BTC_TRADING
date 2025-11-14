#!/usr/bin/env python3
"""
Compute live paper-trader prediction accuracy from a signal log.

Usage:
    PYTHONPATH=. python scripts/calc_paper_accuracy.py \
        --log results/live_trading/btcusdt_paper_log_20251109_141946.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


BUCKETS = [
    (0.0, 0.50, "<0.50"),
    (0.50, 0.60, "0.50-0.59"),
    (0.60, 0.70, "0.60-0.69"),
    (0.70, 0.80, "0.70-0.79"),
    (0.80, 0.90, "0.80-0.89"),
    (0.90, 1.01, "0.90-1.00"),
]


def bucket_label(value: float) -> str:
    for low, high, label in BUCKETS:
        if low <= value < high:
            return label
    return "UNKNOWN"

def load_records(path: Path):
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute accuracy from paper trader log.")
    parser.add_argument("--log", required=True, type=Path, help="Path to paper_log_*.txt")
    args = parser.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(f"Log file not found: {args.log}")

    total = hits = misses = 0
    per_direction = defaultdict(lambda: Counter(hit=0, miss=0))
    per_bucket = defaultdict(lambda: Counter(hit=0, miss=0))

    for record in load_records(args.log):
        status = record.get("evaluation_status")
        if status not in ("hit", "miss"):
            continue
        direction = record.get("direction") or "UNKNOWN"
        confidence_value = record.get("confidence")
        bucket = bucket_label(confidence_value) if isinstance(confidence_value, (int, float)) else "UNKNOWN"
        total += 1
        if status == "hit":
            hits += 1
            per_direction[direction]["hit"] += 1
            per_bucket[bucket]["hit"] += 1
        else:
            misses += 1
            per_direction[direction]["miss"] += 1
            per_bucket[bucket]["miss"] += 1

    if total == 0:
        print("No evaluable signals (hit/miss) found in log.")
        return

    accuracy = hits / total
    print(f"Log: {args.log}")
    print(f"Total evaluated signals: {total}")
    print(f"Hits: {hits}")
    print(f"Misses: {misses}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    print("\nPer direction:")
    for direction, stats in per_direction.items():
        dir_total = stats["hit"] + stats["miss"]
        if dir_total == 0:
            continue
        dir_acc = stats["hit"] / dir_total
        print(
            f"  {direction}: total={dir_total} hits={stats['hit']} "
            f"misses={stats['miss']} accuracy={dir_acc*100:.2f}%"
        )

    print("\nPer confidence bucket:")
    for _, _, label in BUCKETS:
        stats = per_bucket.get(label)
        if not stats:
            continue
        bucket_total = stats["hit"] + stats["miss"]
        if bucket_total == 0:
            continue
        bucket_acc = stats["hit"] / bucket_total
        print(
            f"  {label}: total={bucket_total} hits={stats['hit']} "
            f"misses={stats['miss']} accuracy={bucket_acc*100:.2f}%"
        )


if __name__ == "__main__":
    main()
