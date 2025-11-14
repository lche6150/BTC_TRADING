#!/usr/bin/env python3
"""Analyze confidence distribution inside a signal log JSONL file."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


BUCKETS: List[Tuple[float, float, str]] = [
    (0.0, 0.50, "<0.50"),
    (0.50, 0.60, "0.50-0.59"),
    (0.60, 0.70, "0.60-0.69"),
    (0.70, 0.80, "0.70-0.79"),
    (0.80, 0.90, "0.80-0.89"),
    (0.90, 1.01, "0.90-1.00"),
]


def bucket_label(conf: float) -> str:
    for low, high, label in BUCKETS:
        if low <= conf < high:
            return label
    return ">=1.0"


def analyze_log(path: Path) -> None:
    confidences: List[float] = []
    statuses = Counter()
    reasons = Counter()

    with path.open("r") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            conf = rec.get("confidence")
            if isinstance(conf, str):
                try:
                    conf = float(conf)
                except ValueError:
                    conf = None
            if isinstance(conf, (int, float)):
                confidences.append(float(conf))
            status = rec.get("status") or ("accepted" if rec.get("executed") else "unknown")
            statuses[status] += 1
            reason = rec.get("reason")
            if reason:
                reasons[reason] += 1

    if not confidences:
        print("No confidence values found in log")
        return

    arr = np.array(confidences)
    print(f"Total confidence entries: {len(arr)}")
    print(f"Mean: {arr.mean():.4f}, Median: {np.median(arr):.4f}, Std: {arr.std():.4f}")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"Quantile {q:.2f}: {np.quantile(arr, q):.4f}")

    bucket_counts = Counter(bucket_label(c) for c in confidences)
    print("\nConfidence buckets:")
    total = len(confidences)
    for _, _, label in BUCKETS:
        count = bucket_counts.get(label, 0)
        pct = (count / total) * 100 if total else 0.0
        print(f"  {label}: {count} ({pct:.1f}%)")

    print("\nStatus counts:")
    total_status = sum(statuses.values())
    for status, count in statuses.most_common():
        pct = (count / total_status) * 100 if total_status else 0.0
        print(f"  {status}: {count} ({pct:.1f}%)")

    if reasons:
        print("\nRejection reasons:")
        total_reject = sum(reasons.values())
        for reason, count in reasons.most_common():
            pct = (count / total_reject) * 100 if total_reject else 0.0
            print(f"  {reason}: {count} ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze confidence distribution in a signal log")
    parser.add_argument("--log", required=True, type=Path, help="Path to signal_log.jsonl or paper_log.txt")
    args = parser.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(f"Log file not found: {args.log}")

    analyze_log(args.log)


if __name__ == "__main__":
    main()
