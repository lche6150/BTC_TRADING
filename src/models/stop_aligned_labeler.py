#!/usr/bin/env python3
"""Label generation utilities aligned with trading stop/target logic."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def generate_stop_aligned_labels(
    features: pd.DataFrame,
    future_periods: int = 20,
    stop_low: float = 0.003,
    stop_high: float = 0.02,
    target_multiplier: float = 1.75,
) -> pd.Series:
    """
    Produce binary labels that reflect whether a long or short trade would
    reach its target before the corresponding stop within `future_periods`.

    Returns a Series containing:
        1  -> long target hits before long stop within horizon
        0  -> short target hits before short stop within horizon
       -1 -> neither side produces a clean outcome (discard)
    """
    if "close" not in features.columns:
        raise ValueError("Features must include 'close' price column.")

    close = features["close"].astype(float).values
    vol_series: Optional[pd.Series] = None
    if "volatility_1h" in features.columns:
        vol_series = features["volatility_1h"]
    elif "volatility" in features.columns:
        vol_series = features["volatility"]

    if vol_series is not None:
        vol = vol_series.fillna(stop_low).abs().values
    else:
        vol = np.full(len(features), stop_low)

    n = len(features)
    labels = np.full(n, -1, dtype=int)

    max_index = n - future_periods
    for i in range(max_index):
        entry_price = close[i]
        if entry_price <= 0:
            continue

        stop_pct = float(np.clip(vol[i], stop_low, stop_high))

        long_stop = entry_price * (1 - stop_pct)
        long_target = entry_price * (1 + stop_pct * target_multiplier)

        short_stop = entry_price * (1 + stop_pct)
        short_target = entry_price * (1 - stop_pct * target_multiplier)

        long_event = None
        short_event = None
        long_step = None
        short_step = None

        for step in range(1, future_periods + 1):
            price = close[i + step]

            if long_event is None:
                if price <= long_stop:
                    long_event = "stop"
                    long_step = step
                elif price >= long_target:
                    long_event = "target"
                    long_step = step

            if short_event is None:
                if price >= short_stop:
                    short_event = "stop"
                    short_step = step
                elif price <= short_target:
                    short_event = "target"
                    short_step = step

            if long_event and short_event:
                break

        long_success_step = long_step if long_event == "target" else None
        short_success_step = short_step if short_event == "target" else None

        if long_success_step is not None and (
            short_success_step is None or long_success_step <= short_success_step
        ):
            labels[i] = 1
        elif short_success_step is not None and (
            long_success_step is None or short_success_step < long_success_step
        ):
            labels[i] = 0
        else:
            labels[i] = -1

    return pd.Series(labels, index=features.index)


__all__ = ["generate_stop_aligned_labels"]
