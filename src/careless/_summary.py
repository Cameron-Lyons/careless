"""Shared summary statistics utilities for careless detection functions."""

from typing import Any

import numpy as np


def calculate_summary_stats(
    values: np.ndarray,
    suffix: str = "",
) -> dict[str, Any]:
    """
    Calculate common summary statistics for an array of values.

    Parameters:
    - values: Array of values (may contain NaN)
    - suffix: Optional suffix for dictionary keys (e.g., "_score" -> "mean_score")

    Returns:
    - Dictionary with mean, std, min, max, median statistics
    """
    valid_values = values[~np.isnan(values)]

    if len(valid_values) == 0:
        return {
            f"mean{suffix}": np.nan,
            f"std{suffix}": np.nan,
            f"min{suffix}": np.nan,
            f"max{suffix}": np.nan,
            f"median{suffix}": np.nan,
        }

    return {
        f"mean{suffix}": float(np.mean(valid_values)),
        f"std{suffix}": float(np.std(valid_values)),
        f"min{suffix}": float(np.min(valid_values)),
        f"max{suffix}": float(np.max(valid_values)),
        f"median{suffix}": float(np.median(valid_values)),
    }
