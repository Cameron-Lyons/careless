"""This module contains the evenodd function for calculating even-odd consistency scores."""

import numpy as np
from typing import List, Union, Tuple


def calculate_correlations(even_cols: np.ndarray, odd_cols: np.ndarray) -> np.ndarray:
    """Calculates correlations between even and odd columns."""
    # Ensure we only compare up to the smallest pair count
    min_cols = min(even_cols.shape[1], odd_cols.shape[1])
    correlations = np.array(
        [
            np.corrcoef(even_cols[:, i], odd_cols[:, i])[0, 1] if min_cols > 0 else 0
            for i in range(min_cols)
        ]
    )
    correlations[np.isnan(correlations)] = 0  # handle NaN
    return correlations


def evenodd(
    x: Union[List[List[float]], np.ndarray], factors: List[int], diag: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate even-odd consistency scores for each individual based on the provided factors.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their responses.
    - factors: List of integers specifying the length of each factor in the dataset.
    - diag: Boolean to optionally return a column with the number of available even/odd pairs.

    Returns:
    - A numpy array of even-odd consistency scores or a tuple of scores and diagnostic values.
    """

    x = np.array(x)
    num_factors = len(factors)
    avg_correlations = np.zeros(x.shape[0])
    diag_vals = np.zeros(x.shape[0], dtype=int)

    start_col = 0
    for factor_size in factors:
        end_col = start_col + factor_size
        even_cols = x[:, start_col:end_col:2]
        odd_cols = x[:, start_col + 1 : end_col : 2]

        corrs = calculate_correlations(even_cols, odd_cols)
        avg_correlations += corrs
        diag_vals += len(
            corrs
        )  # Update to reflect actual number of correlations calculated

        start_col = end_col

    if num_factors:
        avg_correlations /= num_factors

    return (avg_correlations, diag_vals) if diag else avg_correlations
