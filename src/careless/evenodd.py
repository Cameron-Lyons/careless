"""This module contains the evenodd function for calculating even-odd consistency scores."""

import warnings

import numpy as np


def calculate_correlations(even_cols: np.ndarray, odd_cols: np.ndarray) -> np.ndarray:
    """
    Calculates correlations between even and odd columns for each individual.

    Parameters:
    - even_cols: Array of even-indexed columns (rows are individuals)
    - odd_cols: Array of odd-indexed columns (rows are individuals)

    Returns:
    - Array of correlation coefficients for each individual
    """
    if even_cols.shape[0] != odd_cols.shape[0]:
        raise ValueError("even_cols and odd_cols must have the same number of rows")

    num_individuals = even_cols.shape[0]
    min_cols = min(even_cols.shape[1], odd_cols.shape[1])

    if min_cols == 0:
        return np.full(num_individuals, np.nan)

    correlations = np.zeros(num_individuals)

    for i in range(num_individuals):
        even_vals = even_cols[i, :min_cols]
        odd_vals = odd_cols[i, :min_cols]

        valid_mask = ~(np.isnan(even_vals) | np.isnan(odd_vals))
        if np.sum(valid_mask) < 2:
            correlations[i] = np.nan
        else:
            even_clean = even_vals[valid_mask]
            odd_clean = odd_vals[valid_mask]

            if len(even_clean) < 2:
                correlations[i] = np.nan
            else:
                try:
                    corr_matrix = np.corrcoef(even_clean, odd_clean)
                    correlations[i] = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0
                except Exception:
                    correlations[i] = np.nan

    return correlations


def evenodd(
    x: list[list[float]] | np.ndarray, factors: list[int], diag: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculate even-odd consistency scores for each individual based on the provided factors.

    This function splits each factor into even and odd columns, calculates correlations
    between corresponding pairs, and returns the average correlation for each individual.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their responses.
          Can be a 2D list or numpy array.
    - factors: List of integers specifying the length of each factor in the dataset.
               The sum of factors should equal the number of columns in x.
    - diag: Boolean to optionally return diagnostic values (number of valid correlations per individual).

    Returns:
    - A numpy array of even-odd consistency scores (average correlations per individual)
    - If diag=True, returns a tuple of (scores, diagnostic_values)

    Raises:
    - ValueError: If factors don't sum to the number of columns, or if data is empty

    Example:
        >>> data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
        >>> factors = [4, 2]  # First factor has 4 items, second has 2
        >>> scores = evenodd(data, factors)
        >>> print(scores)
        [0.5, 0.5]  # Average correlations for each individual
    """

    if not factors:
        raise ValueError("factors list cannot be empty")

    if x is None:
        raise ValueError("input data cannot be None")

    if isinstance(x, np.ndarray) and x.size == 0:
        raise ValueError("input data cannot be empty")

    if isinstance(x, list) and len(x) == 0:
        raise ValueError("input data cannot be empty")

    x_array = np.array(x, dtype=float)
    if x_array.ndim == 1:
        x_array = x_array.reshape(1, -1)
    if x_array.ndim != 2:
        raise ValueError("input data must be 2-dimensional")
    num_individuals = x_array.shape[0]

    expected_cols = sum(factors)
    if x_array.shape[1] != expected_cols:
        raise ValueError(
            f"sum of factors ({expected_cols}) must equal number of columns ({x_array.shape[1]})"
        )

    all_correlations = []
    diag_vals = np.zeros(num_individuals, dtype=int)

    start_col = 0
    for factor_size in factors:
        if factor_size < 2:
            start_col += factor_size
            continue

        end_col = start_col + factor_size

        even_cols = x_array[:, start_col:end_col:2]
        odd_cols = x_array[:, start_col + 1 : end_col : 2]

        corrs = calculate_correlations(even_cols, odd_cols)

        if len(corrs) > 0:
            all_correlations.append(corrs)
            diag_vals += (~np.isnan(corrs)).astype(int)

        start_col = end_col

    if all_correlations:
        stacked_corrs = np.column_stack(all_correlations)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_correlations = np.nanmean(stacked_corrs, axis=1)
        avg_correlations = np.nan_to_num(avg_correlations, nan=0.0)
    else:
        avg_correlations = np.full(num_individuals, np.nan)

    return (avg_correlations, diag_vals) if diag else avg_correlations
