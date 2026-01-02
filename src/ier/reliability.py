"""
Resampled individual reliability for detecting careless responding.

This method estimates the reliability/consistency of each individual's
responses using split-half or bootstrap approaches.
"""

import warnings

import numpy as np

from ier._validation import MatrixLike, validate_matrix_input


def individual_reliability(
    x: MatrixLike,
    n_splits: int = 100,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Calculate resampled individual reliability for each person.

    Estimates how consistent each individual's responses are by repeatedly
    splitting items into halves and correlating the split scores.
    Low reliability suggests inconsistent (potentially careless) responding.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - n_splits: Number of random split-half iterations (default 100).
    - random_seed: Optional seed for reproducibility.

    Returns:
    - A numpy array of reliability estimates for each individual.
      Values range from -1 to 1, with higher values indicating more
      consistent responding.

    Raises:
    - ValueError: If inputs are invalid or too few items

    Example:
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 5, 2, 4, 1, 5], [3, 3, 3, 3, 3, 3]]
        >>> rel = individual_reliability(data, n_splits=50)
        >>> print(rel)  # First person: high, second: variable, third: undefined
    """
    x_array = validate_matrix_input(x, min_columns=4)
    n_persons = x_array.shape[0]
    n_items = x_array.shape[1]

    if random_seed is not None:
        np.random.seed(random_seed)

    correlations = np.zeros((n_persons, n_splits))
    half = n_items // 2

    for split_idx in range(n_splits):
        indices = np.random.permutation(n_items)
        first_half = indices[:half]
        second_half = indices[half : 2 * half]

        half1 = x_array[:, first_half]
        half2 = x_array[:, second_half]

        valid = ~np.isnan(half1) & ~np.isnan(half2)
        valid_counts = valid.sum(axis=1)

        half1_masked = np.where(valid, half1, np.nan)
        half2_masked = np.where(valid, half2, np.nan)

        with np.errstate(invalid="ignore"):
            mean1 = np.nanmean(half1_masked, axis=1, keepdims=True)
            mean2 = np.nanmean(half2_masked, axis=1, keepdims=True)

            centered1 = np.where(valid, half1 - mean1, 0.0)
            centered2 = np.where(valid, half2 - mean2, 0.0)

            cov = (centered1 * centered2).sum(axis=1)
            std1 = np.sqrt((centered1**2).sum(axis=1))
            std2 = np.sqrt((centered2**2).sum(axis=1))

            split_corr = cov / (std1 * std2)

        split_corr = np.where(valid_counts < 2, np.nan, split_corr)
        split_corr = np.where((std1 == 0) | (std2 == 0), np.nan, split_corr)
        correlations[:, split_idx] = split_corr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        reliability = np.nanmean(correlations, axis=1)

        with np.errstate(invalid="ignore"):
            result: np.ndarray = (2 * reliability) / (1 + reliability)

    return result


def individual_reliability_flag(
    x: MatrixLike,
    threshold: float = 0.3,
    n_splits: int = 100,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Flag individuals with low reliability scores.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - threshold: Reliability threshold below which to flag (default 0.3).
    - n_splits: Number of split-half iterations.
    - random_seed: Optional seed for reproducibility.

    Returns:
    - Boolean array where True indicates potentially careless responding.

    Example:
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 5, 2, 4, 1, 5], [3, 3, 3, 3, 3, 3]]
        >>> flags = individual_reliability_flag(data, threshold=0.5)
    """
    rel = individual_reliability(x, n_splits=n_splits, random_seed=random_seed)
    result: np.ndarray = (rel < threshold) | np.isnan(rel)
    return result
