"""
Takes a matrix of item responses and identifies item pairs that are highly correlated within the
overall dataset. What defines "highly correlated" is set by the critical value (e.g., r > .60). Each
respondents' psychometric synonym score is then computed as the within-person correlation be-
tween the identified item-pairs. Alternatively computes the psychometric antonym score which is a
variant that uses item pairs that are highly negatively correlated.

This module provides functions for detecting careless responding patterns by analyzing how
individuals respond to psychometrically similar (synonym) or opposite (antonym) items.
"""

import random
from typing import Any

import numpy as np


def get_highly_correlated_pairs(
    item_correlations: np.ndarray, critval: float, anto: bool
) -> np.ndarray:
    """
    Identify item pairs that meet the correlation threshold.

    Parameters:
    - item_correlations: Correlation matrix between items
    - critval: Critical value for correlation threshold
    - anto: If True, find negatively correlated pairs; if False, find positively correlated pairs

    Returns:
    - Array of item pair indices (i, j) that meet the threshold
    """
    if anto:
        return np.argwhere(np.tril(item_correlations, -1) <= critval)
    else:
        return np.argwhere(np.tril(item_correlations, -1) >= critval)


def compute_person_correlations(response_i: np.ndarray, response_j: np.ndarray) -> np.ndarray:
    """
    Compute within-person correlations between item pairs.

    Parameters:
    - response_i: Responses to first item in each pair
    - response_j: Responses to second item in each pair

    Returns:
    - Array of within-person correlations for each item pair
    """
    if response_i.shape[0] == 0 or response_j.shape[0] == 0:
        return np.array([])

    mean_i = response_i.mean(axis=1, keepdims=True)
    mean_j = response_j.mean(axis=1, keepdims=True)
    std_i = response_i.std(axis=1, keepdims=True)
    std_j = response_j.std(axis=1, keepdims=True)

    std_i[std_i == 0] = 1
    std_j[std_j == 0] = 1

    numerator = (response_i - mean_i) * (response_j - mean_j)
    denominator = std_i * std_j

    result: np.ndarray = numerator / denominator
    return result


def psychsyn(
    x: list[list[float]] | np.ndarray,
    critval: float = 0.60,
    anto: bool = False,
    diag: bool = False,
    resample_na: bool = False,
    random_seed: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculate psychometric synonym (or antonym) scores based on the provided item response matrix.

    Psychometric synonyms are item pairs that are highly correlated across the sample.
    This function identifies such pairs and computes within-person correlations between them.
    High scores indicate consistent responding to psychometrically similar items.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their item responses.
          Can be a 2D list or numpy array.
    - critval: Minimum magnitude of correlation for items to be considered synonyms/antonyms.
               Default is 0.60 for synonyms, typically -0.60 for antonyms.
    - anto: Boolean indicating whether to compute antonym scores (highly negatively correlated items).
    - diag: Boolean to optionally return the number of item pairs available for each observation.
    - resample_na: Boolean to indicate resampling when encountering NA for a respondent.
    - random_seed: Optional seed for random number generation when resample_na=True.

    Returns:
    - A numpy array of psychometric synonym/antonym scores, or
    - A tuple of (scores, diagnostic_values) if diag=True.

    Raises:
    - ValueError: If inputs are invalid (empty data, invalid critval, etc.)
    - TypeError: If input is not a list or numpy array

    Example:
        >>> data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [1, 1, 1, 4, 5, 6]]
        >>> scores = psychsyn(data, critval=0.5)
        >>> print(scores)
        [0.87, 0.92, 0.45]  # Third person has lower consistency

        >>> # With diagnostic output
        >>> scores, diag = psychsyn(data, critval=0.5, diag=True)
        >>> print(f"Scores: {scores}, Pairs per person: {diag}")
    """

    if x is None:
        raise ValueError("input data cannot be None")

    if not isinstance(x, (list, np.ndarray)):
        raise TypeError("input data must be a list or numpy array")

    x_array = np.asarray(x)

    if x_array.ndim != 2:
        raise ValueError("input data must be 2-dimensional")

    if x_array.shape[0] == 0 or x_array.shape[1] == 0:
        raise ValueError("input data cannot be empty")

    if x_array.shape[1] < 2:
        raise ValueError("data must have at least 2 columns (items)")

    if not isinstance(critval, (int, float)):
        raise ValueError("critval must be a number")

    if anto and critval > 0:
        raise ValueError("critval should be negative for antonym analysis")

    if not anto and critval < 0:
        raise ValueError("critval should be positive for synonym analysis")

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    item_correlations = np.corrcoef(x_array, rowvar=False)

    item_correlations[np.isnan(item_correlations)] = 0

    item_pairs = get_highly_correlated_pairs(item_correlations, critval, anto)

    if len(item_pairs) == 0:
        if diag:
            return np.full(x_array.shape[0], np.nan), np.zeros(x_array.shape[0], dtype=int)
        else:
            return np.full(x_array.shape[0], np.nan)

    response_i = x_array[:, item_pairs[:, 0]]
    response_j = x_array[:, item_pairs[:, 1]]

    person_corrs = compute_person_correlations(response_i, response_j)

    invalid_pairs = np.isnan(response_i) | np.isnan(response_j)
    person_corrs[invalid_pairs] = np.nan

    if resample_na:
        person_corrs = _resample_missing_correlations(person_corrs)

    scores = np.nanmean(person_corrs, axis=1)

    if np.any(np.isnan(scores)) and len(item_pairs) > 0:
        scores = np.nan_to_num(scores, nan=0.0)

    if diag:
        diag_values: np.ndarray = np.sum(~np.isnan(person_corrs), axis=1)
        return scores, diag_values
    else:
        result: np.ndarray = scores
        return result


def _resample_missing_correlations(person_corrs: np.ndarray) -> np.ndarray:
    """
    Resample missing correlations based on available data.

    Parameters:
    - person_corrs: Array of person correlations with potential NaN values

    Returns:
    - Array with resampled values replacing NaN
    """
    result = person_corrs.copy()

    for person_index in range(result.shape[0]):
        person_row = result[person_index]

        if np.isnan(person_row).all():
            overall_mean = np.abs(np.nanmean(result))
            result[person_index] = np.random.choice([-1, 1], size=len(person_row)) * overall_mean
        else:
            valid_corrs = person_row[~np.isnan(person_row)]
            if valid_corrs.size > 0:
                abs_mean_corr = np.abs(np.mean(valid_corrs))
                missing_mask = np.isnan(person_row)
                result[person_index][missing_mask] = (
                    np.random.choice([-1, 1], size=missing_mask.sum()) * abs_mean_corr
                )

    return result


def psychsyn_critval(
    x: list[list[float]] | np.ndarray, anto: bool = False, min_correlation: float = 0.0
) -> list[tuple[int, int, float]]:
    """
    Calculate and order pairwise correlations for all items in the provided item response matrix.

    This function helps identify appropriate critical values for psychsyn analysis by showing
    the distribution of item correlations.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their item responses.
    - anto: Boolean indicating whether to order correlations by largest negative values.
    - min_correlation: Minimum correlation magnitude to include in results.

    Returns:
    - A list of tuples containing (item_i, item_j, correlation), ordered by magnitude.

    Example:
        >>> data = [[1, 2, 3, 4], [2, 3, 4, 5], [1, 1, 1, 4]]
        >>> pairs = psychsyn_critval(data, min_correlation=0.3)
        >>> print(pairs[:3])  # Top 3 correlations
        [(0, 1, 0.87), (1, 2, 0.82), (0, 2, 0.65)]
    """

    if x is None:
        raise ValueError("input data cannot be None")

    if not isinstance(x, (list, np.ndarray)):
        raise TypeError("input data must be a list or numpy array")

    x_array = np.asarray(x)

    if x_array.ndim != 2:
        raise ValueError("input data must be 2-dimensional")

    if x_array.shape[0] == 0 or x_array.shape[1] == 0:
        raise ValueError("input data cannot be empty")

    if x_array.shape[1] < 2:
        raise ValueError("data must have at least 2 columns (items)")

    item_correlations = np.corrcoef(x_array, rowvar=False)

    correlation_list = []
    for i in range(item_correlations.shape[0]):
        for j in range(i + 1, item_correlations.shape[1]):
            corr = item_correlations[i, j]
            if not np.isnan(corr) and abs(corr) >= min_correlation:
                correlation_list.append((i, j, corr))

    if anto:
        correlation_list.sort(key=lambda x: x[2])
    else:
        correlation_list.sort(key=lambda x: -x[2])

    return correlation_list


def psychant(
    x: list[list[float]] | np.ndarray,
    critval: float = -0.60,
    diag: bool = False,
    resample_na: bool = False,
    random_seed: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculate the psychometric antonym score.

    Psychometric antonyms are item pairs that are highly negatively correlated across the sample.
    This function is a convenience wrapper around psychsyn with antonym settings.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their item responses.
    - critval: Minimum magnitude of negative correlation for items to be considered antonyms.
               Default is -0.60.
    - diag: Boolean to optionally return the number of item pairs available for each observation.
    - resample_na: Boolean to indicate resampling when encountering NA for a respondent.
    - random_seed: Optional seed for random number generation when resample_na=True.

    Returns:
    - A numpy array of psychometric antonym scores, or
    - A tuple of (scores, diagnostic_values) if diag=True.

    Example:
        >>> data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [1, 1, 1, 4, 5, 6]]
        >>> scores = psychant(data, critval=-0.5)
        >>> print(scores)
        [0.23, 0.18, 0.45]  # Higher scores indicate inconsistent responding
    """
    return psychsyn(
        x, critval=critval, anto=True, diag=diag, resample_na=resample_na, random_seed=random_seed
    )


def psychsyn_summary(
    x: list[list[float]] | np.ndarray, critval: float = 0.60, anto: bool = False
) -> dict[str, Any]:
    """
    Calculate summary statistics for psychometric synonym/antonym analysis.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their item responses.
    - critval: Critical value for correlation threshold.
    - anto: If True, analyze antonyms; if False, analyze synonyms.

    Returns:
    - Dictionary with summary statistics and item pair information.

    Example:
        >>> data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [1, 1, 1, 4, 5, 6]]
        >>> summary = psychsyn_summary(data, critval=0.5)
        >>> print(summary)
        {'mean_score': 0.75, 'std_score': 0.24, 'item_pairs': 3, ...}
    """

    scores, diag = psychsyn(x, critval=critval, anto=anto, diag=True)

    x_array = np.asarray(x)
    item_correlations = np.corrcoef(x_array, rowvar=False)
    item_correlations[np.isnan(item_correlations)] = 0
    item_pairs = get_highly_correlated_pairs(item_correlations, critval, anto)

    valid_scores = scores[~np.isnan(scores)]

    if len(valid_scores) == 0:
        return {
            "mean_score": np.nan,
            "std_score": np.nan,
            "min_score": np.nan,
            "max_score": np.nan,
            "median_score": np.nan,
            "item_pairs": len(item_pairs),
            "total_individuals": len(scores),
            "valid_individuals": 0,
            "missing_individuals": len(scores),
        }

    return {
        "mean_score": float(np.mean(valid_scores)),
        "std_score": float(np.std(valid_scores)),
        "min_score": float(np.min(valid_scores)),
        "max_score": float(np.max(valid_scores)),
        "median_score": float(np.median(valid_scores)),
        "item_pairs": len(item_pairs),
        "total_individuals": len(scores),
        "valid_individuals": len(valid_scores),
        "missing_individuals": int(np.sum(np.isnan(scores))),
    }
