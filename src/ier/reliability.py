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

    if n_items < 4:
        raise ValueError("need at least 4 items for split-half reliability")

    if random_seed is not None:
        np.random.seed(random_seed)

    correlations = np.zeros((n_persons, n_splits))

    for split_idx in range(n_splits):
        indices = np.random.permutation(n_items)
        half = n_items // 2
        first_half = indices[:half]
        second_half = indices[half : 2 * half]

        scores1 = np.nanmean(x_array[:, first_half], axis=1)
        scores2 = np.nanmean(x_array[:, second_half], axis=1)

        for i in range(n_persons):
            s1, s2 = scores1[i], scores2[i]
            if np.isnan(s1) or np.isnan(s2) or (np.std([s1]) == 0 and np.std([s2]) == 0):
                correlations[i, split_idx] = np.nan
            else:
                all_scores1 = x_array[i, first_half]
                all_scores2 = x_array[i, second_half]
                valid = ~np.isnan(all_scores1) & ~np.isnan(all_scores2)
                if valid.sum() < 2:
                    correlations[i, split_idx] = np.nan
                else:
                    std1 = np.std(all_scores1[valid])
                    std2 = np.std(all_scores2[valid])
                    if std1 == 0 or std2 == 0:
                        correlations[i, split_idx] = np.nan
                    else:
                        correlations[i, split_idx] = np.corrcoef(
                            all_scores1[valid], all_scores2[valid]
                        )[0, 1]

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
