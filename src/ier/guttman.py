"""
Guttman errors for person-fit analysis in detecting careless responding.

Guttman errors count the number of response reversals relative to item
difficulty ordering. High error counts suggest inconsistent or careless responding.
"""

import numpy as np

from ier._validation import MatrixLike, validate_matrix_input


def guttman(
    x: MatrixLike,
    na_rm: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Calculate Guttman errors for each individual.

    Guttman errors measure the number of times a person's responses violate
    the expected ordering based on item difficulty (mean endorsement).
    An error occurs when a person scores higher on a harder item than an
    easier item.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - na_rm: If True, handle missing values by excluding them from comparisons.
    - normalize: If True, return proportion of errors (0-1 scale).
                 If False, return raw error counts.

    Returns:
    - A numpy array of Guttman error scores for each individual.
      Higher values indicate more inconsistent responding.

    Raises:
    - ValueError: If inputs are invalid

    Example:
        >>> data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [3, 3, 3, 3, 3]]
        >>> scores = guttman(data)
        >>> print(scores)  # Second person has high errors (reversed pattern)
    """
    x_array = validate_matrix_input(x, min_columns=2)
    n_persons = x_array.shape[0]
    n_items = x_array.shape[1]

    item_difficulty = np.nanmean(x_array, axis=0) if na_rm else np.mean(x_array, axis=0)

    difficulty_order = np.argsort(item_difficulty)
    x_sorted = x_array[:, difficulty_order]

    i_indices, j_indices = np.triu_indices(n_items, k=1)
    item_easy = x_sorted[:, i_indices]
    item_hard = x_sorted[:, j_indices]

    if na_rm:
        valid = ~np.isnan(item_easy) & ~np.isnan(item_hard)
        comparisons = np.sum(valid, axis=1).astype(float)
        error_mask = valid & (item_easy < item_hard)
    else:
        comparisons = np.full(n_persons, len(i_indices), dtype=float)
        error_mask = item_easy < item_hard

    errors = np.sum(error_mask, axis=1).astype(float)

    result: np.ndarray
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            result = errors / comparisons
        result = np.where(comparisons == 0, np.nan, result)
    else:
        result = errors

    return result


def guttman_flag(
    x: MatrixLike,
    threshold: float = 0.5,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Flag individuals with high Guttman error rates.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - threshold: Error rate threshold for flagging (default 0.5).
    - na_rm: If True, handle missing values.

    Returns:
    - Boolean array where True indicates potentially careless responding.

    Example:
        >>> data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [3, 3, 3, 3, 3]]
        >>> flags = guttman_flag(data, threshold=0.4)
    """
    scores = guttman(x, na_rm=na_rm, normalize=True)
    return scores > threshold
