"""
Infrequency / bogus item scoring for detecting insufficient effort responding.

Counts the number of failed attention-check (bogus/infrequency) items per respondent.
These are items with known correct answers that attentive respondents should get right
(e.g., "Please select 'Strongly Agree' for this item").

References:
- Huang, J. L., Curran, P. G., Keeney, J., Poposki, E. M., & DeShon, R. P. (2012).
  Detecting and deterring insufficient effort responding to surveys.
  Journal of Business and Psychology, 27(1), 99-114.
- Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data.
  Psychological Methods, 17(3), 437-455.
"""

import numpy as np

from ier._validation import MatrixLike, validate_matrix_input


def infrequency(
    x: MatrixLike,
    item_indices: list[int],
    expected_responses: list[float],
    proportion: bool = False,
) -> np.ndarray:
    """
    Count failed attention-check items per respondent.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - item_indices: Column indices (0-based) of the attention-check items.
    - expected_responses: Expected correct response for each attention-check item.
    - proportion: If True, return proportion of failed items instead of count.

    Returns:
    - A numpy array of failure counts (or proportions) per respondent.

    Raises:
    - ValueError: If item_indices and expected_responses have different lengths,
                  if item_indices is empty, or if indices are out of bounds.

    Example:
        >>> data = [[5, 3, 1], [5, 5, 5], [1, 3, 5]]
        >>> scores = infrequency(data, item_indices=[0, 2], expected_responses=[5, 1])
        >>> print(scores)
        [0. 1. 2.]
    """
    x_array = validate_matrix_input(x, check_type=False)
    n_cols = x_array.shape[1]

    if len(item_indices) == 0:
        raise ValueError("item_indices cannot be empty")

    if len(item_indices) != len(expected_responses):
        raise ValueError(
            f"item_indices ({len(item_indices)}) and expected_responses "
            f"({len(expected_responses)}) must have the same length"
        )

    for idx in item_indices:
        if idx < 0 or idx >= n_cols:
            raise ValueError(f"item index {idx} out of bounds for data with {n_cols} columns")

    failures = np.zeros(x_array.shape[0], dtype=float)

    for idx, expected in zip(item_indices, expected_responses, strict=True):
        col = x_array[:, idx]
        nan_mask = np.isnan(col)
        mismatch = col != expected
        failures += np.where(nan_mask, 0.0, mismatch.astype(float))

    if proportion:
        failures = failures / len(item_indices)

    return failures


def infrequency_flag(
    x: MatrixLike,
    item_indices: list[int],
    expected_responses: list[float],
    threshold: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Count failed attention-check items and flag respondents exceeding a threshold.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - item_indices: Column indices (0-based) of the attention-check items.
    - expected_responses: Expected correct response for each attention-check item.
    - threshold: Number of failed items at or above which to flag (default 1).

    Returns:
    - Tuple of (failure_counts, flags) where flags is True for flagged respondents.

    Example:
        >>> data = [[5, 3, 1], [5, 5, 5], [1, 3, 5]]
        >>> scores, flags = infrequency_flag(data, [0, 2], [5, 1], threshold=2)
        >>> print(flags)
        [False False  True]
    """
    scores = infrequency(x, item_indices, expected_responses, proportion=False)
    flags = scores >= threshold
    return scores, flags
