"""
U3 polytomous index for detecting unusual response patterns.

The U3 polytomous index measures the proportion of extreme responses,
which can indicate careless responding patterns like extreme response style
or random clicking at scale endpoints.
"""

import numpy as np

from ier._validation import MatrixLike, validate_matrix_input


def u3_poly(
    x: MatrixLike,
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> np.ndarray:
    """
    Calculate U3 polytomous index for each individual.

    The U3 index measures the proportion of responses at the extreme ends
    of the response scale. High values may indicate extreme response style
    or careless responding.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - scale_min: Minimum value of the response scale. If None, inferred from data.
    - scale_max: Maximum value of the response scale. If None, inferred from data.

    Returns:
    - A numpy array of U3 values (proportion of extreme responses) for each individual.
      Values range from 0 to 1.

    Raises:
    - ValueError: If inputs are invalid

    Example:
        >>> data = [[1, 1, 5, 5, 3], [3, 3, 3, 3, 3], [1, 5, 1, 5, 1]]
        >>> u3 = u3_poly(data, scale_min=1, scale_max=5)
        >>> print(u3)  # Third person has highest extreme responding
    """
    x_array = validate_matrix_input(x, min_columns=1)

    if scale_min is None:
        scale_min = np.nanmin(x_array)
    if scale_max is None:
        scale_max = np.nanmax(x_array)

    extreme_low = x_array == scale_min
    extreme_high = x_array == scale_max
    extreme = extreme_low | extreme_high

    valid = ~np.isnan(x_array)
    valid_counts = np.sum(valid, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        u3_scores = np.sum(extreme & valid, axis=1) / valid_counts

    u3_scores = np.where(valid_counts == 0, np.nan, u3_scores)

    return u3_scores


def midpoint_responding(
    x: MatrixLike,
    scale_min: float | None = None,
    scale_max: float | None = None,
    tolerance: float = 0.0,
) -> np.ndarray:
    """
    Calculate proportion of midpoint responses for each individual.

    Excessive midpoint responding may indicate satisficing or
    inattentive responding.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - scale_min: Minimum value of the response scale.
    - scale_max: Maximum value of the response scale.
    - tolerance: Range around midpoint to count as midpoint response.

    Returns:
    - A numpy array of midpoint response proportions.

    Example:
        >>> data = [[1, 2, 5, 4, 3], [3, 3, 3, 3, 3], [1, 5, 1, 5, 1]]
        >>> mid = midpoint_responding(data, scale_min=1, scale_max=5)
        >>> print(mid)  # Second person has all midpoint responses
    """
    x_array = validate_matrix_input(x, min_columns=1)

    if scale_min is None:
        scale_min = np.nanmin(x_array)
    if scale_max is None:
        scale_max = np.nanmax(x_array)

    midpoint = (scale_min + scale_max) / 2

    is_midpoint = np.abs(x_array - midpoint) <= tolerance
    valid = ~np.isnan(x_array)
    valid_counts = np.sum(valid, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mid_scores = np.sum(is_midpoint & valid, axis=1) / valid_counts

    mid_scores = np.where(valid_counts == 0, np.nan, mid_scores)

    return mid_scores


def response_pattern(
    x: MatrixLike,
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Calculate multiple response pattern indices.

    Returns a dictionary with various response style indicators.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - scale_min: Minimum value of the response scale.
    - scale_max: Maximum value of the response scale.

    Returns:
    - Dictionary with:
        - "extreme": proportion of extreme responses (U3)
        - "midpoint": proportion of midpoint responses
        - "acquiescence": mean response (higher = more agreement bias)
        - "variability": response variability (SD)

    Example:
        >>> data = [[1, 2, 5, 4, 3], [3, 3, 3, 3, 3], [5, 5, 5, 5, 5]]
        >>> patterns = response_pattern(data, scale_min=1, scale_max=5)
    """
    x_array = validate_matrix_input(x, min_columns=1)

    if scale_min is None:
        scale_min = np.nanmin(x_array)
    if scale_max is None:
        scale_max = np.nanmax(x_array)

    extreme_low = x_array == scale_min
    extreme_high = x_array == scale_max
    extreme = extreme_low | extreme_high
    midpoint = (scale_min + scale_max) / 2
    is_midpoint = x_array == midpoint

    valid = ~np.isnan(x_array)
    valid_counts = np.sum(valid, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        extreme_scores = np.sum(extreme & valid, axis=1) / valid_counts
        mid_scores = np.sum(is_midpoint & valid, axis=1) / valid_counts

    extreme_scores = np.where(valid_counts == 0, np.nan, extreme_scores)
    mid_scores = np.where(valid_counts == 0, np.nan, mid_scores)

    return {
        "extreme": extreme_scores,
        "midpoint": mid_scores,
        "acquiescence": np.nanmean(x_array, axis=1),
        "variability": np.nanstd(x_array, axis=1),
    }
