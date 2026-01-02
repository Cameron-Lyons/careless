"""
Response time indices for detecting careless responding.

Extremely fast or unusually consistent response times may indicate
careless or inattentive responding.
"""

import numpy as np

from ier._validation import MatrixLike, validate_matrix_input


def response_time(
    times: MatrixLike,
    metric: str = "median",
) -> np.ndarray:
    """
    Calculate response time summary statistics for each individual.

    Very low response times may indicate careless responding where
    participants rush through items without reading them.

    Parameters:
    - times: A matrix of response times where rows are individuals and
             columns are items. Times should be in consistent units (e.g., seconds).
    - metric: Summary statistic to compute. Options:
              "mean" - average response time per item
              "median" - median response time per item
              "sd" - standard deviation of response times
              "min" - minimum response time

    Returns:
    - A numpy array of response time statistics for each individual.

    Raises:
    - ValueError: If inputs are invalid or metric is unknown

    Example:
        >>> times = [[2.1, 3.4, 2.8], [0.5, 0.4, 0.6], [2.5, 2.3, 2.7]]
        >>> avg_times = response_time(times, metric="mean")
        >>> print(avg_times)  # Second person has suspiciously fast times
    """
    times_array = validate_matrix_input(times, min_columns=1)

    result: np.ndarray
    if metric == "mean":
        result = np.nanmean(times_array, axis=1)
    elif metric == "median":
        result = np.nanmedian(times_array, axis=1)
    elif metric == "sd":
        result = np.nanstd(times_array, axis=1)
    elif metric == "min":
        result = np.nanmin(times_array, axis=1)
    else:
        raise ValueError(f"unknown metric: {metric}. Use 'mean', 'median', 'sd', or 'min'")
    return result


def response_time_flag(
    times: MatrixLike,
    threshold: float | None = None,
    method: str = "median",
    cutoff_percentile: float = 5.0,
) -> np.ndarray:
    """
    Flag individuals with suspiciously fast response times.

    Parameters:
    - times: A matrix of response times.
    - threshold: Absolute threshold for flagging (in same units as times).
                 If None, uses cutoff_percentile to determine threshold.
    - method: Method for computing per-person response time ("mean" or "median").
    - cutoff_percentile: Percentile below which to flag (default 5th percentile).
                         Only used if threshold is None.

    Returns:
    - Boolean array where True indicates potentially careless responding.

    Example:
        >>> times = [[2.1, 3.4, 2.8], [0.5, 0.4, 0.6], [2.5, 2.3, 2.7]]
        >>> flags = response_time_flag(times, threshold=1.0)
    """
    person_times = response_time(times, metric=method)

    if threshold is None:
        valid_times = person_times[~np.isnan(person_times)]
        if len(valid_times) == 0:
            return np.full(len(person_times), False, dtype=bool)
        threshold = np.percentile(valid_times, cutoff_percentile)

    return person_times < threshold


def response_time_consistency(
    times: MatrixLike,
) -> np.ndarray:
    """
    Calculate response time consistency (coefficient of variation).

    Very low consistency (uniform times) may indicate "clicking through"
    behavior where the person isn't reading items.

    Parameters:
    - times: A matrix of response times.

    Returns:
    - A numpy array of coefficient of variation values for each individual.
      Lower values indicate more uniform (potentially suspicious) timing.

    Example:
        >>> times = [[2.1, 3.4, 2.8], [1.0, 1.0, 1.0], [2.5, 2.3, 2.7]]
        >>> cv = response_time_consistency(times)
        >>> print(cv)  # Second person has very consistent (suspicious) times
    """
    times_array = validate_matrix_input(times, min_columns=2)

    means = np.nanmean(times_array, axis=1)
    stds = np.nanstd(times_array, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        cv: np.ndarray = stds / means

    return cv
