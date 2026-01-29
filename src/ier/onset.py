"""
Carelessness onset detection via changepoint analysis.

Detects the item index at which a respondent's behavior shifts from attentive
to careless responding, using running intra-individual response variability (IRV)
and the Shao & Zhang self-normalized cumulative sum changepoint test.

References:
- Shao, X., & Zhang, X. (2010). Testing for change points in time series.
  Journal of the American Statistical Association, 105(491), 1228-1240.
- Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data.
  Psychological Methods, 17(3), 437-455.
"""

import numpy as np

from ier._validation import MatrixLike, validate_matrix_input


def onset(
    x: MatrixLike,
    window_size: int = 10,
    min_items: int = 20,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Detect the item index at which carelessness begins for each respondent.

    Computes running IRV over sliding windows, then applies a self-normalized
    cumulative sum changepoint test to identify the transition point from
    attentive to careless responding.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - window_size: Size of the sliding window for running IRV (default 10).
    - min_items: Minimum number of items required for onset detection (default 20).
    - na_rm: If True, handles missing values.

    Returns:
    - A numpy array of onset item indices per respondent. NaN if no changepoint
      is detected or if the respondent has fewer than min_items valid responses.

    Raises:
    - ValueError: If window_size < 2 or min_items < window_size.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> attentive = rng.choice([1, 2, 3, 4, 5], size=(1, 15))
        >>> careless = np.full((1, 15), 3)
        >>> data = np.hstack([attentive, careless])
        >>> onset(data, window_size=5, min_items=10)
    """
    x_array = validate_matrix_input(x, check_type=False)

    if window_size < 2:
        raise ValueError("window_size must be at least 2")

    if min_items < window_size:
        raise ValueError("min_items must be at least as large as window_size")

    if not na_rm and np.isnan(x_array).any():
        raise ValueError("data contains missing values. Set na_rm=True to handle them")

    n_rows = x_array.shape[0]
    result = np.full(n_rows, np.nan)

    for i in range(n_rows):
        row = x_array[i, :]
        if na_rm:
            row = row[~np.isnan(row)]

        if len(row) < min_items:
            continue

        running_irv = _running_inconsistency(row, window_size)

        if len(running_irv) < 3:
            continue

        cp = _shao_zhang_changepoint(running_irv)
        if cp is not None:
            result[i] = float(cp + window_size - 1)

    return result


def onset_flag(
    x: MatrixLike,
    window_size: int = 10,
    min_items: int = 20,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Flag respondents for whom a carelessness onset was detected.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - window_size: Size of the sliding window for running IRV.
    - min_items: Minimum number of items required for onset detection.
    - na_rm: If True, handles missing values.

    Returns:
    - Boolean array where True indicates a carelessness onset was detected.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> attentive = rng.choice([1, 2, 3, 4, 5], size=(1, 15))
        >>> careless = np.full((1, 15), 3)
        >>> data = np.hstack([attentive, careless])
        >>> onset_flag(data, window_size=5, min_items=10)
    """
    onset_indices = onset(x, window_size=window_size, min_items=min_items, na_rm=na_rm)
    result: np.ndarray = ~np.isnan(onset_indices)
    return result


def _running_inconsistency(row: np.ndarray, window_size: int) -> np.ndarray:
    """Compute running standard deviation over sliding windows."""
    n = len(row)
    if n < window_size:
        return np.array([])

    n_windows = n - window_size + 1
    result = np.zeros(n_windows)

    for j in range(n_windows):
        window = row[j : j + window_size]
        result[j] = np.std(window)

    return result


def _shao_zhang_changepoint(series: np.ndarray) -> int | None:
    """
    Apply the Shao & Zhang self-normalized cumulative sum test to detect a changepoint.

    Returns the index of the changepoint, or None if the test statistic does not
    exceed the critical value.
    """
    n = len(series)
    if n < 3:
        return None

    mean_val = np.mean(series)
    cumsum = np.cumsum(series - mean_val)

    var_estimate = np.zeros(n)
    for k in range(1, n):
        partial = series[:k] - np.mean(series[:k])
        var_estimate[k] = np.sum(partial**2)

    var_estimate = np.maximum(var_estimate, 1e-10)

    test_stats = np.zeros(n)
    for k in range(1, n - 1):
        test_stats[k] = cumsum[k] ** 2 / var_estimate[k]

    trim = max(1, n // 10)
    candidate_range = test_stats[trim : n - trim]

    if len(candidate_range) == 0:
        return None

    max_stat = np.max(candidate_range)
    critical_value = 1.358

    if max_stat > critical_value:
        return int(trim + np.argmax(candidate_range))

    return None
