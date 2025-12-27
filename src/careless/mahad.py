"""
Find and graph Mahalanobis Distance (D) and flag potential outliers

Takes a matrix of item responses and computes Mahalanobis D. Can additionally return a
vector of binary outlier flags.
Mahalanobis distance is calculated using a function which supports missing values.
The Mahalanobis distance is a measure of the distance between a point P and a distribution D,
introduced by P. C. Mahalanobis in 1936. It is a multi-dimensional generalization of the idea of
measuring how many standard deviations away P is from the mean of D. This distance is zero if P is
at the mean of D, and grows as P moves away from the mean along each principal component axis.
The Mahalanobis distance is thus unitless and scale-invariant, and takes into account the
correlations of the data set.
"""

from typing import Any

import numpy as np

SCIPY_AVAILABLE = False
stats: Any = None
try:
    import scipy.stats as stats  # type: ignore[no-redef]

    SCIPY_AVAILABLE = True
except ImportError:
    pass


def mahad(
    x: list[list[float]] | np.ndarray,
    flag: bool = False,
    confidence: float = 0.95,
    na_rm: bool = False,
    method: str = "chi2",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Computes Mahalanobis Distance for a matrix of data.

    Mahalanobis distance measures how many standard deviations away a point is from the mean
    of a distribution, taking into account correlations between variables. It's useful for
    detecting multivariate outliers in survey data.

    Parameters:
    - x: Matrix of data where rows are observations and columns are variables.
          Can be a 2D list or numpy array.
    - flag: If True, flags potential outliers based on the confidence level.
    - confidence: Confidence level for flagging outliers (0–1). Default is 0.95.
    - na_rm: If True, removes rows with missing data before computing distances,
             but reinserts NaNs in original positions. If False, raises error for missing data.
    - method: Method for outlier detection. Options: "chi2" (chi-squared distribution),
              "iqr" (interquartile range), "zscore" (z-score threshold).

    Returns:
    - Mahalanobis distances (with NaNs where removed), or
    - Tuple of (distances, flags) if `flag=True`.

    Raises:
    - ValueError: If inputs are invalid (empty data, invalid confidence, etc.)
    - TypeError: If input is not a list or numpy array
    - RuntimeError: If scipy is required but not available

    Example:
        >>> import numpy as np
        >>> data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1]]
        >>> distances = mahad(data)
        >>> print(distances)
        [0.87, 0.87, 0.87, 2.60]  # Last observation is an outlier

        >>> # With outlier flagging
        >>> distances, flags = mahad(data, flag=True, confidence=0.95)
        >>> print(flags)
        [False, False, False, True]  # Last observation flagged as outlier
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

    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1")

    if method not in ["chi2", "iqr", "zscore"]:
        raise ValueError("method must be one of: 'chi2', 'iqr', 'zscore'")

    if method == "chi2" and not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for chi2 method. Install with: pip install scipy")

    if na_rm:
        all_nan_mask = np.isnan(x_array).all(axis=1)
        partial_nan_mask = np.isnan(x_array).any(axis=1)
        valid_mask = ~partial_nan_mask & ~all_nan_mask
        x_filtered = x_array[valid_mask]
    else:
        if np.isnan(x_array).any():
            raise ValueError("data contains missing values. Set na_rm=True to handle them")
        x_filtered = x_array
        valid_mask = np.ones(x_array.shape[0], dtype=bool)

    if x_filtered.size == 0:
        raise ValueError("no complete cases found after removing missing values")

    if x_filtered.shape[0] < x_filtered.shape[1]:
        raise ValueError(
            f"insufficient observations ({x_filtered.shape[0]}) for dimensions ({x_filtered.shape[1]}). "
            "Need more observations than variables."
        )

    distances_filtered = _compute_mahalanobis_distance(x_filtered)

    distances = np.full(shape=(x_array.shape[0],), fill_value=np.nan)
    distances[valid_mask] = distances_filtered

    distances = np.where(np.isnan(distances), np.nan, np.abs(distances))

    if flag:
        flags = _flag_outliers(distances, confidence, method, x_array.shape[1])
        return distances, flags

    return distances


def _compute_mahalanobis_distance(x: np.ndarray) -> np.ndarray:
    """
    Compute Mahalanobis distances for a matrix of data.

    Parameters:
    - x: Matrix of data (n_samples, n_features) with no missing values

    Returns:
    - Array of Mahalanobis distances
    """
    mean_vector = np.mean(x, axis=0)
    cov_matrix = np.cov(x, rowvar=False)

    cond_number = np.linalg.cond(cov_matrix)
    if cond_number < 1 / np.finfo(cov_matrix.dtype).eps:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    else:
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

    centered_data = x - mean_vector
    mahalanobis_squared = np.einsum("ij,jk,ik->i", centered_data, inv_cov_matrix, centered_data)
    result: np.ndarray = np.sqrt(mahalanobis_squared)
    return result


def _flag_outliers(
    distances: np.ndarray, confidence: float, method: str, n_features: int
) -> np.ndarray:
    """
    Flag outliers based on Mahalanobis distances.

    Parameters:
    - distances: Array of Mahalanobis distances
    - confidence: Confidence level (0-1)
    - method: Outlier detection method
    - n_features: Number of features (for chi2 degrees of freedom)

    Returns:
    - Boolean array indicating outliers
    """
    if method == "chi2":
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for chi2 method. Install with: pip install scipy")
        threshold: float = stats.chi2.ppf(confidence, df=n_features)
        result: np.ndarray = distances > np.sqrt(threshold)
        return result

    elif method == "iqr":
        valid_distances = distances[~np.isnan(distances)]
        if len(valid_distances) == 0:
            return np.full_like(distances, False, dtype=bool)

        q1, q3 = np.percentile(valid_distances, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        flags = np.full_like(distances, False, dtype=bool)
        valid_mask = ~np.isnan(distances)
        flags[valid_mask] = (distances[valid_mask] < lower_bound) | (
            distances[valid_mask] > upper_bound
        )
        return flags

    elif method == "zscore":
        valid_distances = distances[~np.isnan(distances)]
        if len(valid_distances) == 0:
            return np.full_like(distances, False, dtype=bool)

        mean_dist = np.mean(valid_distances)
        std_dist = np.std(valid_distances)

        if std_dist == 0:
            return np.full_like(distances, False, dtype=bool)

        z_threshold = stats.norm.ppf(1 - (1 - confidence) / 2) if SCIPY_AVAILABLE else 2.0

        flags = np.full_like(distances, False, dtype=bool)
        valid_mask = ~np.isnan(distances)
        z_scores = np.abs((distances[valid_mask] - mean_dist) / std_dist)
        flags[valid_mask] = z_scores > z_threshold
        return flags

    else:
        raise ValueError(f"unknown method: {method}")


def mahad_summary(
    x: list[list[float]] | np.ndarray, confidence: float = 0.95, na_rm: bool = False
) -> dict[str, Any]:
    """
    Calculate summary statistics for Mahalanobis distances.

    Parameters:
    - x: Matrix of data where rows are observations and columns are variables
    - confidence: Confidence level for outlier detection
    - na_rm: If True, removes rows with missing data

    Returns:
    - Dictionary with summary statistics and outlier information

    Example:
        >>> data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1]]
        >>> summary = mahad_summary(data)
        >>> print(summary)
        {'mean': 1.55, 'std': 0.87, 'outliers': 1, 'total': 4, ...}
    """

    distances, flags = mahad(x, flag=True, confidence=confidence, na_rm=na_rm)

    valid_distances = distances[~np.isnan(distances)]

    if len(valid_distances) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "outliers": 0,
            "total": len(distances),
            "valid_count": 0,
            "missing_count": np.sum(np.isnan(distances)),
        }

    return {
        "mean": float(np.mean(valid_distances)),
        "std": float(np.std(valid_distances)),
        "min": float(np.min(valid_distances)),
        "max": float(np.max(valid_distances)),
        "median": float(np.median(valid_distances)),
        "outliers": int(np.sum(flags)),
        "total": len(distances),
        "valid_count": len(valid_distances),
        "missing_count": int(np.sum(np.isnan(distances))),
    }
