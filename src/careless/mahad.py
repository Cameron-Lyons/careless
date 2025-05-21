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

import numpy as np
from scipy import stats
from typing import Union, Tuple

def mahad(
    x: np.ndarray,
    flag: bool = False,
    confidence: float = 0.95,
    na_rm: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Computes Mahalanobis Distance for a matrix of data.

    Parameters:
    - x: Matrix of data (n_samples, n_features).
    - flag: If True, flags potential outliers based on the confidence level.
    - confidence: Confidence level for flagging outliers (0–1).
    - na_rm: If True, removes rows with missing data before computing distances,
             but reinserts NaNs in original positions.

    Returns:
    - Mahalanobis distances (with NaNs where removed), or
    - Tuple of (distances, flags) if `flag=True`.
    """
    x = np.asarray(x)

    if confidence < 0 or confidence > 1:
        raise ValueError("Confidence must be between 0 and 1")

    # Identify rows to keep (complete cases)
    if na_rm:
        all_nan_mask = np.isnan(x).all(axis=1)
        partial_nan_mask = np.isnan(x).any(axis=1)
        valid_mask = ~partial_nan_mask & ~all_nan_mask
        x_filtered = x[valid_mask]
    else:
        x_filtered = x
        valid_mask = np.ones(x.shape[0], dtype=bool)

    if x_filtered.size == 0 or x_filtered.shape[0] < x_filtered.shape[1]:
        raise ValueError(
            "The input must have more observations than dimensions and cannot be empty."
        )

    # Compute Mahalanobis distance on filtered data
    mean_vector = np.mean(x_filtered, axis=0)
    cov_matrix = np.cov(x_filtered, rowvar=False)

    cond_number = np.linalg.cond(cov_matrix)
    if cond_number < 1 / np.finfo(cov_matrix.dtype).eps:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    else:
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

    centered_data = x_filtered - mean_vector
    mahalanobis_squared = np.einsum(
        "ij,jk,ik->i", centered_data, inv_cov_matrix, centered_data
    )
    distances_filtered = np.sqrt(mahalanobis_squared)

    # Reconstruct full-length array with NaNs
    distances = np.full(shape=(x.shape[0],), fill_value=np.nan)
    distances[valid_mask] = distances_filtered

    if flag:
        threshold = stats.chi2.ppf(confidence, df=x.shape[1])
        flags = distances > np.sqrt(threshold)
        return distances, flags

    return distances
