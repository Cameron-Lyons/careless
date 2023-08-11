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
import scipy.stats as stats
from typing import Tuple, Union


def mahad(
    x: np.ndarray,
    flag: bool = False,
    confidence: float = 0.95,
    na_rm: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Computes Mahalanobis Distance for a matrix of data.

    Parameters:
    - x: Matrix of data.
    - flag: If True, flags potential outliers based on the confidence level.
    - confidence: Confidence level for flagging outliers.
    - na_rm: If True, removes rows with missing data.

    Returns:
    - Mahalanobis distances or a tuple of distances and outlier flags.
    """

    if na_rm:
        x = x[~np.isnan(x).any(axis=1)]

    mean_vector = np.mean(x, axis=0)
    cov_matrix = np.cov(x, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    centered_data = x - mean_vector
    distances = np.sqrt(np.sum(centered_data @ inv_cov_matrix * centered_data, axis=1))

    if flag:
        threshold = stats.chi2.ppf(confidence, df=x.shape[1])
        flags = distances**2 > threshold
        return distances, flags

    return distances
