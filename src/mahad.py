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
from typing import List


def mahalanobis(x: np.ndarray, mean: np.ndarray, inv_cov_matrix: np.ndarray) -> float:
    """Compute the Mahalanobis Distance"""
    x_minus_mean = x - mean
    return np.sqrt(x_minus_mean.T.dot(inv_cov_matrix).dot(x_minus_mean))


def compute_mahalanobis(data: np.ndarray) -> List[float]:
    """Computes Mahalanobis Distance for a dataset"""
    mean = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    distances = [
        mahalanobis(data[i, :], mean, inv_cov_matrix) for i in range(data.shape[0])
    ]
    return distances


def flag_outliers(distances: List[float], threshold: float = 2.5) -> List[int]:
    """Flag values that exceed the threshold"""
    return [1 if d > threshold else 0 for d in distances]
