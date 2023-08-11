"""Find and graph Mahalanobis Distance (D) and flag potential outliers

Takes a matrix of item responses and computes Mahalanobis D. Can additionally return a
vector of binary outlier flags.
Mahalanobis distance is calculated using a function which supports missing values."""

import numpy as np
from scipy.spatial import distance
from typing import List, Tuple


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
