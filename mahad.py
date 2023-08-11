"""Find and graph Mahalanobis Distance (D) and flag potential outliers

Takes a matrix of item responses and computes Mahalanobis D. Can additionally return a
vector of binary outlier flags.
Mahalanobis distance is calculated using a function which supports missing values."""

import numpy as np
from scipy.spatial import distance


def mahalanobis(x, mean, inv_cov_matrix):
    """Compute the Mahalanobis Distance"""
    x_minus_mean = x - mean
    return np.sqrt(x_minus_mean.T.dot(inv_cov_matrix).dot(x_minus_mean))
