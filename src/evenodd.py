"""This module contains the evenodd function for calculating even-odd consistency scores."""
import numpy as np
from typing import List, Union, Tuple


def evenodd(
    x: Union[List[List[float]], np.ndarray], factors: List[int], diag: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate even-odd consistency scores for each individual based on the provided factors.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their responses.
    - factors: List of integers specifying the length of each factor in the dataset.
    - diag: Boolean to optionally return a column with the number of available even/odd pairs.

    Returns:
    - A numpy array of even-odd consistency scores or a tuple of scores and diagnostic values.
    """

    x = np.array(x)

    avg_correlations = np.zeros(x.shape[0])

    diag_vals = np.zeros(x.shape[0], dtype=int)

    start_col = 0
    for factor_size in factors:
        end_col = start_col + factor_size

        even_cols = x[:, start_col:end_col:2]
        odd_cols = x[:, (start_col + 1) : end_col : 2]

        correlations = np.array(
            [np.corrcoef(e, o)[0, 1] for e, o in zip(even_cols, odd_cols)]
        )

        correlations[np.isnan(correlations)] = 0

        avg_correlations += correlations

        diag_vals += np.minimum(even_cols.shape[1], odd_cols.shape[1])

        start_col = end_col

    avg_correlations /= len(factors)

    if diag:
        return avg_correlations, diag_vals
    else:
        return avg_correlations
