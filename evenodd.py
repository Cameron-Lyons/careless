"""
Takes a matrix of item responses and a vector of integers representing the length each factor. The
even-odd consistency score is then computed as the within-person correlation between the even and
odd subscales over all the factors.
"""

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

    correlations = []
    if diag:
        diag_values = []

    for person_responses in x:
        person_corrs = []
        start_idx = 0

        for factor in factors:
            even_indices = list(range(start_idx, start_idx + factor, 2))
            odd_indices = list(range(start_idx + 1, start_idx + factor, 2))

            even_responses = person_responses[even_indices]
            odd_responses = person_responses[odd_indices]

            # Handle missing data by creating a mask of non-missing pairs
            mask = ~np.isnan(even_responses) & ~np.isnan(odd_responses)
            if diag:
                diag_values.append(np.sum(mask))

            even_responses = even_responses[mask]
            odd_responses = odd_responses[mask]

            if len(even_responses) > 0:  # Guard against empty data after masking
                person_corrs.append(np.corrcoef(even_responses, odd_responses)[0, 1])

            start_idx += factor

        correlations.append(
            np.nanmean(person_corrs)
        )  # Handle potential NaNs in correlations

    if diag:
        return np.array(correlations), np.array(diag_values)
    else:
        return np.array(correlations)
