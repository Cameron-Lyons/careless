"""
Takes a matrix of item responses and identifies item pairs that are highly correlated within the
overall dataset. What defines "highly correlated" is set by the critical value (e.g., r > .60). Each
respondents' psychometric synonym score is then computed as the within-person correlation be-
tween the identified item-pairs. Alternatively computes the psychometric antonym score which is a
variant that uses item pairs that are highly negatively correlated
"""

import numpy as np
from typing import List, Union, Tuple


def get_highly_correlated_pairs(
    item_correlations: np.ndarray, critval: float, anto: bool
) -> np.ndarray:
    if anto:
        return np.argwhere(np.tril(item_correlations, -1) <= critval)
    return np.argwhere(np.tril(item_correlations, -1) >= critval)


def compute_person_correlations(
    response_i: np.ndarray, response_j: np.ndarray
) -> np.ndarray:
    return (
        (response_i - response_i.mean(axis=1, keepdims=True))
        * (response_j - response_j.mean(axis=1, keepdims=True))
    ) / (response_i.std(axis=1, keepdims=True) * response_j.std(axis=1, keepdims=True))


def psychsyn(
    x: Union[List[List[float]], np.ndarray],
    critval: float = 0.60,
    anto: bool = False,
    diag: bool = False,
    resample_na: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate psychometric synonym (or antonym) scores based on the provided item response matrix.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their item responses.
    - critval: Minimum magnitude of correlation for items to be considered synonyms/antonyms.
    - anto: Boolean indicating whether to compute antonym scores (highly negatively correlated items).
    - diag: Boolean to optionally return the number of item pairs available for each observation.
    - resample_na: Boolean to indicate resampling when encountering NA for a respondent.

    Returns:
    - A numpy array of psychometric synonym/antonym scores or a tuple of scores and diagnostic values.
    """

    item_correlations = np.corrcoef(x, rowvar=False)
    item_pairs = get_highly_correlated_pairs(item_correlations, critval, anto)

    response_i = x[:, item_pairs[:, 0]]
    response_j = x[:, item_pairs[:, 1]]

    person_corrs = compute_person_correlations(response_i, response_j)

    invalid_pairs = np.isnan(response_i) | np.isnan(response_j)
    person_corrs[invalid_pairs] = np.nan

    if resample_na:
        nan_corrs = np.isnan(person_corrs.mean(axis=1))
        person_corrs[nan_corrs] = np.random.choice(
            [-1, 1], size=nan_corrs.sum()
        ) * np.abs(person_corrs[nan_corrs])

    scores = np.nanmean(person_corrs, axis=1)

    if diag:
        diag_values = np.sum(~np.isnan(person_corrs), axis=1)
        return scores, diag_values
    else:
        return scores


def psychsyn_critval(
    x: Union[List[List[float]], np.ndarray], anto: bool = False
) -> List[Tuple[int, int, float]]:
    """
    Calculate and order pairwise correlations for all items in the provided item response matrix.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their item responses.
    - anto: Boolean indicating whether to order correlations by largest negative values.

    Returns:
    - A list of tuples containing item pairs and their correlation, ordered by magnitude.
    """

    item_correlations = np.corrcoef(x, rowvar=False)

    correlation_list = [
        (i, j, item_correlations[i, j])
        for i in range(item_correlations.shape[0])
        for j in range(i + 1, item_correlations.shape[1])
    ]

    if anto:
        correlation_list = sorted(correlation_list, key=lambda x: x[2])
    else:
        correlation_list = sorted(correlation_list, key=lambda x: -x[2])

    return correlation_list


def psychant(
    x: Union[List[List[float]], np.ndarray], critval: float = -0.60, diag: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate the psychometric antonym score.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their item responses.
    - critval: Minimum magnitude of correlation for items to be considered antonyms.
    - diag: Boolean to optionally return the number of item pairs available for each observation.

    Returns:
    - A numpy array of psychometric antonym scores or a tuple of scores and diagnostic values.
    """
    return psychsyn(x, critval=critval, anto=True, diag=diag)
