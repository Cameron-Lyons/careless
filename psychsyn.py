"""
Takes a matrix of item responses and identifies item pairs that are highly correlated within the
overall dataset. What defines "highly correlated" is set by the critical value (e.g., r > .60). Each
respondents’ psychometric synonym score is then computed as the within-person correlation be-
tween the identified item-pairs. Alternatively computes the psychometric antonym score which is a
variant that uses item pairs that are highly negatively correlated
"""
import numpy as np
from typing import List, Union, Tuple


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

    if anto:
        item_pairs = np.argwhere(np.tril(item_correlations, -1) <= -critval)
    else:
        item_pairs = np.argwhere(np.tril(item_correlations, -1) >= critval)

    # Calculate the within-person correlation for these item pairs
    scores = []
    diag_values = []

    for person_responses in x:
        person_corrs = []

        for i, j in item_pairs:
            response_i = person_responses[i]
            response_j = person_responses[j]

            if not np.isnan(response_i) and not np.isnan(response_j):
                person_corrs.append(np.corrcoef(response_i, response_j)[0, 1])

        if resample_na and np.isnan(np.mean(person_corrs)):
            person_corrs = [
                np.random.choice([-1, 1]) * np.abs(val) for val in person_corrs
            ]

        scores.append(np.nanmean(person_corrs))
        if diag:
            diag_values.append(len(person_corrs))

    if diag:
        return np.array(scores), np.array(diag_values)
    else:
        return np.array(scores)
