"""
Person-total correlation for detecting careless responding.

The person-total correlation measures how similar an individual's response pattern
is to the overall sample mean response pattern. Low correlations may indicate
careless or random responding.
"""

import numpy as np

from careless._validation import MatrixLike, validate_matrix_input


def person_total(
    x: MatrixLike,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Calculate person-total correlation for each individual.

    The person-total correlation (also called "personal biserial") measures
    the correlation between each individual's responses and the mean response
    across all individuals for each item. Low values suggest responses that
    deviate substantially from typical patterns.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - na_rm: If True, use pairwise complete observations for correlations.

    Returns:
    - A numpy array of person-total correlations for each individual.

    Raises:
    - ValueError: If inputs are invalid

    Example:
        >>> data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]]
        >>> scores = person_total(data)
        >>> print(scores)
        [1.0, -1.0, 1.0]
    """
    x_array = validate_matrix_input(x, min_columns=2)
    n_persons = x_array.shape[0]

    item_means = np.nanmean(x_array, axis=0) if na_rm else np.mean(x_array, axis=0)

    correlations = np.zeros(n_persons)

    for i in range(n_persons):
        person_responses = x_array[i, :]

        if na_rm:
            valid_mask = ~np.isnan(person_responses) & ~np.isnan(item_means)
            if valid_mask.sum() < 2:
                correlations[i] = np.nan
                continue
            p = person_responses[valid_mask]
            m = item_means[valid_mask]
        else:
            p = person_responses
            m = item_means

        if np.std(p) == 0 or np.std(m) == 0:
            correlations[i] = np.nan
        else:
            correlations[i] = np.corrcoef(p, m)[0, 1]

    return correlations
