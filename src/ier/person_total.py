"""
Person-total correlation for detecting careless responding.

The person-total correlation measures how similar an individual's response pattern
is to the overall sample mean response pattern. Low correlations may indicate
careless or random responding.
"""

import numpy as np

from ier._validation import MatrixLike, validate_matrix_input


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

    item_means = np.nanmean(x_array, axis=0) if na_rm else np.mean(x_array, axis=0)

    if na_rm:
        valid_mask = ~np.isnan(x_array)
        item_means_broadcast = np.where(valid_mask, item_means, np.nan)
    else:
        item_means_broadcast = np.broadcast_to(item_means, x_array.shape)

    with np.errstate(invalid="ignore"):
        x_centered = x_array - np.nanmean(x_array, axis=1, keepdims=True)
        m_centered = item_means_broadcast - np.nanmean(item_means_broadcast, axis=1, keepdims=True)

        cov = np.nansum(x_centered * m_centered, axis=1)
        x_std = np.sqrt(np.nansum(x_centered**2, axis=1))
        m_std = np.sqrt(np.nansum(m_centered**2, axis=1))

        correlations = cov / (x_std * m_std)

    valid_counts = np.sum(~np.isnan(x_array), axis=1) if na_rm else x_array.shape[1]
    correlations = np.where(valid_counts < 2, np.nan, correlations)

    return correlations
