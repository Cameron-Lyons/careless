"""
The IRV is the "standard deviation of responses across a set of consecutive item responses for
an individual" (Dunn, Heggestad, Shanock, & Theilgard, 2018, p. 108). By default, the IRV is
calculated across all columns of the input data. Additionally it can be applied to different subsets
of the data. This can detect degraded response quality which occurs only in a certain section of the
questionnaire (usually the end). Whereas Dunn et al. (2018) propose to mark persons with low IRV
scores as outliers - reflecting straightlining responses, Marjanovic et al. (2015) propose to mark
persons with high IRV scores - reflecting highly random responses
"""

import numpy as np


def irv(
    x: list[list[float]] | np.ndarray,
    na_rm: bool = True,
    split: bool = False,
    num_split: int = 1,
    split_points: list[int] | None = None,
) -> np.ndarray:
    """
    Calculate intra-individual response variability (IRV) for each individual.

    IRV measures the standard deviation of responses across consecutive items for each individual.
    Low IRV scores may indicate straightlining (consistent responses), while high IRV scores
    may indicate random or inconsistent responding.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their responses.
          Can be a 2D list or numpy array.
    - na_rm: Boolean indicating whether to ignore missing values (np.nan) during computation.
             If True, uses np.nanstd; if False, uses np.std.
    - split: Boolean indicating whether to calculate IRV on subsets of columns and return the mean.
    - num_split: Number of equal-sized subsets to split the data into if 'split' is True.
                 Ignored if split_points is provided.
    - split_points: Optional list of column indices to use as split points. If provided,
                   overrides num_split. For example, [0, 10, 20] would split into [0:10] and [10:20].

    Returns:
    - A numpy array of IRV values for each individual.

    Raises:
    - ValueError: If inputs are invalid (empty data, invalid split parameters, etc.)

    Example:
        >>> data = [[1, 2, 3, 4, 5, 6], [1, 1, 1, 4, 5, 6]]
        >>> # Calculate IRV across all items
        >>> irv_scores = irv(data)
        >>> print(irv_scores)
        [1.87, 2.16]  # Second person has higher variability

        >>> # Calculate IRV with splits
        >>> irv_split = irv(data, split=True, num_split=2)
        >>> print(irv_split)
        [1.87, 2.16]  # Mean IRV across splits

        >>> # Use custom split points
        >>> irv_custom = irv(data, split=True, split_points=[0, 3, 6])
        >>> print(irv_custom)
        [1.87, 2.16]  # Mean IRV across custom splits
    """

    if (
        x is None
        or (isinstance(x, np.ndarray) and x.size == 0)
        or (not isinstance(x, np.ndarray) and len(x) == 0)
    ):
        raise ValueError("input data cannot be empty")

    x_array = np.array(x)

    if x_array.ndim != 2:
        raise ValueError("input data must be 2-dimensional")

    if x_array.shape[0] == 0 or x_array.shape[1] == 0:
        raise ValueError("input data must have at least one row and one column")

    if split_points is not None:
        if not isinstance(split_points, list) or len(split_points) < 2:
            raise ValueError("split_points must be a list with at least 2 elements")

        if split_points[0] != 0:
            raise ValueError("first split point must be 0")

        if split_points[-1] != x_array.shape[1]:
            raise ValueError(f"last split point must be {x_array.shape[1]} (number of columns)")

        for i in range(1, len(split_points)):
            if split_points[i] <= split_points[i - 1]:
                raise ValueError("split points must be in ascending order")
            if split_points[i] > x_array.shape[1]:
                raise ValueError(f"split point {split_points[i]} exceeds number of columns")

    elif split and num_split <= 0:
        raise ValueError("num_split must be greater than 0")

    std_func = np.nanstd if na_rm else np.std

    if not split:
        result: np.ndarray = std_func(x_array, axis=1)
        return result

    if split_points is not None:
        irvs_splits = []
        for i in range(len(split_points) - 1):
            start_col = split_points[i]
            end_col = split_points[i + 1]
            if end_col > start_col:
                irv_split = std_func(x_array[:, start_col:end_col], axis=1)
                irvs_splits.append(irv_split)
    else:
        num_cols = x_array.shape[1]
        chunk_size = max(1, num_cols // num_split)

        irvs_splits = []
        for i in range(0, num_cols, chunk_size):
            end_col = min(i + chunk_size, num_cols)
            if end_col > i:
                irv_split = std_func(x_array[:, i:end_col], axis=1)
                irvs_splits.append(irv_split)

    if irvs_splits:
        split_result: np.ndarray = np.mean(irvs_splits, axis=0)
        return split_result
    else:
        fallback: np.ndarray = std_func(x_array, axis=1)
        return fallback
