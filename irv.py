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
from typing import List, Union, Optional


def irv(
    x: Union[List[List[float]], np.ndarray],
    na_rm: bool = True,
    split: bool = False,
    num_split: Optional[int] = 1,
) -> np.ndarray:
    """
    Calculate intra-individual response variability (IRV) for each individual.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their responses.
    - na_rm: Boolean indicating whether to ignore missing values (np.nan) during computation.
    - split: Boolean indicating whether to additionally calculate the IRV on subsets of columns.
    - num_split: Number of subsets the data is to be split into if 'split' is True.

    Returns:
    - A numpy array of IRV values for each individual.
    """

    if na_rm:
        x = np.nan_to_num(x, nan=np.mean(x, axis=1, keepdims=True))

    irvs = np.std(x, axis=1)

    if split:
        chunk_size = x.shape[1] // num_split
        irvs_splits = [
            np.std(x[:, i : i + chunk_size], axis=1)
            for i in range(0, x.shape[1], chunk_size)
        ]
        irvs = np.mean(irvs_splits, axis=0)

    return irvs
