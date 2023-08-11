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
from typing import List, Union


def irv(data: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Calculate intra-individual response variability (IRV) for each individual.

    Parameters:
    - data: 2D list or numpy array where rows are individuals and columns are their responses.

    Returns:
    - A numpy array of IRV values for each individual.
    """

    return np.std(data, axis=1)
