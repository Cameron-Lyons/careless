"""This module contains the evenodd function for calculating even-odd consistency scores."""

import numpy as np
from typing import List, Union, Tuple


def calculate_correlations(even_cols: np.ndarray, odd_cols: np.ndarray) -> np.ndarray:
    """
    Calculates correlations between even and odd columns.
    
    Parameters:
    - even_cols: Array of even-indexed columns
    - odd_cols: Array of odd-indexed columns
    
    Returns:
    - Array of correlation coefficients between corresponding even/odd pairs
    """
    min_cols = min(even_cols.shape[1], odd_cols.shape[1])
    if min_cols == 0:
        return np.array([])
    
    correlations = np.array([
        np.corrcoef(even_cols[:, i], odd_cols[:, i])[0, 1] 
        for i in range(min_cols)
    ])

    correlations[np.isnan(correlations)] = 0 
    return correlations


def evenodd(
    x: Union[List[List[float]], np.ndarray], 
    factors: List[int], 
    diag: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate even-odd consistency scores for each individual based on the provided factors.
    
    This function splits each factor into even and odd columns, calculates correlations
    between corresponding pairs, and returns the average correlation for each individual.
    
    Parameters:
    - x: A matrix of data where rows are individuals and columns are their responses.
          Can be a 2D list or numpy array.
    - factors: List of integers specifying the length of each factor in the dataset.
               The sum of factors should equal the number of columns in x.
    - diag: Boolean to optionally return diagnostic values (number of valid correlations per individual).
    
    Returns:
    - A numpy array of even-odd consistency scores (average correlations per individual)
    - If diag=True, returns a tuple of (scores, diagnostic_values)
    
    Raises:
    - ValueError: If factors don't sum to the number of columns, or if data is empty
    
    Example:
        >>> data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
        >>> factors = [4, 2]  # First factor has 4 items, second has 2
        >>> scores = evenodd(data, factors)
        >>> print(scores)
        [0.5, 0.5]  # Average correlations for each individual
    """
    
    if not factors:
        raise ValueError("factors list cannot be empty")
    
    if not x or len(x) == 0:
        raise ValueError("input data cannot be empty")
    
    x_array = np.array(x)
    
    if x_array.ndim != 2:
        raise ValueError("input data must be 2-dimensional")
    
    expected_cols = sum(factors)
    if x_array.shape[1] != expected_cols:
        raise ValueError(
            f"sum of factors ({expected_cols}) must equal number of columns ({x_array.shape[1]})"
        )
    
    num_individuals = x_array.shape[0]
    all_correlations = []
    diag_vals = np.zeros(num_individuals, dtype=int)
    
    start_col = 0
    for factor_size in factors:
        if factor_size < 2:
            start_col += factor_size
            continue
            
        end_col = start_col + factor_size
        
        even_cols = x_array[:, start_col:end_col:2]
        odd_cols = x_array[:, start_col + 1:end_col:2]
        
        corrs = calculate_correlations(even_cols, odd_cols)
        
        if len(corrs) > 0:
            all_correlations.append(corrs)
            diag_vals += len(corrs)
        
        start_col = end_col
    
    if all_correlations:
        stacked_corrs = np.column_stack(all_correlations)
        avg_correlations = np.mean(stacked_corrs, axis=1)
    else:
        avg_correlations = np.zeros(num_individuals)
    
    return (avg_correlations, diag_vals) if diag else avg_correlations
