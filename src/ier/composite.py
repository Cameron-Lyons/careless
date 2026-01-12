"""
Composite index combining multiple IER detection indices.

Research suggests combining multiple indices improves detection accuracy. The "Best Subset"
approach (Curran, 2016; Meade & Craig, 2012) recommends combining indices that capture
different types of careless responding: consistency-based, pattern-based, and outlier-based.

References:
- Curran, P. G. (2016). Methods for the detection of carelessly invalid responses in
  survey data. Journal of Experimental Social Psychology, 66, 4-19.
- Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data.
  Psychological Methods, 17(3), 437-455.
"""

import contextlib
from typing import Any, Literal

import numpy as np

from ier._validation import MatrixLike, validate_matrix_input
from ier.evenodd import evenodd
from ier.irv import irv
from ier.longstring import longstring
from ier.mahad import mahad
from ier.person_total import person_total
from ier.psychsyn import psychsyn


def composite(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: Literal["mean", "sum", "max"] = "mean",
    standardize: bool = True,
    na_rm: bool = True,
    psychsyn_critval: float = 0.6,
    evenodd_factors: list[int] | None = None,
) -> np.ndarray:
    """
    Calculate a composite IER index combining multiple detection methods.

    This function computes multiple IER indices, standardizes them to z-scores,
    and combines them into a single composite score. Higher composite scores
    indicate greater likelihood of careless responding.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - indices: List of indices to include. Options: "irv", "longstring", "mahad",
              "psychsyn", "evenodd", "person_total". Default includes all except
              evenodd (which requires factor specification).
    - method: How to combine indices. "mean" (default), "sum", or "max".
    - standardize: If True (default), standardize each index to z-scores before combining.
    - na_rm: Handle missing values in individual indices.
    - psychsyn_critval: Critical correlation value for psychometric synonyms (default 0.6).
    - evenodd_factors: Factor lengths for even-odd consistency. Required if "evenodd"
                      is in indices. List of integers where each integer is the number
                      of items in that factor (e.g., [5, 5, 5] for three 5-item scales).

    Returns:
    - A numpy array of composite scores for each individual. Higher scores indicate
      greater likelihood of careless responding.

    Raises:
    - ValueError: If invalid indices specified or evenodd requested without factors.

    Example:
        >>> data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        >>> scores = composite(data)
        >>> print(scores)
        [-0.5, 1.2, -0.3]

        >>> scores = composite(data, indices=["irv", "longstring"])
        >>> print(scores)
        [-0.7, 1.5, -0.2]
    """
    x_array = validate_matrix_input(x, check_type=False)

    valid_indices = {"irv", "longstring", "mahad", "psychsyn", "evenodd", "person_total"}

    if indices is None:
        indices = ["irv", "longstring", "mahad", "psychsyn", "person_total"]

    for idx in indices:
        if idx not in valid_indices:
            raise ValueError(f"invalid index '{idx}'. Valid options: {valid_indices}")

    if "evenodd" in indices and evenodd_factors is None:
        raise ValueError("evenodd_factors must be provided when using evenodd index")

    if method not in ["mean", "sum", "max"]:
        raise ValueError("method must be 'mean', 'sum', or 'max'")

    index_scores: dict[str, np.ndarray] = {}

    if "irv" in indices:
        irv_scores = irv(x_array, na_rm=na_rm)
        index_scores["irv"] = -irv_scores

    if "longstring" in indices:
        response_strings = [
            "".join(str(int(v)) if not np.isnan(v) else "" for v in row) for row in x_array
        ]
        ls_results = longstring(response_strings)
        ls_scores = np.array([r[1] if r is not None else 0 for r in ls_results], dtype=float)
        index_scores["longstring"] = ls_scores

    if "mahad" in indices:
        with contextlib.suppress(ValueError):
            mahad_result = mahad(x_array, na_rm=na_rm)
            if isinstance(mahad_result, np.ndarray):
                index_scores["mahad"] = mahad_result

    if "psychsyn" in indices:
        with contextlib.suppress(ValueError), np.errstate(divide="ignore", invalid="ignore"):
            psyn_result = psychsyn(x_array, critval=psychsyn_critval, resample_na=na_rm)
            if isinstance(psyn_result, np.ndarray):
                index_scores["psychsyn"] = -psyn_result

    if "evenodd" in indices and evenodd_factors is not None:
        with contextlib.suppress(ValueError):
            eo_result = evenodd(x_array, factors=evenodd_factors)
            if isinstance(eo_result, np.ndarray):
                index_scores["evenodd"] = -eo_result

    if "person_total" in indices:
        with contextlib.suppress(ValueError):
            index_scores["person_total"] = -person_total(x_array, na_rm=na_rm)

    if len(index_scores) == 0:
        raise ValueError("no valid indices could be computed from the data")

    if standardize:
        standardized_scores = {}
        for name, scores in index_scores.items():
            valid_mask = ~np.isnan(scores)
            if np.sum(valid_mask) > 1:
                mean_val = np.nanmean(scores)
                std_val = np.nanstd(scores)
                if std_val > 0:
                    standardized_scores[name] = (scores - mean_val) / std_val
                else:
                    standardized_scores[name] = np.zeros_like(scores)
            else:
                standardized_scores[name] = scores
        index_scores = standardized_scores

    score_matrix = np.column_stack(list(index_scores.values()))

    if method == "mean":
        result: np.ndarray = np.nanmean(score_matrix, axis=1)
    elif method == "sum":
        result = np.nansum(score_matrix, axis=1)
    else:
        result = np.nanmax(score_matrix, axis=1)

    return result


def composite_flag(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: Literal["mean", "sum", "max"] = "mean",
    threshold: float | None = None,
    percentile: float = 95.0,
    standardize: bool = True,
    na_rm: bool = True,
    psychsyn_critval: float = 0.6,
    evenodd_factors: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate composite IER scores and flag potential careless responders.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - indices: List of indices to include (see composite() for options).
    - method: How to combine indices ("mean", "sum", or "max").
    - threshold: Absolute threshold above which to flag. If None, uses percentile.
    - percentile: Percentile cutoff for flagging (default 95th percentile).
    - standardize: Standardize indices to z-scores before combining.
    - na_rm: Handle missing values.
    - psychsyn_critval: Critical value for psychometric synonyms.
    - evenodd_factors: Factor structure for even-odd consistency.

    Returns:
    - Tuple of (composite_scores, flags) where flags is True for suspected
      careless responders.

    Example:
        >>> data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        >>> scores, flags = composite_flag(data)
        >>> print(flags)
        [False, True, False]
    """
    scores = composite(
        x,
        indices=indices,
        method=method,
        standardize=standardize,
        na_rm=na_rm,
        psychsyn_critval=psychsyn_critval,
        evenodd_factors=evenodd_factors,
    )

    valid_scores = scores[~np.isnan(scores)]

    if threshold is None:
        if len(valid_scores) == 0:
            threshold = 0.0
        else:
            threshold = float(np.percentile(valid_scores, percentile))

    flags = np.zeros(len(scores), dtype=bool)
    valid_mask = ~np.isnan(scores)
    flags[valid_mask] = scores[valid_mask] > threshold

    return scores, flags


def composite_summary(
    x: MatrixLike,
    indices: list[str] | None = None,
    method: Literal["mean", "sum", "max"] = "mean",
    standardize: bool = True,
    na_rm: bool = True,
    psychsyn_critval: float = 0.6,
    evenodd_factors: list[int] | None = None,
) -> dict[str, Any]:
    """
    Calculate composite scores with detailed summary statistics.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - indices: List of indices to include.
    - method: Combination method.
    - standardize: Standardize before combining.
    - na_rm: Handle missing values.
    - psychsyn_critval: Critical value for psychometric synonyms.
    - evenodd_factors: Factor structure for even-odd consistency.

    Returns:
    - Dictionary with composite scores, individual index scores, and statistics.

    Example:
        >>> data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        >>> summary = composite_summary(data)
        >>> print(summary.keys())
        dict_keys(['composite', 'indices', 'mean', 'std', 'min', 'max', ...])
    """
    x_array = validate_matrix_input(x, check_type=False)

    if indices is None:
        indices = ["irv", "longstring", "mahad", "psychsyn", "person_total"]

    if "evenodd" in indices and evenodd_factors is None:
        raise ValueError("evenodd_factors must be provided when using evenodd index")

    individual_scores: dict[str, np.ndarray] = {}

    if "irv" in indices:
        individual_scores["irv"] = irv(x_array, na_rm=na_rm)

    if "longstring" in indices:
        response_strings = [
            "".join(str(int(v)) if not np.isnan(v) else "" for v in row) for row in x_array
        ]
        ls_results = longstring(response_strings)
        individual_scores["longstring"] = np.array(
            [r[1] if r is not None else 0 for r in ls_results], dtype=float
        )

    if "mahad" in indices:
        with contextlib.suppress(ValueError):
            mahad_result = mahad(x_array, na_rm=na_rm)
            if isinstance(mahad_result, np.ndarray):
                individual_scores["mahad"] = mahad_result

    if "psychsyn" in indices:
        with contextlib.suppress(ValueError), np.errstate(divide="ignore", invalid="ignore"):
            psyn_result = psychsyn(x_array, critval=psychsyn_critval, resample_na=na_rm)
            if isinstance(psyn_result, np.ndarray):
                individual_scores["psychsyn"] = psyn_result

    if "evenodd" in indices and evenodd_factors is not None:
        with contextlib.suppress(ValueError):
            eo_result = evenodd(x_array, factors=evenodd_factors)
            if isinstance(eo_result, np.ndarray):
                individual_scores["evenodd"] = eo_result

    if "person_total" in indices:
        with contextlib.suppress(ValueError):
            individual_scores["person_total"] = person_total(x_array, na_rm=na_rm)

    composite_scores = composite(
        x_array,
        indices=indices,
        method=method,
        standardize=standardize,
        na_rm=na_rm,
        psychsyn_critval=psychsyn_critval,
        evenodd_factors=evenodd_factors,
    )

    valid_composite = composite_scores[~np.isnan(composite_scores)]

    return {
        "composite": composite_scores,
        "indices": individual_scores,
        "indices_used": list(individual_scores.keys()),
        "method": method,
        "standardized": standardize,
        "mean": float(np.nanmean(composite_scores)) if len(valid_composite) > 0 else np.nan,
        "std": float(np.nanstd(composite_scores)) if len(valid_composite) > 0 else np.nan,
        "min": float(np.nanmin(composite_scores)) if len(valid_composite) > 0 else np.nan,
        "max": float(np.nanmax(composite_scores)) if len(valid_composite) > 0 else np.nan,
        "n_total": len(composite_scores),
        "n_valid": int(np.sum(~np.isnan(composite_scores))),
    }
