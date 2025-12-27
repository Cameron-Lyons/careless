"""Unit tests for the careless library.

This module contains comprehensive unit tests for all functions in the careless
library, including longstring, IRV, MAHAD, even-odd, and psychometric synonym
detection functions.
"""

import unittest
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.stats as stats

from careless.evenodd import evenodd
from careless.irv import irv
from careless.longstring import (
    avgstr_message,
    longstr_message,
    longstring,
    run_length_decode,
    run_length_encode,
)
from careless.mahad import mahad, mahad_summary
from careless.psychsyn import psychant, psychsyn, psychsyn_critval, psychsyn_summary


class TestLongstring(unittest.TestCase):
    """Test suite for longstring module functions.

    Tests run-length encoding/decoding, longest string detection,
    average string length calculation, and the main longstring function.
    """

    def test_run_length_encode(self) -> None:
        """Test run-length encoding produces correct character-count tuples."""
        self.assertEqual(
            run_length_encode("AAABBBCCDAA"),
            [("A", 3), ("B", 3), ("C", 2), ("D", 1), ("A", 2)],
        )
        self.assertEqual(run_length_encode("A"), [("A", 1)])
        self.assertEqual(run_length_encode(""), [])

    def test_run_length_encode_validation(self) -> None:
        """Test run-length encoding raises TypeError for invalid inputs."""
        with self.assertRaises(TypeError):
            run_length_encode(123)
        with self.assertRaises(TypeError):
            run_length_encode(None)

    def test_run_length_decode(self) -> None:
        """Test run-length decoding reconstructs original string correctly."""
        self.assertEqual(
            run_length_decode([("A", 3), ("B", 3), ("C", 2), ("D", 1), ("A", 2)]),
            "AAABBBCCDAA",
        )
        self.assertEqual(run_length_decode([("A", 1)]), "A")
        self.assertEqual(run_length_decode([]), "")

    def test_run_length_decode_validation(self) -> None:
        """Test run-length decoding raises TypeError for invalid inputs."""
        with self.assertRaises(TypeError):
            run_length_decode("not a list")
        with self.assertRaises(TypeError):
            run_length_decode(None)

    def test_longstr_message(self) -> None:
        """Test longest string message returns correct character and length."""
        self.assertEqual(longstr_message("AAAABBBCCDAA"), ("A", 4))
        self.assertEqual(longstr_message("A"), ("A", 1))
        self.assertEqual(longstr_message(""), None)

    def test_longstr_message_validation(self) -> None:
        """Test longest string message raises TypeError for invalid inputs."""
        with self.assertRaises(TypeError):
            longstr_message(123)
        with self.assertRaises(TypeError):
            longstr_message(None)

    def test_avgstr_message(self) -> None:
        """Test average string length calculation returns correct values."""
        self.assertAlmostEqual(avgstr_message("AAABBBCCDAA"), 2.2)
        self.assertEqual(avgstr_message("A"), 1.0)
        self.assertEqual(avgstr_message(""), 0.0)

    def test_avgstr_message_validation(self) -> None:
        """Test average string message raises TypeError for invalid inputs."""
        with self.assertRaises(TypeError):
            avgstr_message(123)
        with self.assertRaises(TypeError):
            avgstr_message(None)

    def test_longstring(self) -> None:
        """Test main longstring function with single strings and lists."""
        self.assertEqual(longstring("AAAABBBCCDAA"), ("A", 4))
        self.assertEqual(longstring(["AAAABBBCCDAA", "A", ""]), [("A", 4), ("A", 1), None])

        self.assertAlmostEqual(longstring("AAABBBCCDAA", avg=True), 2.2)
        self.assertAlmostEqual(longstring(["AAABBBCCDAA", "A", ""], avg=True), [2.2, 1.0, 0.0])

    def test_longstring_numpy_array(self) -> None:
        """Test longstring function accepts numpy array input."""
        data: npt.NDArray[np.str_] = np.array(["AAAABBBCCDAA", "A", ""])
        result: list[tuple[str, int] | None] = longstring(data)
        expected: list[tuple[str, int] | None] = [("A", 4), ("A", 1), None]
        self.assertEqual(result, expected)

    def test_longstring_validation(self) -> None:
        """Test longstring function raises appropriate errors for invalid inputs."""
        with self.assertRaises(ValueError):
            longstring(None)
        with self.assertRaises(ValueError):
            longstring([])
        with self.assertRaises(TypeError):
            longstring(123)
        with self.assertRaises(TypeError):
            longstring(["abc", 123, "def"])

    def test_longstring_edge_cases(self) -> None:
        """Test longstring function handles edge cases correctly."""
        self.assertEqual(longstring("a"), ("a", 1))

        self.assertEqual(longstring("abcdef"), ("a", 1))

        self.assertEqual(longstring("aaaa"), ("a", 4))


class TestIRV(unittest.TestCase):
    """Test suite for intra-individual response variability (IRV) function.

    Tests basic IRV calculation, handling of missing values, and split-half
    IRV computation with various configurations.
    """

    def test_basic_irv(self) -> None:
        """Test basic IRV calculation produces expected standard deviations."""
        x: npt.NDArray[np.float64] = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [1, 3, 5, 7]])
        result: npt.NDArray[np.float64] = irv(x)
        expected: npt.NDArray[np.float64] = np.array([1.11803399, 2.23606798, 2.23606798])
        np.testing.assert_almost_equal(result, expected)

    def test_irv_with_list_input(self) -> None:
        """Test IRV function accepts list input and converts appropriately."""
        x: list[list[int]] = [[1, 2, 3, 4], [2, 4, 6, 8], [1, 3, 5, 7]]
        result: npt.NDArray[np.float64] = irv(x)
        expected: npt.NDArray[np.float64] = np.array([1.11803399, 2.23606798, 2.23606798])
        np.testing.assert_almost_equal(result, expected)

    def test_irv_with_na(self) -> None:
        """Test IRV calculation correctly handles missing values when na_rm=True."""
        x: npt.NDArray[np.float64] = np.array([[1, np.nan, 3, 4], [2, 4, 6, 8], [np.nan, 3, 5, 7]])
        result: npt.NDArray[np.float64] = irv(x, na_rm=True)
        expected: npt.NDArray[np.float64] = np.array([1.2472191, 2.236068, 1.6329932])
        np.testing.assert_almost_equal(result, expected)

    def test_irv_with_split(self) -> None:
        """Test split-half IRV calculation with automatic splitting."""
        x: npt.NDArray[np.float64] = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [1, 3, 5, 7]])
        result: npt.NDArray[np.float64] = irv(x, split=True, num_split=2)
        expected: npt.NDArray[np.float64] = np.array([0.5, 1.0, 1.0])
        np.testing.assert_almost_equal(result, expected)

    def test_irv_with_custom_split_points(self) -> None:
        """Test split IRV with custom split point indices."""
        x: npt.NDArray[np.float64] = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [1, 3, 5, 7]])
        result: npt.NDArray[np.float64] = irv(x, split=True, split_points=[0, 2, 4])
        self.assertEqual(len(result), 3)

    def test_irv_with_split_and_na(self) -> None:
        """Test split IRV correctly handles missing values."""
        x: npt.NDArray[np.float64] = np.array([[1, 2, np.nan, 4], [2, 4, 6, 8], [1, 3, np.nan, 7]])
        result: npt.NDArray[np.float64] = irv(x, na_rm=True, split=True, num_split=2)
        expected: npt.NDArray[np.float64] = np.array([0.25, 1.0, 0.5])
        np.testing.assert_almost_equal(result, expected)

    def test_irv_validation(self) -> None:
        """Test IRV function raises appropriate errors for invalid inputs."""
        with self.assertRaises(ValueError):
            irv(None)
        with self.assertRaises(ValueError):
            irv([])
        with self.assertRaises(ValueError):
            irv([[1, 2, 3]], split=True, split_points=[0, 5])
        with self.assertRaises(ValueError):
            irv([[1, 2, 3]], split=True, split_points=[0, 2, 1])

    def test_irv_edge_cases(self) -> None:
        """Test IRV handles edge cases like single-column and excessive splits."""
        x: npt.NDArray[np.float64] = np.array([[1], [2], [3]])
        result: npt.NDArray[np.float64] = irv(x)
        self.assertEqual(len(result), 3)

        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = irv(x, split=True, num_split=5)
        self.assertEqual(len(result), 2)


class TestMahadFunction(unittest.TestCase):
    """Test suite for Mahalanobis distance (MAHAD) function.

    Tests distance calculation, outlier flagging with different methods,
    handling of missing values, and summary statistics generation.
    """

    def setUp(self) -> None:
        """Initialize test data with normal cases and one outlier."""
        self.data: npt.NDArray[np.float64] = np.array(
            [[1, 2, 3], [2, 3, 4], [3, 4, 5], [10, 10, 10]]
        )

    def test_basic_functionality(self) -> None:
        """Test basic Mahalanobis distance calculation."""
        distances: npt.NDArray[np.float64] = mahad(self.data)
        self.assertEqual(len(distances), 4)
        self.assertTrue((distances >= 0).all())

    def test_with_list_input(self) -> None:
        """Test MAHAD function accepts list input."""
        data: list[list[int]] = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [10, 10, 10]]
        distances: npt.NDArray[np.float64] = mahad(data)
        self.assertEqual(len(distances), 4)
        self.assertTrue((distances >= 0).all())

    def test_with_na_rm(self) -> None:
        """Test MAHAD handles missing values correctly when na_rm=True."""
        self.data_with_nan: npt.NDArray[np.float64] = np.array(
            [[1, 2, 3], [2, np.nan, 4], [5, 6, 7], [10, 10, 10]], dtype=float
        )
        distances: npt.NDArray[np.float64] = mahad(self.data_with_nan, na_rm=True)
        self.assertTrue(np.all(np.isnan(distances) | (distances >= 0)))

    def test_without_na_rm_raises_error(self) -> None:
        """Test MAHAD raises ValueError when data contains NaN and na_rm=False."""
        self.data_with_nan: npt.NDArray[np.float64] = np.array(
            [[1, 2, 3], [2, np.nan, 4], [5, 6, 7], [10, 10, 10]]
        )
        with self.assertRaises(ValueError):
            mahad(self.data_with_nan, na_rm=False)

    def test_flagging(self) -> None:
        """Test outlier flagging returns boolean array."""
        _: npt.NDArray[np.float64]
        flags: npt.NDArray[np.bool_]
        _, flags = mahad(self.data, flag=True)
        self.assertEqual(len(flags), 4)
        self.assertTrue(isinstance(flags[0], np.bool_))

    def test_flagging_with_different_methods(self) -> None:
        """Test outlier flagging works with IQR and z-score methods."""
        _: npt.NDArray[np.float64]
        flags_iqr: npt.NDArray[np.bool_]
        _, flags_iqr = mahad(self.data, flag=True, method="iqr")
        self.assertEqual(len(flags_iqr), 4)

        flags_zscore: npt.NDArray[np.bool_]
        _, flags_zscore = mahad(self.data, flag=True, method="zscore")
        self.assertEqual(len(flags_zscore), 4)

    def test_flagging_with_threshold(self) -> None:
        """Test outlier flagging respects confidence threshold."""
        distances: npt.NDArray[np.float64]
        flags: npt.NDArray[np.bool_]
        distances, flags = mahad(self.data, flag=True, confidence=0.99)
        threshold: float = stats.chi2.ppf(0.99, df=self.data.shape[1])
        flagged_distances: npt.NDArray[np.float64] = distances[flags]
        self.assertTrue((flagged_distances**2 > threshold).all())

    def test_no_negative_distances(self) -> None:
        """Test Mahalanobis distances are never negative."""
        distances: npt.NDArray[np.float64] = mahad(self.data)
        self.assertTrue((distances >= 0).all())

    def test_invalid_confidence(self) -> None:
        """Test MAHAD raises ValueError for invalid confidence values."""
        with self.assertRaises(ValueError):
            mahad(self.data, flag=True, confidence=1.1)
        with self.assertRaises(ValueError):
            mahad(self.data, flag=True, confidence=-0.1)

    def test_invalid_method(self) -> None:
        """Test MAHAD raises ValueError for invalid flagging method."""
        with self.assertRaises(ValueError):
            mahad(self.data, method="invalid")

    def test_validation(self) -> None:
        """Test MAHAD raises appropriate errors for invalid inputs."""
        with self.assertRaises(ValueError):
            mahad(None)
        with self.assertRaises(ValueError):
            mahad([])
        with self.assertRaises(ValueError):
            mahad([[1, 2]])

    def test_mahad_summary(self) -> None:
        """Test MAHAD summary returns expected statistics dictionary."""
        summary: dict[str, Any] = mahad_summary(self.data)
        self.assertIn("mean", summary)
        self.assertIn("std", summary)
        self.assertIn("outliers", summary)
        self.assertIn("total", summary)


class TestEvenOddFunction(unittest.TestCase):
    """Test suite for even-odd consistency scoring function.

    Tests basic functionality, handling of missing data, diagnostic output,
    and validation of factor specifications.
    """

    def test_basic_functionality(self) -> None:
        """Test basic even-odd consistency scoring."""
        data: npt.NDArray[np.float64] = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
        factors: list[int] = [3, 3]
        scores: npt.NDArray[np.float64] = evenodd(data, factors)
        self.assertEqual(len(scores), 2)

    def test_with_list_input(self) -> None:
        """Test even-odd function accepts list input."""
        data: list[list[int]] = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
        factors: list[int] = [3, 3]
        scores: npt.NDArray[np.float64] = evenodd(data, factors)
        self.assertEqual(len(scores), 2)

    def test_with_missing_data(self) -> None:
        """Test even-odd handles missing values in input data."""
        data: npt.NDArray[np.float64] = np.array([[1, np.nan, 3, 4, np.nan, 6], [2, 3, 4, 5, 6, 7]])
        factors: list[int] = [3, 3]
        scores: npt.NDArray[np.float64] = evenodd(data, factors)
        self.assertEqual(len(scores), 2)

    def test_diag_output(self) -> None:
        """Test even-odd returns diagnostic values when diag=True."""
        data: npt.NDArray[np.float64] = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
        factors: list[int] = [3, 3]
        scores: npt.NDArray[np.float64]
        diag_vals: npt.NDArray[np.float64]
        scores, diag_vals = evenodd(data, factors, diag=True)
        self.assertEqual(len(scores), 2)
        self.assertEqual(len(diag_vals), 2)

    def test_varying_factors(self) -> None:
        """Test even-odd with different factor sizes."""
        data: npt.NDArray[np.float64] = np.array(
            [[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]]
        )
        factors: list[int] = [4, 4]
        scores: npt.NDArray[np.float64] = evenodd(data, factors)
        self.assertEqual(len(scores), 2)

    def test_single_item_factors(self) -> None:
        """Test even-odd with single-item factors."""
        data: npt.NDArray[np.float64] = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        factors: list[int] = [1, 2, 2]
        scores: npt.NDArray[np.float64] = evenodd(data, factors)
        self.assertEqual(len(scores), 2)

    def test_validation(self) -> None:
        """Test even-odd raises appropriate errors for invalid inputs."""
        with self.assertRaises(ValueError):
            evenodd([], [])
        with self.assertRaises(ValueError):
            evenodd([[1, 2, 3]], [])
        with self.assertRaises(ValueError):
            evenodd([[1, 2, 3]], [2, 2])
        with self.assertRaises(ValueError):
            evenodd([], [2, 2])


class TestPsychometricFunctions(unittest.TestCase):
    """Test suite for psychometric synonym/antonym detection functions.

    Tests psychsyn scoring, critical value calculation, psychant function,
    resampling for missing values, and summary statistics.
    """

    def setUp(self) -> None:
        """Initialize test data with simple progressive values."""
        self.data: npt.NDArray[np.float64] = np.array(
            [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
        )

    def test_psychsyn_basic(self) -> None:
        """Test basic psychometric synonym scoring."""
        scores: npt.NDArray[np.float64] = psychsyn(self.data)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychsyn_with_list_input(self) -> None:
        """Test psychsyn function accepts list input."""
        data: list[list[int]] = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
        scores: npt.NDArray[np.float64] = psychsyn(data)
        self.assertEqual(len(scores), len(data))

    def test_psychsyn_diag(self) -> None:
        """Test psychsyn returns diagnostic values when diag=True."""
        scores: npt.NDArray[np.float64]
        diag_vals: npt.NDArray[np.float64]
        scores, diag_vals = psychsyn(self.data, diag=True)
        self.assertEqual(len(scores), self.data.shape[0])
        self.assertEqual(len(diag_vals), self.data.shape[0])

    def test_psychsyn_resample_na(self) -> None:
        """Test psychsyn resamples correlations for missing values."""
        self.data = self.data.astype(float)
        self.data[0, 0] = np.nan
        scores: npt.NDArray[np.float64] = psychsyn(self.data, resample_na=True)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychsyn_with_random_seed(self) -> None:
        """Test psychsyn produces reproducible results with random seed."""
        scores1: npt.NDArray[np.float64] = psychsyn(self.data, resample_na=True, random_seed=42)
        scores2: npt.NDArray[np.float64] = psychsyn(self.data, resample_na=True, random_seed=42)
        np.testing.assert_array_equal(scores1, scores2)

    def test_psychsyn_critval(self) -> None:
        """Test critical value calculation returns tuples with indices and correlations."""
        results: list[tuple[int, int, float]] = psychsyn_critval(self.data)
        self.assertTrue(all(isinstance(t, tuple) and len(t) == 3 for t in results))

    def test_psychsyn_critval_with_min_correlation(self) -> None:
        """Test critical value filtering respects minimum correlation threshold."""
        results: list[tuple[int, int, float]] = psychsyn_critval(self.data, min_correlation=0.5)
        self.assertTrue(all(abs(t[2]) >= 0.5 for t in results))

    def test_psychsyn_anto(self) -> None:
        """Test psychsyn with antonym detection mode enabled."""
        scores: npt.NDArray[np.float64] = psychsyn(self.data, anto=True, critval=-0.6)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychant(self) -> None:
        """Test psychant convenience function for antonym detection."""
        scores: npt.NDArray[np.float64] = psychant(self.data)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychant_with_resample_na(self) -> None:
        """Test psychant handles missing value resampling."""
        scores: npt.NDArray[np.float64] = psychant(self.data, resample_na=True, random_seed=42)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychsyn_summary(self) -> None:
        """Test psychsyn summary returns expected statistics dictionary."""
        summary: dict[str, Any] = psychsyn_summary(self.data)
        self.assertIn("mean_score", summary)
        self.assertIn("std_score", summary)
        self.assertIn("item_pairs", summary)
        self.assertIn("total_individuals", summary)

    def test_psychsyn_validation(self) -> None:
        """Test psychsyn raises appropriate errors for invalid inputs."""
        with self.assertRaises(ValueError):
            psychsyn(None)
        with self.assertRaises(ValueError):
            psychsyn([])
        with self.assertRaises(ValueError):
            psychsyn([[1]])
        with self.assertRaises(ValueError):
            psychsyn([[1, 2], [3, 4]], critval=0.5, anto=True)

    def test_psychsyn_edge_cases(self) -> None:
        """Test psychsyn handles edge cases like high thresholds and constant data."""
        data: npt.NDArray[np.float64] = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 4]])
        scores: npt.NDArray[np.float64] = psychsyn(data, critval=0.99)
        self.assertTrue(np.all(np.isnan(scores)) or np.all(scores == 0))

        data = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
        scores = psychsyn(data, critval=0.5)
        self.assertEqual(len(scores), 3)


if __name__ == "__main__":
    unittest.main()
