"""Unit tests for new IER detection functions."""

import unittest

import numpy as np

from ier.composite import composite, composite_flag, composite_summary
from ier.guttman import guttman, guttman_flag
from ier.lz import lz, lz_flag
from ier.mad import mad, mad_flag
from ier.person_total import person_total
from ier.reliability import individual_reliability, individual_reliability_flag
from ier.response_time import (
    response_time,
    response_time_consistency,
    response_time_flag,
)
from ier.semantic import semantic_ant, semantic_syn
from ier.u3_poly import midpoint_responding, response_pattern, u3_poly


class TestPersonTotal(unittest.TestCase):
    """Tests for person-total correlation."""

    def test_basic_functionality(self) -> None:
        """Test basic person-total correlation."""
        data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]]
        result = person_total(data)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 1.0, places=5)
        self.assertAlmostEqual(result[2], 1.0, places=5)

    def test_reversed_pattern(self) -> None:
        """Test that reversed patterns get lower correlation."""
        data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]]
        result = person_total(data)
        self.assertLess(result[1], result[0])

    def test_with_nan(self) -> None:
        """Test handling of missing values."""
        data = [[1, 2, np.nan, 4, 5], [1, 2, 3, 4, 5]]
        result = person_total(data, na_rm=True)
        self.assertEqual(len(result), 2)
        self.assertFalse(np.isnan(result[0]))


class TestSemanticSyn(unittest.TestCase):
    """Tests for semantic synonym/antonym functions."""

    def test_basic_synonym(self) -> None:
        """Test basic semantic synonym detection."""
        data = [[1, 1, 5, 5], [1, 2, 5, 4], [3, 3, 3, 3]]
        pairs = [(0, 1), (2, 3)]
        result = semantic_syn(data, pairs)
        self.assertEqual(len(result), 3)

    def test_empty_pairs_raises(self) -> None:
        """Test that empty pairs raises ValueError."""
        data = [[1, 2, 3], [4, 5, 6]]
        with self.assertRaises(ValueError):
            semantic_syn(data, [])

    def test_invalid_indices_raises(self) -> None:
        """Test that invalid indices raise ValueError."""
        data = [[1, 2, 3], [4, 5, 6]]
        with self.assertRaises(ValueError):
            semantic_syn(data, [(0, 10)])

    def test_semantic_ant_wrapper(self) -> None:
        """Test semantic_ant is a proper wrapper."""
        data = [[1, 5, 2, 4], [1, 5, 1, 5]]
        pairs = [(0, 1), (2, 3)]
        result = semantic_ant(data, pairs)
        self.assertEqual(len(result), 2)


class TestGuttman(unittest.TestCase):
    """Tests for Guttman error functions."""

    def test_basic_functionality(self) -> None:
        """Test basic Guttman error calculation."""
        data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [3, 3, 3, 3, 3]]
        result = guttman(data)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(0 <= r <= 1 or np.isnan(r) for r in result))

    def test_normalized_range(self) -> None:
        """Test that normalized Guttman errors are in valid range."""
        data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [3, 3, 3, 3, 3]]
        result = guttman(data, normalize=True)
        for r in result:
            if not np.isnan(r):
                self.assertGreaterEqual(r, 0)
                self.assertLessEqual(r, 1)

    def test_flag_function(self) -> None:
        """Test Guttman flagging."""
        data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]
        flags = guttman_flag(data, threshold=0.3)
        self.assertEqual(len(flags), 2)
        self.assertTrue(
            np.issubdtype(flags.dtype, np.bool_),
            msg=f"Expected boolean dtype, got {flags.dtype}",
        )


class TestResponseTime(unittest.TestCase):
    """Tests for response time functions."""

    def test_basic_metrics(self) -> None:
        """Test basic response time metrics."""
        times = [[2.0, 3.0, 4.0], [1.0, 1.0, 1.0]]
        self.assertAlmostEqual(response_time(times, metric="mean")[0], 3.0)
        self.assertAlmostEqual(response_time(times, metric="median")[0], 3.0)
        self.assertAlmostEqual(response_time(times, metric="min")[0], 2.0)

    def test_invalid_metric_raises(self) -> None:
        """Test that invalid metric raises ValueError."""
        times = [[1.0, 2.0, 3.0]]
        with self.assertRaises(ValueError):
            response_time(times, metric="invalid")

    def test_flag_function(self) -> None:
        """Test response time flagging."""
        times = [[2.0, 3.0, 4.0], [0.1, 0.1, 0.1], [2.5, 2.5, 2.5]]
        flags = response_time_flag(times, threshold=0.5)
        self.assertTrue(flags[1])
        self.assertFalse(flags[0])

    def test_consistency(self) -> None:
        """Test response time consistency (CV)."""
        times = [[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]
        cv = response_time_consistency(times)
        self.assertGreater(cv[0], cv[1])


class TestIndividualReliability(unittest.TestCase):
    """Tests for individual reliability functions."""

    def test_basic_functionality(self) -> None:
        """Test basic individual reliability."""
        data = [[1, 2, 1, 2, 1, 2], [1, 5, 2, 4, 3, 3], [3, 3, 3, 3, 3, 3]]
        result = individual_reliability(data, n_splits=10, random_seed=42)
        self.assertEqual(len(result), 3)

    def test_too_few_items_raises(self) -> None:
        """Test that too few items raises ValueError."""
        data = [[1, 2, 3], [4, 5, 6]]
        with self.assertRaises(ValueError):
            individual_reliability(data)

    def test_flag_function(self) -> None:
        """Test individual reliability flagging."""
        data = [[1, 2, 1, 2, 1, 2], [1, 5, 2, 4, 3, 3]]
        flags = individual_reliability_flag(data, n_splits=10, random_seed=42)
        self.assertEqual(len(flags), 2)
        self.assertTrue(
            np.issubdtype(flags.dtype, np.bool_),
            msg=f"Expected boolean dtype, got {flags.dtype}",
        )


class TestU3Poly(unittest.TestCase):
    """Tests for U3 polytomous and response pattern functions."""

    def test_basic_u3(self) -> None:
        """Test basic U3 calculation."""
        data = [[1, 5, 1, 5, 1], [3, 3, 3, 3, 3], [1, 2, 3, 4, 5]]
        result = u3_poly(data, scale_min=1, scale_max=5)
        self.assertEqual(len(result), 3)
        self.assertGreater(result[0], result[1])

    def test_midpoint_responding(self) -> None:
        """Test midpoint responding calculation."""
        data = [[1, 5, 1, 5, 1], [3, 3, 3, 3, 3], [1, 2, 3, 4, 5]]
        result = midpoint_responding(data, scale_min=1, scale_max=5)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[1], 1.0)

    def test_response_pattern(self) -> None:
        """Test response pattern returns all indices."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
        result = response_pattern(data, scale_min=1, scale_max=5)
        self.assertIn("extreme", result)
        self.assertIn("midpoint", result)
        self.assertIn("acquiescence", result)
        self.assertIn("variability", result)


class TestComposite(unittest.TestCase):
    """Tests for composite IER index functions."""

    def test_basic_functionality(self) -> None:
        """Test basic composite calculation."""
        data = [
            [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
        ]
        result = composite(data)
        self.assertEqual(len(result), 3)

    def test_straightliner_highest_score(self) -> None:
        """Test that straightliners get highest composite score."""
        data = [
            [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
        ]
        result = composite(data)
        self.assertEqual(np.argmax(result), 1)

    def test_specific_indices(self) -> None:
        """Test composite with specific indices."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        result = composite(data, indices=["irv", "longstring"])
        self.assertEqual(len(result), 3)

    def test_sum_method(self) -> None:
        """Test composite with sum method."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
        result = composite(data, method="sum")
        self.assertEqual(len(result), 2)

    def test_max_method(self) -> None:
        """Test composite with max method."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
        result = composite(data, method="max")
        self.assertEqual(len(result), 2)

    def test_no_standardize(self) -> None:
        """Test composite without standardization."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
        result = composite(data, standardize=False)
        self.assertEqual(len(result), 2)

    def test_with_evenodd(self) -> None:
        """Test composite with evenodd index."""
        data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]
        result = composite(data, indices=["irv", "evenodd"], evenodd_factors=[5, 5])
        self.assertEqual(len(result), 2)

    def test_flag_function(self) -> None:
        """Test composite flagging."""
        data = [
            [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
        ]
        scores, flags = composite_flag(data, threshold=1.0)
        self.assertEqual(len(flags), 3)
        self.assertTrue(
            np.issubdtype(flags.dtype, np.bool_),
            msg=f"Expected boolean dtype, got {flags.dtype}",
        )
        self.assertTrue(flags[1])

    def test_flag_with_percentile(self) -> None:
        """Test composite flagging with percentile."""
        data = [
            [1, 2, 3, 4, 5],
            [3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1],
            [2, 3, 4, 5, 1],
        ]
        scores, flags = composite_flag(data, percentile=75.0)
        self.assertEqual(len(flags), 4)

    def test_summary_function(self) -> None:
        """Test composite summary."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        summary = composite_summary(data)
        self.assertIn("composite", summary)
        self.assertIn("indices", summary)
        self.assertIn("indices_used", summary)
        self.assertIn("mean", summary)
        self.assertIn("std", summary)

    def test_invalid_index_raises(self) -> None:
        """Test that invalid index raises ValueError."""
        data = [[1, 2, 3, 4, 5]]
        with self.assertRaises(ValueError):
            composite(data, indices=["invalid_index"])

    def test_evenodd_without_factors_raises(self) -> None:
        """Test that evenodd without factors raises ValueError."""
        data = [[1, 2, 3, 4, 5]]
        with self.assertRaises(ValueError):
            composite(data, indices=["evenodd"])

    def test_invalid_method_raises(self) -> None:
        """Test that invalid method raises ValueError."""
        data = [[1, 2, 3, 4, 5]]
        with self.assertRaises(ValueError):
            composite(data, method="invalid")  # type: ignore[arg-type]


class TestMAD(unittest.TestCase):
    """Tests for Mean Absolute Difference functions."""

    def test_basic_functionality(self) -> None:
        """Test basic MAD calculation."""
        data = [
            [5, 1, 5, 1],
            [5, 5, 5, 5],
            [3, 3, 3, 3],
        ]
        result = mad(data, positive_items=[0, 2], negative_items=[1, 3], scale_max=5)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 4.0)
        self.assertAlmostEqual(result[2], 0.0)

    def test_attentive_responder(self) -> None:
        """Test that attentive responders get low MAD."""
        data = [[5, 1, 4, 2], [4, 2, 5, 1]]
        result = mad(data, positive_items=[0, 2], negative_items=[1, 3], scale_max=5)
        for score in result:
            self.assertLess(score, 1.0)

    def test_careless_responder_high_mad(self) -> None:
        """Test that careless responders ignoring item direction get high MAD."""
        data = [[5, 5, 5, 5], [1, 1, 1, 1]]
        result = mad(data, positive_items=[0, 2], negative_items=[1, 3], scale_max=5)
        for score in result:
            self.assertGreater(score, 3.0)

    def test_item_pairs_input(self) -> None:
        """Test using item_pairs instead of positive/negative items."""
        data = [[5, 1, 4, 2], [3, 3, 3, 3]]
        pairs = [(0, 1), (2, 3)]
        result = mad(data, item_pairs=pairs, scale_max=5)
        self.assertEqual(len(result), 2)

    def test_flag_function(self) -> None:
        """Test MAD flagging."""
        data = [
            [5, 1, 5, 1],
            [5, 5, 5, 5],
            [3, 3, 3, 3],
        ]
        scores, flags = mad_flag(
            data, positive_items=[0, 2], negative_items=[1, 3], scale_max=5, threshold=3.0
        )
        self.assertEqual(len(flags), 3)
        self.assertTrue(
            np.issubdtype(flags.dtype, np.bool_),
            msg=f"Expected boolean dtype, got {flags.dtype}",
        )
        self.assertTrue(flags[1])
        self.assertFalse(flags[0])

    def test_with_nan(self) -> None:
        """Test handling of missing values."""
        data = [[5, 1, np.nan, 1], [5, 5, 5, 5]]
        result = mad(data, positive_items=[0, 2], negative_items=[1, 3], scale_max=5, na_rm=True)
        self.assertEqual(len(result), 2)
        self.assertFalse(np.isnan(result[0]))

    def test_invalid_no_items_raises(self) -> None:
        """Test that missing item specification raises ValueError."""
        data = [[1, 2, 3, 4]]
        with self.assertRaises(ValueError):
            mad(data, positive_items=[0, 1])

    def test_invalid_index_raises(self) -> None:
        """Test that out-of-bounds index raises ValueError."""
        data = [[1, 2, 3, 4]]
        with self.assertRaises(ValueError):
            mad(data, positive_items=[0, 10], negative_items=[1, 2], scale_max=5)

    def test_both_item_specs_raises(self) -> None:
        """Test that specifying both item_pairs and positive/negative raises."""
        data = [[1, 2, 3, 4]]
        with self.assertRaises(ValueError):
            mad(data, positive_items=[0], negative_items=[1], item_pairs=[(0, 1)], scale_max=5)

    def test_scale_max_inference(self) -> None:
        """Test that scale_max is inferred from data when not provided."""
        data = [[5, 1, 5, 1], [3, 3, 3, 3]]
        result = mad(data, positive_items=[0, 2], negative_items=[1, 3])
        self.assertEqual(len(result), 2)


class TestLz(unittest.TestCase):
    """Tests for standardized log-likelihood (lz) functions."""

    def test_basic_functionality(self) -> None:
        """Test basic lz calculation."""
        data = [
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ]
        result = lz(data)
        self.assertEqual(len(result), 3)

    def test_normal_pattern_positive_lz(self) -> None:
        """Test that normal response patterns get non-negative lz."""
        data = [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ]
        result = lz(data)
        for score in result:
            self.assertGreater(score, -2.0)

    def test_aberrant_pattern_negative_lz(self) -> None:
        """Test that aberrant patterns tend toward negative lz."""
        data = [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ]
        lz_scores = lz(data)
        self.assertGreater(lz_scores[0], lz_scores[3])

    def test_1pl_model(self) -> None:
        """Test lz with 1PL (Rasch) model."""
        data = [[1, 1, 0, 0], [1, 0, 1, 0]]
        result = lz(data, model="1pl")
        self.assertEqual(len(result), 2)

    def test_2pl_model(self) -> None:
        """Test lz with 2PL model (default)."""
        data = [[1, 1, 0, 0], [1, 0, 1, 0]]
        result = lz(data, model="2pl")
        self.assertEqual(len(result), 2)

    def test_custom_parameters(self) -> None:
        """Test lz with user-specified item parameters."""
        data = [[1, 1, 0, 0], [0, 1, 1, 0]]
        difficulty = [-1.0, -0.5, 0.5, 1.0]
        discrimination = [1.0, 1.0, 1.0, 1.0]
        result = lz(data, difficulty=difficulty, discrimination=discrimination)
        self.assertEqual(len(result), 2)

    def test_custom_theta(self) -> None:
        """Test lz with user-specified theta values."""
        data = [[1, 1, 0, 0], [0, 1, 1, 0]]
        theta = [0.0, 0.5]
        result = lz(data, theta=theta)
        self.assertEqual(len(result), 2)

    def test_flag_function(self) -> None:
        """Test lz flagging."""
        data = [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ]
        scores, flags = lz_flag(data, threshold=-1.5)
        self.assertEqual(len(flags), 2)
        self.assertTrue(
            np.issubdtype(flags.dtype, np.bool_),
            msg=f"Expected boolean dtype, got {flags.dtype}",
        )

    def test_flag_with_custom_threshold(self) -> None:
        """Test lz flagging with custom threshold."""
        data = [[1, 1, 0, 0], [1, 0, 1, 0]]
        scores, flags = lz_flag(data, threshold=0.0)
        self.assertEqual(len(flags), 2)

    def test_with_nan(self) -> None:
        """Test handling of missing values."""
        data = [[1, 1, np.nan, 0], [1, 0, 1, 0]]
        result = lz(data, na_rm=True)
        self.assertEqual(len(result), 2)

    def test_all_correct_responses(self) -> None:
        """Test handling of all correct responses."""
        data = [[1, 1, 1, 1], [0, 0, 0, 0]]
        result = lz(data)
        self.assertEqual(len(result), 2)
        self.assertFalse(np.isnan(result[0]))

    def test_polytomous_dichotomization(self) -> None:
        """Test that polytomous data is dichotomized."""
        data = [[5, 4, 3, 2, 1], [1, 2, 3, 4, 5]]
        result = lz(data)
        self.assertEqual(len(result), 2)

    def test_invalid_model_raises(self) -> None:
        """Test that invalid model raises ValueError."""
        data = [[1, 1, 0, 0]]
        with self.assertRaises(ValueError):
            lz(data, model="invalid")

    def test_mismatched_difficulty_raises(self) -> None:
        """Test that mismatched difficulty length raises ValueError."""
        data = [[1, 1, 0, 0]]
        with self.assertRaises(ValueError):
            lz(data, difficulty=[1.0, 2.0])

    def test_mismatched_discrimination_raises(self) -> None:
        """Test that mismatched discrimination length raises ValueError."""
        data = [[1, 1, 0, 0]]
        with self.assertRaises(ValueError):
            lz(data, discrimination=[1.0, 2.0])

    def test_mismatched_theta_raises(self) -> None:
        """Test that mismatched theta length raises ValueError."""
        data = [[1, 1, 0, 0], [0, 0, 1, 1]]
        with self.assertRaises(ValueError):
            lz(data, theta=[0.0])


if __name__ == "__main__":
    unittest.main()
