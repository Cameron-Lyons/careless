"""Unit tests for new careless detection functions."""

import unittest

import numpy as np

from careless.guttman import guttman, guttman_flag
from careless.person_total import person_total
from careless.reliability import individual_reliability, individual_reliability_flag
from careless.response_time import (
    response_time,
    response_time_consistency,
    response_time_flag,
)
from careless.semantic import semantic_ant, semantic_syn
from careless.u3_poly import midpoint_responding, response_pattern, u3_poly


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
        self.assertTrue(flags.dtype == bool)


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
        self.assertTrue(flags.dtype == bool)


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


if __name__ == "__main__":
    unittest.main()
