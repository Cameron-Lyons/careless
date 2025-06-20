import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
import scipy.stats as stats
from careless.longstring import (
    run_length_encode,
    run_length_decode,
    longstr_message,
    avgstr_message,
    longstring
)
from careless.irv import irv
from careless.mahad import mahad, mahad_summary
from careless.evenodd import evenodd
from careless.psychsyn import psychsyn, psychsyn_critval, psychant, psychsyn_summary


class TestLongstring(unittest.TestCase):
    def test_run_length_encode(self):
        self.assertEqual(
            run_length_encode("AAABBBCCDAA"),
            [("A", 3), ("B", 3), ("C", 2), ("D", 1), ("A", 2)],
        )
        self.assertEqual(run_length_encode("A"), [("A", 1)])
        self.assertEqual(run_length_encode(""), [])

    def test_run_length_encode_validation(self):
        with self.assertRaises(TypeError):
            run_length_encode(123)
        with self.assertRaises(TypeError):
            run_length_encode(None)

    def test_run_length_decode(self):
        self.assertEqual(
            run_length_decode([("A", 3), ("B", 3), ("C", 2), ("D", 1), ("A", 2)]),
            "AAABBBCCDAA",
        )
        self.assertEqual(run_length_decode([("A", 1)]), "A")
        self.assertEqual(run_length_decode([]), "")

    def test_run_length_decode_validation(self):
        with self.assertRaises(TypeError):
            run_length_decode("not a list")
        with self.assertRaises(TypeError):
            run_length_decode(None)

    def test_longstr_message(self):
        self.assertEqual(longstr_message("AAAABBBCCDAA"), ("A", 4))
        self.assertEqual(longstr_message("A"), ("A", 1))
        self.assertEqual(longstr_message(""), None)

    def test_longstr_message_validation(self):
        with self.assertRaises(TypeError):
            longstr_message(123)
        with self.assertRaises(TypeError):
            longstr_message(None)

    def test_avgstr_message(self):
        self.assertAlmostEqual(avgstr_message("AAABBBCCDAA"), 2.2)
        self.assertEqual(avgstr_message("A"), 1.0)
        self.assertEqual(avgstr_message(""), 0.0)

    def test_avgstr_message_validation(self):
        with self.assertRaises(TypeError):
            avgstr_message(123)
        with self.assertRaises(TypeError):
            avgstr_message(None)

    def test_longstring(self):
        self.assertEqual(longstring("AAAABBBCCDAA"), ("A", 4))
        self.assertEqual(
            longstring(["AAAABBBCCDAA", "A", ""]), [("A", 4), ("A", 1), None]
        )

        self.assertAlmostEqual(longstring("AAABBBCCDAA", avg=True), 2.2)
        self.assertAlmostEqual(
            longstring(["AAABBBCCDAA", "A", ""], avg=True), [2.2, 1.0, 0.0]
        )

    def test_longstring_numpy_array(self):
        data = np.array(["AAAABBBCCDAA", "A", ""])
        result = longstring(data)
        expected = [("A", 4), ("A", 1), None]
        self.assertEqual(result, expected)

    def test_longstring_validation(self):
        with self.assertRaises(ValueError):
            longstring(None)
        with self.assertRaises(ValueError):
            longstring([])
        with self.assertRaises(TypeError):
            longstring(123)
        with self.assertRaises(TypeError):
            longstring(["abc", 123, "def"])

    def test_longstring_edge_cases(self):
        self.assertEqual(longstring("a"), ("a", 1))
        
        self.assertEqual(longstring("abcdef"), ("a", 1))
        
        self.assertEqual(longstring("aaaa"), ("a", 4))


class TestIRV(unittest.TestCase):
    def test_basic_irv(self):
        x = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [1, 3, 5, 7]])
        result = irv(x)
        expected = np.array([1.11803399, 2.23606798, 2.23606798])
        np.testing.assert_almost_equal(result, expected)

    def test_irv_with_list_input(self):
        x = [[1, 2, 3, 4], [2, 4, 6, 8], [1, 3, 5, 7]]
        result = irv(x)
        expected = np.array([1.11803399, 2.23606798, 2.23606798])
        np.testing.assert_almost_equal(result, expected)

    def test_irv_with_na(self):
        x = np.array([[1, np.nan, 3, 4], [2, 4, 6, 8], [np.nan, 3, 5, 7]])
        result = irv(x, na_rm=True)
        expected = np.array([1.2472191, 2.236068, 1.6329932])
        np.testing.assert_almost_equal(result, expected)

    def test_irv_with_split(self):
        x = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [1, 3, 5, 7]])
        result = irv(x, split=True, num_split=2)
        expected = np.array([0.5, 1.0, 1.0])
        np.testing.assert_almost_equal(result, expected)

    def test_irv_with_custom_split_points(self):
        x = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [1, 3, 5, 7]])
        result = irv(x, split=True, split_points=[0, 2, 4])
        self.assertEqual(len(result), 3)

    def test_irv_with_split_and_na(self):
        x = np.array([[1, 2, np.nan, 4], [2, 4, 6, 8], [1, 3, np.nan, 7]])
        result = irv(x, na_rm=True, split=True, num_split=2)
        expected = np.array([0.25, 1.0, 0.5])
        np.testing.assert_almost_equal(result, expected)

    def test_irv_validation(self):
        with self.assertRaises(ValueError):
            irv(None)
        with self.assertRaises(ValueError):
            irv([])
        with self.assertRaises(ValueError):
            irv([[1, 2, 3]], split=True, split_points=[0, 5])  # Invalid split points
        with self.assertRaises(ValueError):
            irv([[1, 2, 3]], split=True, split_points=[0, 2, 1])

    def test_irv_edge_cases(self):
        x = np.array([[1], [2], [3]])
        result = irv(x)
        self.assertEqual(len(result), 3)
        
        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = irv(x, split=True, num_split=5)
        self.assertEqual(len(result), 2)


class TestMahadFunction(unittest.TestCase):
    def setUp(self):
        self.data = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [10, 10, 10]])

    def test_basic_functionality(self):
        distances = mahad(self.data)
        self.assertEqual(len(distances), 4)
        self.assertTrue((distances >= 0).all())

    def test_with_list_input(self):
        data = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [10, 10, 10]]
        distances = mahad(data)
        self.assertEqual(len(distances), 4)
        self.assertTrue((distances >= 0).all())

    def test_with_na_rm(self):
        self.data_with_nan = np.array(
            [[1, 2, 3], [2, np.nan, 4], [5, 6, 7], [10, 10, 10]], dtype=float
        )
        distances = mahad(self.data_with_nan, na_rm=True)
        self.assertTrue(np.all(np.isnan(distances) | (distances >= 0)))

    def test_without_na_rm_raises_error(self):
        self.data_with_nan = np.array(
            [[1, 2, 3], [2, np.nan, 4], [5, 6, 7], [10, 10, 10]]
        )
        with self.assertRaises(ValueError):
            mahad(self.data_with_nan, na_rm=False)

    def test_flagging(self):
        _, flags = mahad(self.data, flag=True)
        self.assertEqual(len(flags), 4)
        self.assertTrue(isinstance(flags[0], np.bool_))

    def test_flagging_with_different_methods(self):
        _, flags_iqr = mahad(self.data, flag=True, method="iqr")
        self.assertEqual(len(flags_iqr), 4)
        
        _, flags_zscore = mahad(self.data, flag=True, method="zscore")
        self.assertEqual(len(flags_zscore), 4)

    def test_flagging_with_threshold(self):
        distances, flags = mahad(self.data, flag=True, confidence=0.99)
        threshold = stats.chi2.ppf(0.99, df=self.data.shape[1])
        flagged_distances = distances[flags]
        self.assertTrue((flagged_distances**2 > threshold).all())

    def test_no_negative_distances(self):
        distances = mahad(self.data)
        self.assertTrue((distances >= 0).all())

    def test_invalid_confidence(self):
        with self.assertRaises(ValueError):
            mahad(self.data, flag=True, confidence=1.1)
        with self.assertRaises(ValueError):
            mahad(self.data, flag=True, confidence=-0.1)

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            mahad(self.data, method="invalid")

    def test_validation(self):
        with self.assertRaises(ValueError):
            mahad(None)
        with self.assertRaises(ValueError):
            mahad([])
        with self.assertRaises(ValueError):
            mahad([[1, 2]])

    def test_mahad_summary(self):
        summary = mahad_summary(self.data)
        self.assertIn('mean', summary)
        self.assertIn('std', summary)
        self.assertIn('outliers', summary)
        self.assertIn('total', summary)


class TestEvenOddFunction(unittest.TestCase):
    def test_basic_functionality(self):
        data = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
        factors = [3, 3]
        scores = evenodd(data, factors)
        self.assertEqual(len(scores), 2)

    def test_with_list_input(self):
        data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
        factors = [3, 3]
        scores = evenodd(data, factors)
        self.assertEqual(len(scores), 2)

    def test_with_missing_data(self):
        data = np.array([[1, np.nan, 3, 4, np.nan, 6], [2, 3, 4, 5, 6, 7]])
        factors = [3, 3]
        scores = evenodd(data, factors)
        self.assertEqual(len(scores), 2)

    def test_diag_output(self):
        data = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
        factors = [3, 3]
        scores, diag_vals = evenodd(data, factors, diag=True)
        self.assertEqual(len(scores), 2)
        self.assertEqual(len(diag_vals), 2)

    def test_varying_factors(self):
        data = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]])
        factors = [4, 4]
        scores = evenodd(data, factors)
        self.assertEqual(len(scores), 2)

    def test_single_item_factors(self):
        data = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        factors = [1, 2, 2]
        scores = evenodd(data, factors)
        self.assertEqual(len(scores), 2)

    def test_validation(self):
        with self.assertRaises(ValueError):
            evenodd([], [])
        with self.assertRaises(ValueError):
            evenodd([[1, 2, 3]], [])
        with self.assertRaises(ValueError):
            evenodd([[1, 2, 3]], [2, 2])
        with self.assertRaises(ValueError):
            evenodd([], [2, 2])


class TestPsychometricFunctions(unittest.TestCase):
    def setUp(self):
        self.data = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])

    def test_psychsyn_basic(self):
        scores = psychsyn(self.data)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychsyn_with_list_input(self):
        data = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
        scores = psychsyn(data)
        self.assertEqual(len(scores), len(data))

    def test_psychsyn_diag(self):
        scores, diag_vals = psychsyn(self.data, diag=True)
        self.assertEqual(len(scores), self.data.shape[0])
        self.assertEqual(len(diag_vals), self.data.shape[0])

    def test_psychsyn_resample_na(self):
        self.data = self.data.astype(float)
        self.data[0, 0] = np.nan
        scores = psychsyn(self.data, resample_na=True)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychsyn_with_random_seed(self):
        scores1 = psychsyn(self.data, resample_na=True, random_seed=42)
        scores2 = psychsyn(self.data, resample_na=True, random_seed=42)
        np.testing.assert_array_equal(scores1, scores2)

    def test_psychsyn_critval(self):
        results = psychsyn_critval(self.data)
        self.assertTrue(all(isinstance(t, tuple) and len(t) == 3 for t in results))

    def test_psychsyn_critval_with_min_correlation(self):
        results = psychsyn_critval(self.data, min_correlation=0.5)
        self.assertTrue(all(abs(t[2]) >= 0.5 for t in results))

    def test_psychsyn_anto(self):
        scores = psychsyn(self.data, anto=True, critval=-0.6)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychant(self):
        scores = psychant(self.data)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychant_with_resample_na(self):
        scores = psychant(self.data, resample_na=True, random_seed=42)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychsyn_summary(self):
        summary = psychsyn_summary(self.data)
        self.assertIn('mean_score', summary)
        self.assertIn('std_score', summary)
        self.assertIn('item_pairs', summary)
        self.assertIn('total_individuals', summary)

    def test_psychsyn_validation(self):
        with self.assertRaises(ValueError):
            psychsyn(None)
        with self.assertRaises(ValueError):
            psychsyn([])
        with self.assertRaises(ValueError):
            psychsyn([[1]])
        with self.assertRaises(ValueError):
            psychsyn([[1, 2], [3, 4]], critval=0.5, anto=True)

    def test_psychsyn_edge_cases(self):
        data = np.array([
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [1, 3, 2, 4]
        ])
        scores = psychsyn(data, critval=0.99)
        self.assertTrue(np.all(np.isnan(scores)) or np.all(scores == 0))
        
        data = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
        scores = psychsyn(data, critval=0.5)
        self.assertEqual(len(scores), 3)


if __name__ == "__main__":
    unittest.main()
