import unittest
import numpy as np
import scipy.stats as stats
from careless.longstring import (
    run_length_encode,
    run_length_decode,
    longstr_message,
    avgstr_message,
    longstring,
)
from careless.irv import irv
from careless.mahad import mahad
from careless.evenodd import evenodd
from careless.psychsyn import psychsyn, psychsyn_critval, psychant


class TestLongstring(unittest.TestCase):
    def test_run_length_encode(self):
        self.assertEqual(
            run_length_encode("AAABBBCCDAA"),
            [("A", 3), ("B", 3), ("C", 2), ("D", 1), ("A", 2)],
        )
        self.assertEqual(run_length_encode("A"), [("A", 1)])
        self.assertEqual(run_length_encode(""), [])

    def test_run_length_decode(self):
        self.assertEqual(
            run_length_decode([("A", 3), ("B", 3), ("C", 2), ("D", 1), ("A", 2)]),
            "AAABBBCCDAA",
        )
        self.assertEqual(run_length_decode([("A", 1)]), "A")
        self.assertEqual(run_length_decode([]), "")

    def test_longstr_message(self):
        self.assertEqual(longstr_message("AAAABBBCCDAA"), ("A", 4))
        self.assertEqual(longstr_message("A"), ("A", 1))
        self.assertEqual(longstr_message(""), None)

    def test_avgstr_message(self):
        self.assertAlmostEqual(avgstr_message("AAABBBCCDAA"), 2.2)
        self.assertEqual(avgstr_message("A"), 1.0)
        self.assertEqual(avgstr_message(""), 0.0)

    def test_longstring(self):
        # Testing with avg=False
        self.assertEqual(longstring("AAAABBBCCDAA"), ("A", 4))
        self.assertEqual(
            longstring(["AAAABBBCCDAA", "A", ""]), [("A", 4), ("A", 1), None]
        )

        # Testing with avg=True
        self.assertAlmostEqual(longstring("AAABBBCCDAA", avg=True), 2.2)
        self.assertAlmostEqual(
            longstring(["AAABBBCCDAA", "A", ""], avg=True), [2.2, 1.0, 0.0]
        )


class TestIRV(unittest.TestCase):
    def test_basic_irv(self):
        x = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [1, 3, 5, 7]])
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

    def test_irv_with_split_and_na(self):
        x = np.array([[1, 2, np.nan, 4], [2, 4, 6, 8], [1, 3, np.nan, 7]])
        result = irv(x, na_rm=True, split=True, num_split=2)
        expected = np.array([0.25, 1.0, 0.5])
        np.testing.assert_almost_equal(result, expected)


class TestMahadFunction(unittest.TestCase):
    def setUp(self):
        self.data = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [10, 10, 10]])

    def test_basic_functionality(self):
        distances = mahad(self.data)
        self.assertEqual(len(distances), 4)
        self.assertTrue((distances >= 0).all())

    def test_with_na_rm(self):
        data_with_na = np.array(
            [[1, 2, 3], [np.nan, 3, 4], [3, np.nan, 5], [10, 10, 10]]
        )
        distances = mahad(data_with_na, na_rm=True)
        self.assertEqual(len(distances), 2)

    def test_flagging(self):
        _, flags = mahad(self.data, flag=True)
        self.assertEqual(len(flags), 4)
        self.assertTrue(isinstance(flags[0], np.bool_))

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


class TestEvenOddFunction(unittest.TestCase):
    def test_basic_functionality(self):
        data = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
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

    def test_empty_data(self):
        data = np.array([])
        factors = []
        scores = evenodd(data, factors)
        self.assertEqual(len(scores), 0)


class TestPsychometricFunctions(unittest.TestCase):
    def setUp(self):
        # A toy dataset
        self.data = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])

    def test_psychsyn_basic(self):
        scores = psychsyn(self.data)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychsyn_diag(self):
        scores, diag_vals = psychsyn(self.data, diag=True)
        self.assertEqual(len(scores), self.data.shape[0])
        self.assertEqual(len(diag_vals), self.data.shape[0])

    def test_psychsyn_resample_na(self):
        self.data[0, 0] = np.nan
        scores = psychsyn(self.data, resample_na=True)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychsyn_critval(self):
        results = psychsyn_critval(self.data)
        self.assertTrue(all(isinstance(t, tuple) and len(t) == 3 for t in results))

    def test_psychsyn_anto(self):
        scores = psychsyn(self.data, anto=True)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychant(self):
        scores = psychant(self.data)
        self.assertEqual(len(scores), self.data.shape[0])


if __name__ == "__main__":
    unittest.main()
