import unittest
from ..src.longstring import (
    run_length_encode,
    run_length_decode,
    longstr_message,
    avgstr_message,
    longstring,
)


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


if __name__ == "__main__":
    unittest.main()
