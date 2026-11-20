import unittest
from src.policies.candidate_selection import select_candidate


class SelectionTest(unittest.TestCase):
    def test_selects_lower_residual_including_b(self):
        result = select_candidate([2., 1.], min_gap=.05)
        self.assertEqual(result["selected_candidate_idx"], 1)
        self.assertFalse(result["fallback"])

    def test_ties_and_small_gaps_fallback(self):
        self.assertTrue(select_candidate([1., 1.], min_gap=.05)["fallback"])
        self.assertTrue(select_candidate([1., 1.01], min_gap=.05)["fallback"])
        self.assertTrue(select_candidate([2., 1.], min_gap=1.)["fallback"])

    def test_nonfinite_and_absolute_residual_limit(self):
        self.assertTrue(select_candidate([float("nan"), 1.], min_gap=.05)["fallback"])
        self.assertTrue(select_candidate([20., 30.], min_gap=.05, max_residual=10.)["fallback"])
        with self.assertRaises(ValueError):
            select_candidate([1., 2.], min_gap=-1.)


if __name__ == "__main__":
    unittest.main()
