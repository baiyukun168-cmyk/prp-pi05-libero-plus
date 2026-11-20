import unittest

from src.evaluation.paired_branches import continuation_seed, summarize_pairs


class PairedBranchesTest(unittest.TestCase):
    def test_seed_plan_is_repeatable_and_changes_between_nodes_and_steps(self):
        self.assertEqual(continuation_seed(7, 15, 80), continuation_seed(7, 15, 80))
        self.assertEqual(len({continuation_seed(7, n, s) for n in (0, 15) for s in (80, 85)}), 4)

    def test_paired_success_bounds_and_incomplete_pair_rejection(self):
        records = [{"node": n, "candidate_idx": c, "success": y}
                   for n, ys in enumerate(((True, False), (False, False), (True, True)))
                   for c, y in enumerate(ys)]
        summary = summarize_pairs(records)
        self.assertEqual(summary["pairs"], 3)
        self.assertEqual(summary["random_success_rate"], .5)
        self.assertAlmostEqual(summary["oracle_success_rate"], 2/3)
        self.assertAlmostEqual(summary["discordance_rate"], 1/3)
        with self.assertRaises(ValueError):
            summarize_pairs(records[:-1])


if __name__ == "__main__":
    unittest.main()
