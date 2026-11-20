import unittest

from src.evaluation.camera_validation import summarize, configured_modes


class CameraValidationTest(unittest.TestCase):
    def test_scored_only_configuration_cannot_add_baseline(self):
        self.assertEqual(configured_modes({"modes":["scored"]}, 0), ("scored",))
        self.assertEqual(configured_modes({"modes":["scored"]}, 1), ("scored",))
        with self.assertRaises(ValueError):
            configured_modes({"modes":["unknown"]}, 0)

    def test_counts_trials_and_excludes_warmup_latency(self):
        records = [
            {"mode": "baseline", "success": True, "calls": [{"policy_ms": 1000, "warmup": True}, {"policy_ms": 80, "warmup": False}]},
            {"mode": "baseline", "success": False, "calls": [{"policy_ms": 100, "warmup": False}]},
            {"mode": "shared", "success": True, "calls": [{"policy_ms": 70, "warmup": False}]},
        ]
        result = summarize(records)
        self.assertEqual(result["baseline"]["success_rate"], .5)
        self.assertEqual(result["baseline"]["mean_policy_ms"], 90)
        self.assertEqual(result["baseline"]["latency_samples"], 2)
        self.assertEqual(result["shared"]["mean_policy_ms"], 70)


if __name__ == "__main__":
    unittest.main()
