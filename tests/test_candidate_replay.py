import unittest

import numpy as np

from src.evaluation.candidate_replay import compare_replays, replay_actions, selected_actions


class ReplayTest(unittest.TestCase):
    def test_distinguishes_intermediate_rendering_from_policy_input_changes(self):
        import copy
        first = {"steps": 6, "success": True, "terminated": True,
                 "trace": [{"sim_state": "state", "camera": "image"} for _ in range(7)]}
        second = copy.deepcopy(first)
        second["trace"][2]["camera"] = "different image"
        result = compare_replays(first, second)
        self.assertTrue(result["policy_replay_equal"])
        self.assertFalse(result["strict_equal"])
        second["trace"][5]["camera"] = "different input"
        self.assertFalse(compare_replays(first, second)["policy_replay_equal"])
        second = copy.deepcopy(first)
        second["trace"][2]["sim_state"] = "different physics"
        self.assertFalse(compare_replays(first, second)["policy_replay_equal"])

    def test_trims_final_chunk_and_rejects_missing_replans(self):
        rows = [{"step": i * 5, "replan_idx": i, "selected_candidate_idx": 0,
                 "candidates": [{"candidate_idx": 0, "action_prefix": [[i] * 7] * 5}]}
                for i in range(2)]
        actions = selected_actions(rows, total_steps=7)
        np.testing.assert_array_equal(actions[:, 0], [0, 0, 0, 0, 0, 1, 1])
        rows[1]["step"] = 10
        with self.assertRaises(ValueError):
            selected_actions(rows, total_steps=7)

    def test_replays_settle_and_stops_at_terminal(self):
        class Environment:
            def __init__(self):
                self.env = self
                self.done = False
                self.position = 0

            def reset(self):
                self.position = 0

            def set_init_state(self, state):
                self.position = state
                return {"position": np.array([state])}

            def get_sim_state(self):
                return np.array([self.position])

            def step(self, action):
                self.position += action[0]
                return {"position": np.array([self.position])}, 0, self.position >= 2, {}

        actions = np.ones((4, 7), dtype=np.float32)
        first = replay_actions(Environment(), 0, actions, settle_steps=2)
        second = replay_actions(Environment(), 0, actions, settle_steps=2)
        self.assertEqual(first, second)
        self.assertEqual(first["steps"], 2)
        self.assertTrue(first["success"])
        self.assertEqual(len(first["trace"]), 3)
        changed = replay_actions(Environment(), 1, actions, settle_steps=2)
        self.assertNotEqual(first["trace"][0], changed["trace"][0])


if __name__ == "__main__":
    unittest.main()
