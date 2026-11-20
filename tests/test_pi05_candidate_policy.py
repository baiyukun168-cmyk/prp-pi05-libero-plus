import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class CandidatePolicyTest(unittest.TestCase):
    def setUp(self):
        launcher = (ROOT / "scripts/run_pi05_libero_plus_smoke.sh").read_text(encoding="utf-8")
        base_source = launcher.split('cat > "$session/policy.py" <<\'PY\'\n')[1].split("\nPY\n", 1)[0]
        self.load_policy = Mock()
        image_tools = SimpleNamespace(
            convert_to_uint8=lambda image: image,
            resize_with_pad=lambda image, height, width: image,
        )
        kit = SimpleNamespace(
            ActionChunk=lambda actions: actions,
            BenchmarkName=SimpleNamespace(LIBERO_PLUS=SimpleNamespace(value="libero_plus")),
            Policy=object,
            PolicyCapabilities=lambda **kwargs: kwargs,
            run=Mock(),
        )
        modules = {
            "jax": SimpleNamespace(devices=lambda _: ["fake GPU"]),
            "openpi": ModuleType("openpi"),
            "openpi.policies": SimpleNamespace(
                policy_config=SimpleNamespace(create_trained_policy=self.load_policy)
            ),
            "openpi.training": SimpleNamespace(
                config=SimpleNamespace(get_config=lambda _: SimpleNamespace(
                    model=SimpleNamespace(action_horizon=10, action_dim=32)
                ))
            ),
            "openpi_client": SimpleNamespace(image_tools=image_tools),
            "roboscope_kit": kit,
            "policy": ModuleType("policy"),
        }
        spec = importlib.util.spec_from_file_location(
            "candidate", ROOT / "src/policies/pi05_candidate_policy.py"
        )
        with patch.dict(sys.modules, modules):
            exec(compile(base_source, "generated policy.py", "exec"), modules["policy"].__dict__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.log_path = Path(directory.name) / "candidates.jsonl"
        self.base_policy_class = modules["policy"].Pi05LiberoPlusPolicy
        self.policy = module.Pi05CandidatePolicy(log_path=self.log_path, seed=7)
        self.policy.load(SimpleNamespace(checkpoint_path="/checkpoint"))
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        self.observation = SimpleNamespace(
            episode_id="task-0000-episode-0000",
            step=0,
            features={
                "images.primary": frame,
                "images.wrist": frame,
                "state.proprioception": [0, 0, 0, 0, 0, 0, 1, 0, 0],
                "language.instruction": "test instruction",
            },
        )

    def test_loads_once_and_logs_both_candidates_while_executing_first(self):
        first = np.ones((10, 7), dtype=np.float32)
        second = np.full((10, 7), 2, dtype=np.float32)
        infer = self.load_policy.return_value.infer
        infer.side_effect = [{"actions": first}, {"actions": second}] * 3

        np.testing.assert_array_equal(self.policy.act(self.observation), first[:5])
        self.policy.act(self.observation)
        self.observation.step = 5
        self.policy.act(self.observation)

        self.load_policy.assert_called_once()
        self.assertEqual(self.load_policy.call_args.kwargs["sample_kwargs"], {"num_steps": 4})
        noises = [call.kwargs["noise"] for call in infer.call_args_list]
        self.assertEqual(noises[0].shape, (10, 32))
        self.assertEqual(noises[0].dtype, np.float32)
        np.testing.assert_array_equal(noises[0], noises[2])
        self.assertFalse(np.array_equal(noises[0], noises[1]))
        self.assertFalse(np.array_equal(noises[0], noises[4]))
        records = [json.loads(line) for line in self.log_path.read_text().splitlines()]
        self.assertEqual(len(records), 3)
        self.assertEqual(records[2]["replan_idx"], 1)
        self.assertEqual(records[0]["selected_candidate_idx"], 0)
        np.testing.assert_array_equal(records[0]["candidates"][1]["action_prefix"], second[:5])

    def test_rejects_invalid_second_candidate_before_returning_actions(self):
        valid = np.ones((10, 7), dtype=np.float32)
        invalid = valid.copy()
        invalid[0, 0] = np.nan
        self.load_policy.return_value.infer.side_effect = [{"actions": valid}, {"actions": invalid}]

        with self.assertRaisesRegex(ValueError, "finite"):
            self.policy.act(self.observation)
        self.assertFalse(self.log_path.exists())

    def test_base_policy_uses_default_sampler_and_returns_one_prefix(self):
        self.load_policy.reset_mock()
        policy = self.base_policy_class()
        policy.load(SimpleNamespace(checkpoint_path="/checkpoint"))
        actions = np.arange(70, dtype=np.float32).reshape(10, 7)
        infer = self.load_policy.return_value.infer
        infer.return_value = {"actions": actions}

        np.testing.assert_array_equal(policy.act(self.observation), actions[:5])
        self.assertIsNone(self.load_policy.call_args.kwargs["sample_kwargs"])
        infer.assert_called_once()
        self.assertIsNone(infer.call_args.kwargs["noise"])


if __name__ == "__main__":
    unittest.main()
