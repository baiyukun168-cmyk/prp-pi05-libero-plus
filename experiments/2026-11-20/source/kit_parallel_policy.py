"""Unchanged OpenPI model behind Kit's lifecycle; episode RNG is isolated."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

import jax
import numpy as np
from openpi.policies import policy_config
from openpi.training import config
from openpi_client import image_tools
from roboscope_kit import ActionChunk, BenchmarkName, Policy, PolicyCapabilities, run


class OriginalPi05(Policy):
    def load(self, context):
        self.keys, self.stats = {}, {}
        self.log = (Path(os.environ["PARALLEL_TRIAL_DIR"]) / "policy-timing.jsonl").open("a", buffering=1)
        start = time.perf_counter()
        self.model = policy_config.create_trained_policy(config.get_config("pi05_libero"), context.checkpoint_path)
        load_s = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(2):
            self.model.infer({"observation/image": np.zeros((224,224,3), np.uint8),
                              "observation/wrist_image": np.zeros((224,224,3), np.uint8),
                              "observation/state": np.zeros(8, np.float32), "prompt": "warmup"})
        self.log.write(json.dumps({"type": "load", "load_s": load_s,
                                   "warmup_s": time.perf_counter() - start}) + "\n")
        b = BenchmarkName.LIBERO.value
        return PolicyCapabilities(supported_benchmarks=(b,),
            required_features=("images.primary", "images.wrist", "state.proprioception", "language.instruction"),
            supported_suites={b: ("libero_10",)}, supported_action_chunk_sizes=(5,), max_batch_size=1)

    def reset(self, context):
        # Independent official default RNG stream for each episode, invariant to interleaving.
        self.keys[context.episode_id] = jax.random.key(0)
        self.stats[context.episode_id] = {"hash": hashlib.sha256(), "calls": 0, "ms": []}

    def act(self, observation):
        started = time.perf_counter()
        f = observation.features
        p = np.asarray(f["state.proprioception"], dtype=np.float32)
        quat = p[3:7].copy()
        quat[3] = np.clip(quat[3], -1., 1.)
        den = np.sqrt(1. - quat[3] * quat[3])
        rotation = np.zeros(3) if math.isclose(den, 0.) else quat[:3] * 2. * math.acos(quat[3]) / den
        element = {"observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(f["images.primary"],224,224)),
                   "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(f["images.wrist"],224,224)),
                   "observation/state": np.concatenate((p[:3], rotation, p[7:9])),
                   "prompt": f["language.instruction"]}
        prepared = time.perf_counter()
        self.model._rng = self.keys[observation.episode_id]
        actions = np.asarray(self.model.infer(element)["actions"], dtype=np.float32)
        self.keys[observation.episode_id] = self.model._rng
        finished = time.perf_counter()
        if actions.shape != (10,7) or not np.isfinite(actions).all():
            raise ValueError("Invalid original pi05 actions")
        actions = np.ascontiguousarray(actions[:5])
        stat = self.stats[observation.episode_id]
        stat["hash"].update(actions.tobytes())
        stat["calls"] += 1
        stat["ms"].append((finished-started)*1000.)
        self.log.write(json.dumps({"type":"infer", "episode_id": observation.episode_id,
            "step": observation.step, "preprocess_ms": (prepared-started)*1000.,
            "infer_ms": (finished-prepared)*1000., "policy_ms": (finished-started)*1000.}) + "\n")
        return ActionChunk(actions)

    def end_episode(self, episode_id):
        self.keys.pop(episode_id, None)
        stat = self.stats.pop(episode_id, None)
        if stat is not None:
            self.log.write(json.dumps({"type":"episode_end", "episode_id":episode_id,
                "calls":stat["calls"], "action_sha256":stat["hash"].hexdigest()}) + "\n")


if __name__ == "__main__":
    run(OriginalPi05())
