"""Generate and record two candidates while executing the first candidate."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from roboscope_kit import ActionChunk, run

# The smoke launcher puts its base Policy beside this entry point.
from policy import Pi05LiberoPlusPolicy


class Pi05CandidatePolicy(Pi05LiberoPlusPolicy):
    sample_kwargs = {"num_steps": 4}

    def __init__(self, *, log_path: Path, seed: int):
        self.log_path = log_path
        self.seed = seed

    def candidate_prefixes(self, model_input, noises):
        return [self.action_prefix(model_input, noise=noise) for noise in noises]

    def act(self, observation):
        model_input = self.model_input(observation)
        replan_idx = observation.step // self.action_chunk_size
        episode_key = f"{self.seed}:{observation.episode_id}"
        episode_seed = int.from_bytes(hashlib.sha256(episode_key.encode()).digest()[:8], "big")
        noises = []
        candidates = []
        for candidate_idx in range(2):
            seed = np.random.SeedSequence([episode_seed, replan_idx, candidate_idx])
            noise_seed = int(seed.generate_state(1, dtype=np.uint64)[0])
            noise = np.random.default_rng(noise_seed).standard_normal(self.noise_shape, dtype=np.float32)
            noises.append(noise)
            candidates.append({
                "candidate_idx": candidate_idx,
                "noise_seed": noise_seed,
            })

        prefixes = self.candidate_prefixes(model_input, noises)
        for record, prefix in zip(candidates, prefixes, strict=True):
            record["action_prefix"] = prefix.tolist()

        record = {
            "episode_id": observation.episode_id,
            "episode_seed": episode_seed,
            "step": observation.step,
            "replan_idx": replan_idx,
            "num_steps": self.sample_kwargs["num_steps"],
            "selected_candidate_idx": 0,
            "candidates": candidates,
        }
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        return ActionChunk(prefixes[0])


if __name__ == "__main__":
    log_path = Path(__file__).with_name("candidates.jsonl")
    run(Pi05CandidatePolicy(log_path=log_path, seed=int(sys.argv[1])))
