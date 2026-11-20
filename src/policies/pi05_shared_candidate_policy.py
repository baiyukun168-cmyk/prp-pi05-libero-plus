"""Generate two four-step candidates using one shared vision-language prefix."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models.model import Observation
from roboscope_kit import run

from pi05_candidate_policy import Pi05CandidatePolicy
from shared_prefix_sampler import create_shared_prefix_sampler


class Pi05SharedCandidatePolicy(Pi05CandidatePolicy):
    def load(self, context):
        capabilities = super().load(context)
        self.shared_sampler = create_shared_prefix_sampler(self.policy._model)
        return capabilities

    def candidate_prefixes(self, model_input, noises):
        inputs = self.policy._input_transform(jax.tree.map(lambda value: value, model_input))
        inputs = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], inputs)
        outputs = self.shared_sampler(Observation.from_dict(inputs), jnp.asarray(np.stack(noises)), num_steps=4)
        jax.block_until_ready(outputs)
        outputs = np.asarray(outputs)
        prefixes = []
        for actions in outputs:
            data = self.policy._output_transform({"state": np.asarray(inputs["state"][0]), "actions": actions})
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise ValueError("Expected finite shared candidate actions of shape (10, 7)")
            prefixes.append(np.ascontiguousarray(actions[:self.action_chunk_size]))
        return prefixes


if __name__ == "__main__":
    run(Pi05SharedCandidatePolicy(log_path=Path(__file__).with_name("candidates.jsonl"), seed=int(sys.argv[1])))
