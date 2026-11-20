"""Run one resident OpenPI model for trusted local simulation requests."""

from multiprocessing.connection import Listener
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import jax
import numpy as np
from policy import Pi05LiberoPlusPolicy


def main():
    scored = len(sys.argv)>3 and sys.argv[3] == "scored"
    if scored:
        from pi05_scored_candidate_policy import Pi05ScoredCandidatePolicy
        policy = Pi05ScoredCandidatePolicy(log_path=Path(sys.argv[1]).with_name("selection.jsonl"), seed=int(sys.argv[4]))
    else:
        policy = Pi05LiberoPlusPolicy()
    policy.load(SimpleNamespace(checkpoint_path=sys.argv[2]))
    cold = True
    shared_sampler = None
    counts = {}
    with Listener(sys.argv[1], family="AF_UNIX") as listener:
        with listener.accept() as connection:
            while True:
                try:
                    request = connection.recv()
                except EOFError:
                    break
                steps = request.get("num_steps", 10)
                mode = request.get("mode", "single")
                if mode == "scored":
                    if not scored:
                        raise ValueError("Scored requests require the scored Policy process")
                    action = policy.act(SimpleNamespace(features=request["features"],step=request["step"],episode_id=request["episode_id"]))
                    record = policy.last_record
                    counts[mode] = counts.get(mode,0)+1
                    connection.send({"actions":action.actions,"policy_ms":record["policy_ms"],
                                     "warmup":counts[mode]<=3 or record["cold_generation"] or record["cold_fallback"],
                                     "selection":record})
                    continue
                if scored:
                    raise ValueError("Scored-only process received an unrelated mode")
                if steps not in (4, 10):
                    raise ValueError("Only four-step proposals and ten-step continuation are supported")
                policy.policy._sample_kwargs = {"num_steps": steps}
                noise = np.random.default_rng(request["seed"]).standard_normal(policy.noise_shape, dtype=np.float32)
                jax.effects_barrier()
                started = time.monotonic()
                model_input = policy.model_input(SimpleNamespace(features=request["features"]))
                if mode == "shared":
                    import jax.numpy as jnp
                    from openpi.models.model import Observation
                    from shared_prefix_sampler import create_shared_prefix_sampler
                    if shared_sampler is None:
                        shared_sampler = create_shared_prefix_sampler(policy.policy._model)
                    inputs = policy.policy._input_transform(jax.tree.map(lambda x: x, model_input))
                    inputs = jax.tree.map(lambda x: jnp.asarray(x)[None,...],inputs)
                    second_noise = np.random.default_rng(request["second_seed"]).standard_normal(policy.noise_shape,dtype=np.float32)
                    outputs = shared_sampler(Observation.from_dict(inputs), jnp.asarray(np.stack([noise,second_noise])), num_steps=4)
                    jax.block_until_ready(outputs)
                    prefixes = []
                    for output in np.asarray(outputs):
                        data = policy.policy._output_transform({"state":np.asarray(inputs["state"][0]),"actions":output})
                        action = np.asarray(data["actions"],dtype=np.float32)
                        if action.shape != (10,7) or not np.isfinite(action).all():
                            raise ValueError("Invalid shared action")
                        prefixes.append(np.ascontiguousarray(action[:5]))
                    actions = prefixes[0]
                else:
                    actions = policy.action_prefix(model_input, noise=noise)
                # Official infer copies GPU outputs to NumPy before returning.
                jax.block_until_ready(actions)
                elapsed = (time.monotonic()-started)*1000
                key = (mode,steps)
                counts[key] = counts.get(key,0)+1
                connection.send({"actions": actions, "policy_ms": elapsed, "cold_call": cold,
                                 "warmup": counts[key] <= 3})
                cold = False


if __name__ == "__main__":
    main()
