"""Time resident-model single and serial dual candidate inference on one observation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--include-batch", action="store_true")
    parser.add_argument("--include-shared-prefix", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("Sample count must be positive")
    sys.path.insert(0, str(args.source_run))
    from policy import Pi05LiberoPlusPolicy

    policy = Pi05LiberoPlusPolicy()
    policy.load(SimpleNamespace(checkpoint_path=str(args.checkpoint)))
    with np.load(args.observation) as data:
        features = {
            "images.primary": np.ascontiguousarray(data["agentview_image"][::-1, ::-1]),
            "images.wrist": np.ascontiguousarray(data["robot0_eye_in_hand_image"][::-1, ::-1]),
            "state.proprioception": np.concatenate([data[key] for key in (
                "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos")]),
            "language.instruction": args.observation.with_suffix(".instruction.txt").read_text(),
        }
    observation = SimpleNamespace(features=features)
    records = []
    # Two explicit noises are reused across all conditions and trials.
    noises = [np.random.default_rng(seed).standard_normal(policy.noise_shape, dtype=np.float32) for seed in (7, 8)]
    workloads = (("ten_step_single", 10, 1), ("four_step_single", 4, 1), ("four_step_serial_dual", 4, 2))
    batch_check = None
    shared_check = None
    if args.include_batch:
        workloads += (("four_step_batch_dual", 4, 2),)
    shared_sampler = None
    if args.include_shared_prefix:
        from shared_prefix_sampler import create_shared_prefix_sampler
        shared_sampler = create_shared_prefix_sampler(policy.policy._model)
        workloads += (("four_step_shared_prefix_dual",4,2),)

    def batch_prefixes(model_input, steps, *, shared=False):
        from openpi.models.model import Observation
        resident = policy.policy
        inputs = resident._input_transform(jax.tree.map(lambda x: x, model_input))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        if shared:
            outputs = shared_sampler(Observation.from_dict(inputs),jnp.asarray(np.stack(noises)),num_steps=steps)
        else:
            inputs = jax.tree.map(lambda x: jnp.repeat(x,2,axis=0),inputs)
            outputs = resident._sample_actions(jax.random.key(0), Observation.from_dict(inputs),
                                              noise=jnp.asarray(np.stack(noises)), num_steps=steps)
        jax.block_until_ready(outputs)
        outputs = np.asarray(outputs)
        prefixes = []
        for index in range(2):
            state_index = 0 if shared else index
            data = resident._output_transform({"state": np.asarray(inputs["state"][state_index]), "actions": outputs[index]})
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape != (10,7) or not np.isfinite(actions).all():
                raise ValueError("Invalid batch sampler output")
            prefixes.append(np.ascontiguousarray(actions[:5]))
        return prefixes

    if args.include_batch or args.include_shared_prefix:
        model_input = policy.model_input(observation)
        policy.policy._sample_kwargs = {"num_steps": 4}
        serial = [policy.action_prefix(model_input, noise=noise) for noise in noises]
        if args.include_batch:
            batched = batch_prefixes(model_input, 4)
            batch_check = {"max_abs_action_error": float(np.max(np.abs(np.asarray(serial)-np.asarray(batched)))),
                           "allclose_atol_0_01_rtol_0_01": bool(np.allclose(serial, batched, atol=.01, rtol=.01))}
        if args.include_shared_prefix:
            shared = batch_prefixes(model_input,4,shared=True)
            shared_check = {"max_abs_action_error": float(np.max(np.abs(np.asarray(serial)-np.asarray(shared)))),
                            "allclose_atol_0_01_rtol_0_01": bool(np.allclose(serial,shared,atol=.01,rtol=.01))}
    def measure(name, steps, count, trial, warmup):
        # Version-pinned adapter changes only sampling kwargs on the resident Policy.
        policy.policy._sample_kwargs = {"num_steps": steps}
        jax.effects_barrier()
        start = time.perf_counter()
        model_input = policy.model_input(observation)
        preprocess_ms = (time.perf_counter()-start)*1000
        if name == "four_step_batch_dual":
            outputs = batch_prefixes(model_input, steps)
        elif name == "four_step_shared_prefix_dual":
            outputs = batch_prefixes(model_input, steps, shared=True)
        else:
            outputs = [policy.action_prefix(model_input, noise=noise) for noise in noises[:count]]
        jax.block_until_ready(outputs)
        elapsed_ms = (time.perf_counter()-start)*1000
        return {"workload": name, "num_steps": steps, "candidates": count, "trial": trial,
                "warmup": warmup, "preprocess_ms": preprocess_ms, "policy_ms": elapsed_ms}

    for workload in workloads:
        for trial in range(3):
            records.append(measure(*workload, trial, True))
    # Rotate order to reduce bias from changing GPU temperature and load.
    for trial in range(args.samples):
        for offset in range(len(workloads)):
            records.append(measure(*workloads[(trial+offset)%len(workloads)], trial, False))
    summary = {}
    for name, _, _ in workloads:
        values = [row["policy_ms"] for row in records if row["workload"] == name and not row["warmup"]]
        summary[name] = {"n": len(values), "mean_ms": float(np.mean(values)),
                         "p50_ms": float(np.percentile(values,50)), "p95_ms": float(np.percentile(values,95))}
    reference = summary["ten_step_single"]["mean_ms"]
    for item in summary.values():
        item["speedup_vs_ten_step_single"] = reference/item["mean_ms"]
    device = jax.devices("gpu")[0]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    report = {"scope": "Single fixed observation, resident model, synchronized CPU-returning inference",
              "samples": records, "summary": summary, "jax_version": jax.__version__,
              "device": str(device), "batch_consistency": batch_check, "shared_prefix_consistency": shared_check,
              "allocator_stats_after_all_workloads": device.memory_stats(),
              "gpu_snapshot_after_workloads": gpu, "observation": str(args.observation)}
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
