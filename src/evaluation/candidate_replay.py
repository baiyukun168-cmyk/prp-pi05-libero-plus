"""Check deterministic reset-and-action replay before collecting paired branches."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np


def selected_actions(rows, *, total_steps):
    if total_steps < 1 or len(rows) != (total_steps + 4) // 5:
        raise ValueError("Candidate log length does not match executed steps")
    prefixes = []
    for index, row in enumerate(rows):
        if row["step"] != index * 5 or row["replan_idx"] != index:
            raise ValueError("Candidate log contains missing or unordered replans")
        candidates = {item["candidate_idx"]: item for item in row["candidates"]}
        prefix = np.asarray(candidates[row["selected_candidate_idx"]]["action_prefix"], dtype=np.float32)
        if prefix.shape != (5, 7) or not np.isfinite(prefix).all():
            raise ValueError("Expected finite (5, 7) action prefix")
        prefixes.append(prefix)
    return np.concatenate(prefixes)[:total_steps]


def fingerprint(env, observation):
    values = {"sim_state": env.get_sim_state(), **observation}
    hashes = {}
    for name, value in sorted(values.items()):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"Unsupported object observation: {name}")
        digest = hashlib.sha256()
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
        hashes[name] = digest.hexdigest()
    return hashes


def replay_actions(env, initial_state, actions, *, settle_steps):
    if len(actions) == 0:
        raise ValueError("Replay requires at least one action")
    env.reset()
    observation = env.set_init_state(initial_state)
    dummy = np.asarray([0.] * 6 + [-1.], dtype=np.float32)
    for _ in range(settle_steps):
        observation, _, success, _ = env.step(dummy)
        if success or env.env.done:
            raise RuntimeError("Episode terminated during settle")
    trace = [fingerprint(env, observation)]
    success = False
    terminated = False
    for index, action in enumerate(actions):
        observation, _, success, _ = env.step(action)
        trace.append(fingerprint(env, observation))
        terminated = bool(success or env.env.done)
        if terminated:
            break
    return {"steps": index + 1, "success": bool(success), "terminated": terminated, "trace": trace}


def compare_replays(first, second):
    differences = {}
    for index, (left, right) in enumerate(zip(first["trace"], second["trace"])):
        fields = sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key))
        if fields:
            differences[index] = fields
    same_outcome = all(first[key] == second[key] for key in ("steps", "success", "terminated"))
    same_length = len(first["trace"]) == len(second["trace"])
    states_equal = same_length and not any("sim_state" in fields for fields in differences.values())
    # Step i is the observation before action i. Only multiples of 5 invoke Policy.
    input_differences = [index for index in differences if index % 5 == 0 and index < first["steps"]]
    return {
        "strict_equal": first == second, "state_equal": states_equal,
        "policy_replay_equal": same_outcome and states_equal and not input_differences,
        "differing_fields": differences, "policy_input_differing_steps": input_differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.run_dir / "evaluation/result.json"
    report = json.loads(source.read_text())
    metadata = report["benchmark"]
    if report["result"]["episodes"] != 1 or len(metadata["task_ids"]) != 1:
        raise ValueError("Replay currently requires a single-episode run")
    rows = [json.loads(line) for line in (args.run_dir / "candidates.jsonl").read_text().splitlines()]
    task_id = metadata["task_ids"][0]
    episode_id = next(iter(report["record_success"]))
    episode_index = int(episode_id.rsplit("-", 1)[1])
    if any(row["episode_id"] != episode_id for row in rows):
        raise ValueError("Candidate episode IDs differ from result")
    actions = selected_actions(rows, total_steps=report["result"]["total_steps"])
    args.output_dir.mkdir(parents=True, exist_ok=False)

    from libero.libero import benchmark
    from roboscope_kit.runtime.benchmarks.libero.worker import create_task_runtime

    runs = []
    for attempt in range(2):
        random.seed(metadata["seed"])
        np.random.seed(metadata["seed"])
        suite = benchmark.get_benchmark_dict()[metadata["suite"]]()
        runtime = create_task_runtime(suite, task_id, metadata["seed"])
        try:
            replay = replay_actions(runtime.env, runtime.initial_states[episode_index], actions, settle_steps=10)
        finally:
            runtime.env.close()
        (args.output_dir / f"replay-{attempt}.json").write_text(json.dumps(replay, indent=2) + "\n")
        runs.append(replay)

    first, second = runs
    comparison = compare_replays(first, second)
    passed = comparison["policy_replay_equal"] and first["steps"] == len(actions) and first["success"] == report["record_success"][episode_id]
    summary = {
        "passed": passed, "source_result_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "candidate_log_sha256": hashlib.sha256((args.run_dir / "candidates.jsonl").read_bytes()).hexdigest(),
        "benchmark": metadata, "init_state_id": episode_index, "settle_steps": 10,
        "steps": [run["steps"] for run in runs], "success": [run["success"] for run in runs],
        "trace_lengths": [len(run["trace"]) for run in runs], **comparison,
        "comparison": "Exact state hashes at every step and exact observation hashes at each 5-step replan; full-frame differences retained",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit("Replay gate failed; inspect replay traces before collecting labels")


if __name__ == "__main__":
    main()
