"""Replay paired candidate prefixes with a shared ten-step continuation policy."""

import argparse
import hashlib
import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from src.evaluation.candidate_replay import fingerprint, selected_actions


def continuation_seed(seed, node, step):
    return int.from_bytes(hashlib.sha256(f"continuation:{seed}:{node}:{step}".encode()).digest()[:8], "big")


def summarize_pairs(records):
    pairs = {}
    for record in records:
        pair = pairs.setdefault(record["node"], {})
        if record["candidate_idx"] in pair:
            raise ValueError("Duplicate candidate outcome")
        pair[record["candidate_idx"]] = int(record["success"])
    if not pairs or any(set(pair) != {0, 1} for pair in pairs.values()):
        raise ValueError("Expected complete A/B pairs")
    outcomes = list(pairs.values())
    random_rate = sum(sum(pair.values()) / 2 for pair in outcomes) / len(outcomes)
    oracle_rate = sum(max(pair.values()) for pair in outcomes) / len(outcomes)
    return {"pairs": len(outcomes), "random_success_rate": random_rate,
            "oracle_success_rate": oracle_rate, "oracle_gap": oracle_rate-random_rate,
            "discordance_rate": sum(pair[0] != pair[1] for pair in outcomes)/len(outcomes)}


def collect_source(runtime, connection, metadata, output_dir, max_steps):
    from roboscope_kit.runtime.benchmarks.libero.worker import build_observation
    runtime.env.reset()
    observation = runtime.env.set_init_state(runtime.initial_states[0])
    for _ in range(10):
        observation, _, success, _ = runtime.env.step(np.asarray([0.]*6+[-1.], dtype=np.float32))
        if success or runtime.env.env.done:
            raise RuntimeError("Source episode ended during settle")
    rows = []
    step = 0
    success = False
    episode_id = f"task-{runtime.task_id:04d}-episode-0000"
    while step < max_steps:
        obs = build_observation(episode_id, step, observation, runtime.task_description)
        candidates = []
        for index in (0, 1):
            seed = continuation_seed(metadata["seed"], runtime.task_id, step*2+index)
            connection.send({"features": obs.features, "seed": seed, "num_steps": 4})
            if not connection.poll(180):
                raise TimeoutError("Source candidate generation timed out")
            reply = connection.recv()
            candidates.append({"candidate_idx": index, "noise_seed": seed,
                               "action_prefix": reply["actions"].tolist()})
        rows.append({"episode_id": episode_id, "step": step, "replan_idx": step//5,
                     "selected_candidate_idx": 0, "num_steps": 4, "candidates": candidates})
        for action in candidates[0]["action_prefix"]:
            observation, _, success, _ = runtime.env.step(np.asarray(action,dtype=np.float32))
            step += 1
            if success or runtime.env.env.done or step >= max_steps:
                break
        if success or runtime.env.env.done:
            break
    (output_dir/"source_candidates.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    (output_dir/"source_outcome.json").write_text(json.dumps({"success": bool(success), "steps": step,
                                                           "task_id": runtime.task_id,
                                                           "instruction": runtime.task_description},indent=2)+"\n")
    return rows, selected_actions(rows, total_steps=step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--nodes", type=int, nargs="+", default=[0, 15, 30])
    parser.add_argument("--task-id", type=int, help="Collect a new four-step source rollout on this task first")
    args = parser.parse_args()
    source = json.loads((args.run_dir/"evaluation/result.json").read_text())
    metadata = source["benchmark"]
    rows = [json.loads(line) for line in (args.run_dir/"candidates.jsonl").read_text().splitlines()]
    actions = selected_actions(rows, total_steps=source["result"]["total_steps"])
    if len(set(args.nodes)) != len(args.nodes) or any(n < 0 for n in args.nodes):
        raise ValueError("Nodes must be unique valid replan indices")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    socket = str(args.output_dir/"model.sock")
    if len(socket.encode()) >= 104:
        raise ValueError("Unix model socket path too long")
    model_env = os.environ.copy()
    model_env.pop("LD_LIBRARY_PATH", None)
    model_env["PYTHONPATH"] = os.pathsep.join((str(args.run_dir), str(args.root/"RoboScope-Kit/packages/roboscope-kit/src")))
    model_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    model_log = (args.output_dir/"model.log").open("w")
    process = subprocess.Popen([str(args.root/"openpi/.venv/bin/python"),
                               str(Path(__file__).with_name("branch_model_server.py")),
                               socket, str(args.root/"checkpoints/pi05_libero")],
                              env=model_env, cwd=args.root/"openpi", stdout=model_log, stderr=subprocess.STDOUT)
    connection = None
    try:
        deadline = time.monotonic()+300
        while not Path(socket).exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Model did not start; see model.log")
            time.sleep(.25)
        connection = Client(socket, family="AF_UNIX")
        from libero.libero import benchmark
        from roboscope_kit.runtime.benchmarks.libero.worker import create_task_runtime, build_observation, MAX_STEPS

        if args.task_id is not None:
            metadata = {**metadata, "task_ids": [args.task_id]}
            random.seed(metadata["seed"])
            np.random.seed(metadata["seed"])
            suite = benchmark.get_benchmark_dict()[metadata["suite"]]()
            runtime = create_task_runtime(suite, args.task_id, metadata["seed"])
            try:
                rows, actions = collect_source(runtime, connection, metadata, args.output_dir, MAX_STEPS[metadata["suite"]])
            finally:
                runtime.env.close()
        if any(node >= len(rows) for node in args.nodes):
            raise ValueError("Requested branch node exceeds collected source trajectory")
        records = []
        for node in args.nodes:
            before = None
            for candidate_idx in (0, 1):
                random.seed(metadata["seed"])
                np.random.seed(metadata["seed"])
                suite = benchmark.get_benchmark_dict()[metadata["suite"]]()
                runtime = create_task_runtime(suite, metadata["task_ids"][0], metadata["seed"])
                try:
                    runtime.env.reset()
                    observation = runtime.env.set_init_state(runtime.initial_states[0])
                    for action in [np.asarray([0.]*6+[-1.], dtype=np.float32)]*10 + list(actions[:node*5]):
                        observation, _, success, _ = runtime.env.step(action)
                        if success or runtime.env.env.done:
                            raise RuntimeError("Replay ended before branch")
                    current = fingerprint(runtime.env, observation)
                    if before is not None and current != before:
                        raise RuntimeError(f"Branch input differs at node {node}")
                    before = current
                    np.savez_compressed(args.output_dir/f"node-{node}-candidate-{candidate_idx}.npz",
                                        sim_state=runtime.env.get_sim_state(), **observation)
                    (args.output_dir/f"node-{node}-candidate-{candidate_idx}.instruction.txt").write_text(runtime.task_description)
                    prefix = np.asarray(rows[node]["candidates"][candidate_idx]["action_prefix"], dtype=np.float32)
                    if prefix.shape != (5,7) or not np.isfinite(prefix).all():
                        raise ValueError("Invalid branch prefix")
                    step = node*5
                    pending = list(prefix)
                    calls = []
                    started = time.monotonic()
                    success = False
                    while step < MAX_STEPS[metadata["suite"]]:
                        if not pending:
                            seed = continuation_seed(metadata["seed"], node, step)
                            obs = build_observation(rows[node]["episode_id"], step, observation, runtime.task_description)
                            connection.send({"features": obs.features, "seed": seed})
                            if not connection.poll(180):
                                raise TimeoutError("Model prediction timed out")
                            reply = connection.recv()
                            pending = list(reply["actions"])
                            calls.append({"step": step, "noise_seed": seed, "policy_ms": reply["policy_ms"],
                                          "cold_call": reply["cold_call"], "actions": reply["actions"].tolist()})
                        observation, _, success, _ = runtime.env.step(pending.pop(0))
                        step += 1
                        if success or runtime.env.env.done:
                            break
                    record = {"node": node, "candidate_idx": candidate_idx, "success": bool(success),
                              "task_id": runtime.task_id, "init_state_id": 0, "environment_seed": metadata["seed"],
                              "noise_seed": rows[node]["candidates"][candidate_idx]["noise_seed"],
                              "prefix": prefix.tolist(), "branch_fingerprint": current, "total_steps": step,
                              "branch_elapsed_s": time.monotonic()-started, "continuation_calls": calls}
                    with (args.output_dir/"branches.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, allow_nan=False)+"\n")
                    records.append(record)
                    print(f"Node {node} candidate {candidate_idx}: success={bool(success)}, steps={step}", flush=True)
                finally:
                    runtime.env.close()
        summary = {**summarize_pairs(records), "source_run": args.run_dir.name,
                   "nodes": args.nodes, "benchmark": metadata, "continuation_num_steps": 10,
                   "xla_preallocate": False}
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if connection is not None:
            connection.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        model_log.close()


if __name__ == "__main__":
    main()
