"""Run paired Camera task episodes with a resident baseline/shared model."""

import argparse
import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import random
import subprocess
import time
import uuid

import numpy as np

from src.evaluation.paired_branches import continuation_seed
from src.evaluation.candidate_replay import fingerprint


def summarize(records):
    result = {}
    for mode in sorted({row["mode"] for row in records}):
        rows = [row for row in records if row["mode"] == mode]
        times = [call["policy_ms"] for row in rows for call in row["calls"] if not call["warmup"]]
        calls=[call for row in rows for call in row["calls"]]
        selections=[call["selection"] for call in calls if "selection" in call]
        result[mode] = {"episodes":len(rows), "successes":sum(row["success"] for row in rows),
                        "success_rate":sum(row["success"] for row in rows)/len(rows),
                        "latency_samples":len(times),
                        "mean_policy_ms":float(np.mean(times)) if times else None,
                        "p50_policy_ms":float(np.percentile(times,50)) if times else None,
                        "p95_policy_ms":float(np.percentile(times,95)) if times else None,
                        "fallbacks":sum(item["fallback"] for item in selections),
                        "scored_requests":len(selections),
                        "fallback_rate":sum(item["fallback"] for item in selections)/len(selections) if selections else None}
    return result


def configured_modes(config, task_index):
    modes=tuple(config.get("modes",("baseline","shared")))
    if not modes or len(set(modes))!=len(modes) or any(mode not in ("baseline","shared","scored") for mode in modes):
        raise ValueError("Unknown or duplicate evaluation mode")
    if "scored" in modes and modes != ("scored",):
        raise ValueError("Scored evaluation uses a dedicated resident Policy process")
    return modes if task_index%2==0 else tuple(reversed(modes))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    modes=configured_modes(config,0)
    if args.resume:
        if json.loads((args.output_dir/"config.json").read_text())!=config:
            raise ValueError("Resume configuration differs from archived config")
    else:
        args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    socket=str(args.output_dir/f"model-{uuid.uuid4().hex[:8]}.sock")
    env=os.environ.copy()
    env.pop("LD_LIBRARY_PATH",None)
    policy_dir=args.output_dir.parent/"policies"
    if not policy_dir.is_dir():
        policy_dir=args.root/"src/policies"
    env["PYTHONPATH"]=os.pathsep.join([str(args.source_run),str(args.root/"RoboScope-Kit/packages/roboscope-kit/src"),str(policy_dir)])
    env["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    env["PRP_MIN_SCORE_GAP"]=str(config.get("min_score_gap",.05))
    env.pop("PRP_MAX_FLOW_RESIDUAL",None)
    if config.get("max_flow_residual") is not None:
        env["PRP_MAX_FLOW_RESIDUAL"]=str(config["max_flow_residual"])
    connection=None
    log=(args.output_dir/"model.log").open("a")
    command=[str(args.root/"openpi/.venv/bin/python"),str(Path(__file__).with_name("branch_model_server.py")),socket,str(args.root/"checkpoints/pi05_libero")]
    if modes == ("scored",):
        command.extend(("scored",str(config["seed"])))
    records=[]
    episode_dir=args.output_dir/"episodes"
    episode_dir.mkdir(exist_ok=True)
    if args.resume:
        records=[json.loads(path.read_text()) for path in sorted(episode_dir.glob("*.json"))]
    completed={(row["task_id"],row["mode"]) for row in records}
    if len(completed)!=len(records):
        raise ValueError("Duplicate completed episode records")
    process=subprocess.Popen(command,env=env,cwd=args.root/"openpi",stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+300
        while not Path(socket).exists():
            if process.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError("Model startup failed; inspect model.log")
            time.sleep(.25)
        connection=Client(socket,family="AF_UNIX")
        from libero.libero import benchmark
        from roboscope_kit.runtime.benchmarks.libero.worker import create_task_runtime, build_observation, MAX_STEPS
        from roboscope_kit.runtime.benchmark.resources import GpuSampler, gpu_window
        for task_index, task in enumerate(config["tasks"]):
            before=None
            # Alternate order while preserving per-task paired seeds and initial states.
            modes=configured_modes(config,task_index)
            for mode in modes:
                if (task["task_id"],mode) in completed:
                    continue
                random.seed(config["seed"])
                np.random.seed(config["seed"])
                suite=benchmark.get_benchmark_dict()[config["suite"]]()
                if suite.get_task(task["task_id"]).name != task["official_name"]:
                    raise ValueError("Task ID does not match official Camera metadata")
                runtime=create_task_runtime(suite,task["task_id"],config["seed"])
                sampler=GpuSampler(gpu_devices=("0",))
                sampler.start()
                try:
                    runtime.env.reset()
                    observation=runtime.env.set_init_state(runtime.initial_states[config["init_state_id"]])
                    for _ in range(config["settle_steps"]):
                        observation,_,success,_=runtime.env.step(np.asarray([0.]*6+[-1.],dtype=np.float32))
                        if success or runtime.env.env.done:
                            raise RuntimeError("Episode ended during settle")
                    current=fingerprint(runtime.env,observation)
                    if before is not None and current!=before:
                        raise RuntimeError("Paired initial state or observation differs")
                    before=current
                    step=0
                    calls=[]
                    pending=[]
                    success=False
                    started=time.monotonic()
                    while step<MAX_STEPS[config["suite"]]:
                        if not pending:
                            seed=continuation_seed(config["seed"],task["task_id"],step)
                            second_seed=continuation_seed(config["seed"]+1,task["task_id"],step)
                            obs=build_observation(f"task-{runtime.task_id:04d}-episode-0000",step,observation,runtime.task_description)
                            rpc_start=time.monotonic()
                            connection.send({"features":obs.features,"seed":seed,"second_seed":second_seed,
                                             "episode_id":obs.episode_id,"step":step,
                                             "num_steps":10 if mode=="baseline" else 4,
                                             "mode":"single" if mode=="baseline" else mode})
                            if not connection.poll(240):
                                raise TimeoutError("Prediction timed out")
                            reply=connection.recv()
                            actions=np.asarray(reply["actions"],dtype=np.float32)
                            if actions.shape!=(5,7) or not np.isfinite(actions).all():
                                raise ValueError("Invalid action prefix")
                            pending=list(actions)
                            call={"step":step,"noise_seed":seed,"second_noise_seed":second_seed,
                                          "policy_ms":reply["policy_ms"],"rpc_ms":(time.monotonic()-rpc_start)*1000,
                                          "warmup":reply["warmup"],"actions":actions.tolist()}
                            if "selection" in reply:
                                call["selection"]=reply["selection"]
                                call["noise_seed"]=reply["selection"]["candidates"][0]["noise_seed"]
                                call["second_noise_seed"]=reply["selection"]["candidates"][1]["noise_seed"]
                            calls.append(call)
                        observation,_,success,_=runtime.env.step(pending.pop(0))
                        step+=1
                        if success or runtime.env.env.done:
                            break
                    elapsed=time.monotonic()-started
                finally:
                    samples=sampler.stop()
                    runtime.env.close()
                record={"mode":mode,**task,"instruction":runtime.task_description,"seed":config["seed"],
                        "initial_fingerprint":current,"success":bool(success),"steps":step,
                        "episode_s":elapsed,"calls":calls,"gpu":gpu_window(samples,1),"gpu_samples":len(samples)}
                episode_path=episode_dir/f"task-{runtime.task_id:04d}-{mode}.json"
                temporary=episode_path.with_suffix(".tmp")
                with temporary.open("w") as stream:
                    stream.write(json.dumps(record,allow_nan=False)+"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary,episode_path)
                with (args.output_dir/"episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(record,allow_nan=False)+"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                records.append(record)
                progress=args.output_dir/"progress.tmp"
                progress.write_text(json.dumps(summarize(records),indent=2)+"\n")
                os.replace(progress,args.output_dir/"progress.json")
                print(f"Camera task={runtime.task_id} difficulty={task['difficulty']} mode={mode} success={bool(success)} steps={step}",flush=True)
        if len(records)!=len(config["tasks"])*len(configured_modes(config,0)):
            raise RuntimeError("Final episode count differs from configured workload")
        # Export a canonical journal from authoritative, atomic episode files.
        journal=args.output_dir/"episodes.tmp"
        journal.write_text("".join(json.dumps(record,allow_nan=False)+"\n" for record in records))
        os.replace(journal,args.output_dir/"episodes.jsonl")
        (args.output_dir/"summary.json").write_text(json.dumps(summarize(records),indent=2)+"\n")
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
        log.close()


if __name__=="__main__":
    main()
