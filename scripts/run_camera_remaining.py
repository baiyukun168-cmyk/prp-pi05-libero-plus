"""Evaluate the three remaining complete Camera suites using the archived 419-policy snapshot."""

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


POLICY_FILES=("kit_scored_policy.py","candidate_selection.py","pi05_candidate_policy.py",
              "pi05_scored_candidate_policy.py","shared_prefix_sampler.py","policy.py")


def capture(command):
    return subprocess.check_output(command,text=True,timeout=30).strip()


def run_suite(root,bundle,config_path):
    source=root/"results/camera/kit-full419.xnQoJI"
    kit=root/"RoboScope-Kit"
    config=json.loads(config_path.read_text())
    suite=config["suite"]
    metadata=root/"libero-plus-runtime/LIBERO-plus/libero/libero/benchmark/task_classification.json"
    if hashlib.sha256(metadata.read_bytes()).hexdigest()!=config["metadata_sha256"]:
        raise RuntimeError("Cloud task metadata differs from fixed task selection")
    rows=json.loads(metadata.read_text())[suite]
    actual=[{"task_id":index,"official_name":row["name"],"difficulty":row["difficulty_level"]}
            for index,row in enumerate(rows) if row["category"]=="Camera Viewpoints"]
    if config["modes"]!=["scored"] or config["tasks"]!=actual:
        raise ValueError("Configuration must cover the exact Camera population using the scored policy")
    previous=json.loads((source/"experiment.json").read_text())
    output=Path(tempfile.mkdtemp(prefix=f"kit-{suite}-camera.",dir=root/"results/camera"))
    print(f"RUN_DIR={output}",flush=True)
    for name in POLICY_FILES:
        data=(source/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=previous["source_hashes"][name]:
            raise RuntimeError(f"Archived algorithm changed: {name}")
        (output/name).write_bytes(data)
    shutil.copy2(bundle/"camera_suite_cli.py",output/"camera_suite_cli.py")
    shutil.copy2(bundle/"camera_suite_scored_policy.py",output/"camera_suite_scored_policy.py")
    shutil.copy2(source/"benchmark-manifest.json",output/"benchmark-manifest.json")
    shutil.copy2(config_path,output/"camera-selection.json")
    experiment={"suite":suite,"episodes":len(actual),"episodes_per_task":1,"workers":4,"replicas":1,
                "prompt_source":"task","mps":False,"source_run":str(source),"seed":7,"init_state_index":0,
                "draft_steps":4,"fallback_steps":10,"verification_time":.5,"min_score_gap":.05,
                "max_flow_residual":None,"action_chunk_size":5,"metadata_sha256":config["metadata_sha256"],
                "started_utc":datetime.now(timezone.utc).isoformat(),
                "source_hashes":{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in output.glob("*.py")},
                "versions":{name:capture([str(root/"kit-tools/bin/git"),"-C",str(path),"rev-parse","HEAD"])
                            for name,path in (("openpi",root/"openpi"),("kit",kit),("libero_plus",metadata.parents[3]))},
                "gpu":capture(["nvidia-smi"])}
    (output/"experiment.json").write_text(json.dumps(experiment,indent=2)+"\n")
    env=os.environ.copy()
    env.pop("PRP_MAX_FLOW_RESIDUAL",None)
    env.update(PYTHONDONTWRITEBYTECODE="1",LD_LIBRARY_PATH=str(root/"kit-tools/lib"),MAGICK_HOME=str(root/"kit-tools"),
               MUJOCO_GL="egl",PYOPENGL_PLATFORM="egl",MUJOCO_EGL_DEVICE_ID="0",
               __EGL_VENDOR_LIBRARY_FILENAMES=str(next(root.glob("nvidia-egl.*.json"))),
               OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",
               PYTHONPATH=f"{output}:{kit}/packages/roboscope-kit/src",XLA_PYTHON_CLIENT_PREALLOCATE="false",
               OPENPI_DATA_HOME=str(root/"openpi-cache"),PRP_MIN_SCORE_GAP=".05",CAMERA_SELECTION=str(output/"camera-selection.json"))
    policy={"schema_version":2,"cwd":str(root/"openpi"),"command":["/usr/bin/env","-u","LD_LIBRARY_PATH",
            f"PYTHONPATH={output}:{kit}/packages/roboscope-kit/src",f"SCORED_TRIAL_DIR={output}",
            str(root/"openpi/.venv/bin/python"),str(output/"camera_suite_scored_policy.py")]}
    (output/"policy-config.json").write_text(json.dumps(policy,indent=2)+"\n")
    command=[str(root/"kit-tools/bin/python"),str(output/"camera_suite_cli.py"),"--benchmark","libero_plus",
             "--suite",suite,"--episodes","1","--checkpoint-source",str(root/"checkpoints/pi05_libero"),
             "--benchmark-manifest",str(output/"benchmark-manifest.json"),"--policy-config",str(output/"policy-config.json"),
             "--output-dir",str(output/"evaluation"),"--concurrency","4","--replicas","1","--run-cache","off"]
    (output/"command.json").write_text(json.dumps(command,indent=2)+"\n")
    started=time.monotonic()
    with (output/"cli.log").open("w") as log:
        process=subprocess.Popen(command,env=env,cwd=kit,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        (output/"cli-pid").write_text(str(process.pid))
        with (output/"gpu-samples.jsonl").open("w") as gpu:
            while process.poll() is None:
                sample=subprocess.run(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=15)
                gpu.write(json.dumps({"elapsed_s":time.monotonic()-started,"sample":sample.stdout.strip(),"returncode":sample.returncode})+"\n")
                gpu.flush()
                time.sleep(1)
    finished={"finished_utc":datetime.now(timezone.utc).isoformat(),"exit_code":process.returncode,"total_wall_s":time.monotonic()-started}
    (output/"finished.json").write_text(json.dumps(finished,indent=2)+"\n")
    if process.returncode:
        raise RuntimeError(f"{suite} failed; inspect {output}/cli.log")
    result=json.loads((output/"evaluation/result.json").read_text())
    if result["result"]["episodes"]!=len(actual) or set(result["benchmark"]["task_ids"])!={row["task_id"] for row in actual}:
        raise RuntimeError("Completed population differs from selected Camera tasks")
    print(json.dumps({"suite":suite,"output":str(output),"result":result["result"]}),flush=True)
    return str(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("/root/prp"))
    parser.add_argument("--config-dir",type=Path,default=Path(__file__).parent/"configs")
    args=parser.parse_args()
    if capture(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"]):
        raise RuntimeError("GPU has active compute processes")
    runs=[]
    for suite in ("libero_spatial","libero_object","libero_goal"):
        runs.append(run_suite(args.root,Path(__file__).parent,args.config_dir/f"camera_{suite}_scored.json"))
    print("COMPLETED_RUNS="+json.dumps(runs),flush=True)
    from summarize_camera_all import combine
    metadata=args.root/"libero-plus-runtime/LIBERO-plus/libero/libero/benchmark/task_classification.json"
    report=combine([args.root/"results/camera/kit-full419.xnQoJI",*[Path(run) for run in runs]],json.loads(metadata.read_text()))
    summary=args.root/"results/camera"/f"all1599-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json"
    with summary.open("x") as stream:
        stream.write(json.dumps(report,indent=2)+"\n")
    print("FINAL_CAMERA_SUMMARY="+str(summary),flush=True)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
