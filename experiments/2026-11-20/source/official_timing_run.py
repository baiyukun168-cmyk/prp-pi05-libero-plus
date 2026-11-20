"""Isolated orchestration; never changes upstream sources or existing environments."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

out = Path(__file__).resolve().parent
root = Path("/root/prp")
base = os.environ.copy()
base.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0")
def capture(command):
    return subprocess.check_output(command, text=True).strip()
gpu_command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
if capture(gpu_command):
    raise SystemExit("GPU is in use; benchmark not started")
versions = {}
for name, path in (("openpi", root / "openpi"), ("libero", root / "openpi/third_party/libero")):
    git = [str(root / "kit-tools/bin/git"), "-C", str(path)]
    versions[name] = {"commit": capture(git + ["rev-parse", "HEAD"]),
                      "status": capture(git + ["status", "--short"])}
source = root / "openpi/examples/libero/main.py"
versions["official_eval_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
(out / "manifest.json").write_text(json.dumps({
    "started_utc": datetime.now(timezone.utc).isoformat(), "versions": versions,
    "suite": "standard LIBERO-10 (not LIBERO-Plus Camera)", "trials_per_task": 1,
    "seed": 7, "init_state_index": 0, "flow_steps": 10, "replan_steps": 5,
    "settle_steps": 10, "max_action_steps": 520, "model": "JAX bfloat16 pi05_libero",
    "gpu": capture(["nvidia-smi"]), "note": "Native eval_libero; external timing and env cleanup only",
}, indent=2))
simenv = base.copy()
cleanroot = root / "openpi/third_party/libero/libero/libero"
configdir = out / "libero-config"
configdir.mkdir()
paths = {"benchmark_root": str(cleanroot), "bddl_files": str(cleanroot / "bddl_files"),
         "init_states": str(cleanroot / "init_files"), "assets": str(cleanroot / "assets"),
         "datasets": str(cleanroot.parent / "datasets")}
(configdir / "config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in paths.items()) + "\n")
simenv.update(LD_LIBRARY_PATH=str(root / "kit-tools/lib"), MAGICK_HOME=str(root / "kit-tools"),
              MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID="0",
              __EGL_VENDOR_LIBRARY_FILENAMES=str(next(root.glob("nvidia-egl.*.json"))),
              LIBERO_CONFIG_PATH=str(configdir),
              PYTHONPATH=f"{root}/openpi/third_party/libero:{root}/openpi/packages/openpi-client/src")
modelenv = base.copy()
modelenv.pop("LD_LIBRARY_PATH", None)
modelenv.update(XLA_PYTHON_CLIENT_PREALLOCATE="false", OPENPI_DATA_HOME=str(root / "openpi-cache"))
model = None
status = 1
started = time.perf_counter()
try:
    with (out / "server.log").open("w") as serverlog, (out / "client.log").open("w") as clientlog:
        model = subprocess.Popen([str(root / "openpi/.venv/bin/python"), "-u", str(out / "official_timing_server.py"), str(out)],
                                 cwd=root / "openpi", env=modelenv, stdout=serverlog, stderr=subprocess.STDOUT)
        (out / "server-pid").write_text(str(model.pid))
        for _ in range(240):
            if model.poll() is not None:
                raise RuntimeError("Model server exited; see server.log")
            try:
                with urllib.request.urlopen("http://CLOUD_HOST:8011/healthz", timeout=1) as response:
                    if response.status == 200: break
            except (OSError, TimeoutError): time.sleep(1)
        else: raise TimeoutError("Model server did not become ready")
        (out / "server-ready.json").write_text(json.dumps({"startup_wall_s": time.perf_counter() - started}))
        client = subprocess.Popen([str(root / "libero-plus-runtime/env/bin/python"), "-u", str(out / "official_timing_client.py"), str(out)],
                                  cwd=out, env=simenv, stdout=clientlog, stderr=subprocess.STDOUT)
        (out / "client-pid").write_text(str(client.pid))
        with (out / "gpu-samples.jsonl").open("w") as metrics:
            while client.poll() is None:
                sample = capture(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
                metrics.write(json.dumps({"elapsed_s": time.perf_counter() - started, "memory_mib,utilization_pct": sample}) + "\n")
                metrics.flush()
                time.sleep(1)
        status = client.returncode
finally:
    if model is not None and model.poll() is None:
        model.terminate()
        try: model.wait(timeout=15)
        except subprocess.TimeoutExpired:
            model.kill()
            model.wait()
    (out / "exit-code.txt").write_text(str(status) + "\n")
    (out / "finished.json").write_text(json.dumps({"wall_s": time.perf_counter() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "status": status}, indent=2))
sys.exit(status)
