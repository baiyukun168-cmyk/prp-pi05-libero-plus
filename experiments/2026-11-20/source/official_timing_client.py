"""Time the unchanged official eval_libero loop via external observation wrappers."""
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time

output = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("official_libero_main", "/root/prp/openpi/examples/libero/main.py")
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)
logging.basicConfig(level=logging.INFO)
episodes = []
current = None
environment = None
global_infer_count = 0

class ErrorCapture(logging.Handler):
    def emit(self, record):
        if current is not None:
            current["errors"].append(record.getMessage())

handler = ErrorCapture(level=logging.ERROR)
logging.getLogger().addHandler(handler)
original_client = native._websocket_client_policy.WebsocketClientPolicy

class TimedClient(original_client):
    def infer(self, obs):
        global global_infer_count
        started = time.perf_counter()
        result = super().infer(obs)
        elapsed = time.perf_counter() - started
        current["calls"].append({
            "rpc_s": elapsed,
            "server_infer_ms": result["server_timing"]["infer_ms"],
            "first_request": global_infer_count == 0,
        })
        global_infer_count += 1
        return result

native._websocket_client_policy.WebsocketClientPolicy = TimedClient
original_env = native._get_libero_env

class TimedEnv:
    def __init__(self, env, description, init_s):
        self.env, self.description, self.init_s = env, description, init_s

    def reset(self):
        global current
        current = {"task": self.description, "env_create_s": self.init_s,
                   "calls": [], "env_step_s": 0., "settle_s": 0.,
                   "steps_including_settle": 0, "resize_s": 0., "errors": [],
                   "success": False, "started": time.perf_counter()}
        start = time.perf_counter()
        result = self.env.reset()
        current["reset_s"] = time.perf_counter() - start
        return result

    def set_init_state(self, state):
        start = time.perf_counter()
        result = self.env.set_init_state(state)
        current["set_init_state_s"] = time.perf_counter() - start
        return result

    def step(self, action):
        start = time.perf_counter()
        result = self.env.step(action)
        duration = time.perf_counter() - start
        current["env_step_s"] += duration
        current["steps_including_settle"] += 1
        if current["steps_including_settle"] <= 10:
            current["settle_s"] += duration
        current["success"] = bool(result[2])
        return result

    def close(self):
        self.env.close()

def timed_env(*args, **kwargs):
    global environment
    if environment is not None:
        environment.close()
    start = time.perf_counter()
    env, description = original_env(*args, **kwargs)
    environment = TimedEnv(env, description, time.perf_counter() - start)
    return environment, description

native._get_libero_env = timed_env
original_resize = native.image_tools.resize_with_pad

def timed_resize(*args, **kwargs):
    start = time.perf_counter()
    result = original_resize(*args, **kwargs)
    current["resize_s"] += time.perf_counter() - start
    return result

native.image_tools.resize_with_pad = timed_resize
original_video = native.imageio.mimwrite

def timed_video(*args, **kwargs):
    end = time.perf_counter()
    current["episode_s"] = end - current.pop("started")
    current["action_steps"] = current["steps_including_settle"] - 10
    current["execution_s"] = (current["episode_s"] - current["reset_s"]
                              - current["set_init_state_s"] - current["settle_s"])
    start = time.perf_counter()
    original_video(*args, **kwargs)
    current["video_s"] = time.perf_counter() - start
    episodes.append(current)
    (output / "episodes.json").write_text(json.dumps(episodes, indent=2))
    print("TIMED_EPISODE " + json.dumps({k: v for k, v in current.items() if k != "calls"}), flush=True)

native.imageio.mimwrite = timed_video
start = time.perf_counter()
try:
    native.eval_libero(native.Args(
        host="CLOUD_HOST", port=8011, task_suite_name="libero_10",
        num_trials_per_task=1, seed=7, video_out_path=str(output / "videos"),
    ))
finally:
    if environment is not None:
        environment.close()
    (output / "client-total.json").write_text(json.dumps({"wall_s": time.perf_counter() - start,
                                                          "episodes": len(episodes)}, indent=2))
if len(episodes) != 10 or any(ep["errors"] for ep in episodes):
    raise SystemExit("Evaluation incomplete or errors occurred; inspect episodes.json and client.log")
