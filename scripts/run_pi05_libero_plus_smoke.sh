#!/usr/bin/env bash
set -euo pipefail

root="${PRP_ROOT:-/root/prp}"
runtime="$root/libero-plus-runtime"
openpi="$root/openpi"
kit="$root/RoboScope-Kit"
tools="$root/kit-tools"
checkpoint="$root/checkpoints/pi05_libero"
manifest="$runtime/benchmark-manifest.json"

if [[ ! -s "$manifest" || ! -x "$runtime/env/bin/python" ||
      ! -x "$openpi/.venv/bin/python" || ! -x "$tools/bin/python" ||
      ! -s "$checkpoint/params/manifest.ocdbt" ||
      ! -s "$checkpoint/assets/physical-intelligence/libero/norm_stats.json" ]]; then
    echo "Missing LIBERO-Plus manifest, Python environment, or checkpoint assets." >&2
    exit 1
fi

original_library_path="${LD_LIBRARY_PATH-}"
export LD_LIBRARY_PATH="$tools/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl

if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
    for vendor in "$root"/nvidia-egl.*.json; do
        if [[ -f "$vendor" ]]; then
            export __EGL_VENDOR_LIBRARY_FILENAMES="$vendor"
            break
        fi
    done
fi

"$runtime/env/bin/python" - <<'PY'
import mujoco

model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'
)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.Renderer(model, height=64, width=64) as renderer:
    renderer.update_scene(data)
    frame = renderer.render()
    assert frame.shape == (64, 64, 3), frame.shape
    print("EGL render OK", frame.shape, flush=True)
PY

mkdir -p "$root/results/smoke"
session="$(mktemp -d "$root/results/smoke/pi05-libero-plus.XXXXXX")"
output="$session/evaluation"
echo "Run directory: $session"

cat > "$session/policy.py" <<'PY'
import jax
import numpy as np
from openpi.policies import policy_config
from openpi.training import config
from openpi_client import image_tools
from roboscope_kit import ActionChunk, BenchmarkName, Policy, PolicyCapabilities, run


class Pi05LiberoPlusPolicy(Policy):
    def load(self, context):
        print(f"JAX GPU: {jax.devices('gpu')}", flush=True)
        self.policy = policy_config.create_trained_policy(
            config.get_config("pi05_libero"), context.checkpoint_path
        )
        benchmark = BenchmarkName.LIBERO_PLUS.value
        return PolicyCapabilities(
            supported_benchmarks=(benchmark,),
            required_features=(
                "images.primary", "images.wrist",
                "state.proprioception", "language.instruction",
            ),
            supported_suites={benchmark: ("libero_10",)},
            supported_action_chunk_sizes=(5,),
        )

    def act(self, observation):
        features = observation.features
        proprio = np.asarray(features["state.proprioception"], dtype=np.float32)
        # LIBERO sends position, an XYZW quaternion, then gripper qpos.
        w = np.clip(float(proprio[6]), -1.0, 1.0)
        sin_half_angle = np.sqrt(1.0 - w * w)
        rotation = np.zeros(3, dtype=np.float32)
        if sin_half_angle > 0.0:
            rotation = proprio[3:6] * np.float32(2.0 * np.arccos(w) / sin_half_angle)

        model_input = {
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(features["images.primary"], 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(features["images.wrist"], 224, 224)
            ),
            "observation/state": np.concatenate((proprio[:3], rotation, proprio[7:9])),
            "prompt": features["language.instruction"],
        }
        actions = np.asarray(self.policy.infer(model_input)["actions"], dtype=np.float32)
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite OpenPI actions of shape (10, 7), got {actions.shape}")
        return ActionChunk(np.ascontiguousarray(actions[:5]))


if __name__ == "__main__":
    run(Pi05LiberoPlusPolicy())
PY

"$tools/bin/python" - "$session" "$openpi" "$kit" "$original_library_path" <<'PY'
import json
from pathlib import Path
import sys

session, openpi, kit = (Path(arg) for arg in sys.argv[1:4])
library_path = sys.argv[4]
command = ["/usr/bin/env"]
if library_path:
    command.append(f"LD_LIBRARY_PATH={library_path}")
else:
    command.extend(("-u", "LD_LIBRARY_PATH"))
command.extend((
    f"PYTHONPATH={kit / 'packages/roboscope-kit/src'}",
    str(openpi / ".venv/bin/python"),
    str(session / "policy.py"),
))
config = {"schema_version": 2, "command": command, "cwd": str(openpi)}
(session / "roboscope-policy.json").write_text(json.dumps(config, indent=2) + "\n")
PY

"$tools/bin/python" -m roboscope_kit.runtime.benchmark \
    --benchmark libero_plus \
    --suite libero_10 \
    --quick-test \
    --checkpoint-source "$checkpoint" \
    --benchmark-manifest "$manifest" \
    --policy-config "$session/roboscope-policy.json" \
    --output-dir "$output"

"$tools/bin/python" - "$output/result.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
report = json.loads(path.read_text())
result = report["result"]
print(json.dumps({
    "episodes": result["episodes"],
    "total_steps": result["total_steps"],
    "record_success": report["record_success"],
    "result_file": str(path),
}, indent=2))
if result["episodes"] != 1 or result["total_steps"] <= 0:
    raise SystemExit("Expected one terminal episode with at least one executed policy step.")
PY
