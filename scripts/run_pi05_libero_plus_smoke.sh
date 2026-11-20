#!/usr/bin/env bash
set -euo pipefail

root="${PRP_ROOT:-/root/prp}"
candidate_source=""
run_prefix=pi05-libero-plus
case "${1:-}" in
    "") ;;
    --candidate|--shared-candidate|--scored-candidate)
        script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
        candidate_source="$script_dir/../src/policies/pi05_candidate_policy.py"
        run_prefix=pi05-libero-plus-candidates
        if [[ "$1" == --shared-candidate ]]; then
            candidate_source="$script_dir/../src/policies/pi05_shared_candidate_policy.py"
            run_prefix=pi05-libero-plus-shared
        fi
        if [[ "$1" == --scored-candidate ]]; then
            candidate_source="$script_dir/../src/policies/pi05_scored_candidate_policy.py"
            run_prefix=pi05-libero-plus-scored
        fi
        [[ -s "$candidate_source" ]] || { echo "Missing candidate Policy: $candidate_source" >&2; exit 1; }
        ;;
    *) echo "Usage: $0 [--candidate|--shared-candidate|--scored-candidate]" >&2; exit 2 ;;
esac
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
export MAGICK_HOME="${MAGICK_HOME:-$tools}"
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
session="$(mktemp -d "$root/results/smoke/$run_prefix.XXXXXX")"
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
    action_chunk_size = 5
    sample_kwargs = None

    def load(self, context):
        print(f"JAX GPU: {jax.devices('gpu')}", flush=True)
        model_config = config.get_config("pi05_libero")
        self.noise_shape = (model_config.model.action_horizon, model_config.model.action_dim)
        self.policy = policy_config.create_trained_policy(
            model_config, context.checkpoint_path, sample_kwargs=self.sample_kwargs
        )
        benchmark = BenchmarkName.LIBERO_PLUS.value
        return PolicyCapabilities(
            supported_benchmarks=(benchmark,),
            required_features=(
                "images.primary", "images.wrist",
                "state.proprioception", "language.instruction",
            ),
            supported_suites={benchmark: ("libero_10",)},
            supported_action_chunk_sizes=(self.action_chunk_size,),
        )

    def model_input(self, observation):
        features = observation.features
        proprio = np.asarray(features["state.proprioception"], dtype=np.float32)
        # LIBERO sends position, an XYZW quaternion, then gripper qpos.
        w = np.clip(float(proprio[6]), -1.0, 1.0)
        sin_half_angle = np.sqrt(1.0 - w * w)
        rotation = np.zeros(3, dtype=np.float32)
        if sin_half_angle > 0.0:
            rotation = proprio[3:6] * np.float32(2.0 * np.arccos(w) / sin_half_angle)

        return {
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(features["images.primary"], 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(features["images.wrist"], 224, 224)
            ),
            "observation/state": np.concatenate((proprio[:3], rotation, proprio[7:9])),
            "prompt": features["language.instruction"],
        }
    def action_prefix(self, model_input, *, noise=None):
        actions = np.asarray(self.policy.infer(model_input, noise=noise)["actions"], dtype=np.float32)
        expected_shape = (self.noise_shape[0], 7)
        if actions.shape != expected_shape or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite OpenPI actions of shape {expected_shape}, got {actions.shape}")
        return np.ascontiguousarray(actions[:self.action_chunk_size])

    def act(self, observation):
        return ActionChunk(self.action_prefix(self.model_input(observation)))


if __name__ == "__main__":
    run(Pi05LiberoPlusPolicy())
PY

policy_entry=policy.py
if [[ -n "$candidate_source" ]]; then
    policy_entry="$(basename -- "$candidate_source")"
    cp -- "$candidate_source" "$session/$policy_entry"
    if [[ "$policy_entry" == pi05_shared_candidate_policy.py || "$policy_entry" == pi05_scored_candidate_policy.py ]]; then
        cp -- "$script_dir/../src/policies/pi05_candidate_policy.py" "$script_dir/../src/policies/shared_prefix_sampler.py" "$session/"
        cp -- "$script_dir/../src/policies/candidate_selection.py" "$session/"
    fi
fi

"$tools/bin/python" - "$session" "$openpi" "$kit" "$original_library_path" "$policy_entry" "$manifest" <<'PY'
import json
from pathlib import Path
import sys

session, openpi, kit = (Path(arg) for arg in sys.argv[1:4])
library_path = sys.argv[4]
policy_entry = sys.argv[5]
command = ["/usr/bin/env"]
if library_path:
    command.append(f"LD_LIBRARY_PATH={library_path}")
else:
    command.extend(("-u", "LD_LIBRARY_PATH"))
command.extend((
    f"PYTHONPATH={kit / 'packages/roboscope-kit/src'}",
    str(openpi / ".venv/bin/python"),
    str(session / policy_entry),
))
if policy_entry in ("pi05_candidate_policy.py", "pi05_shared_candidate_policy.py", "pi05_scored_candidate_policy.py"):
    manifest = json.loads(Path(sys.argv[6]).read_text())
    command.append(str(manifest["defaults"].get("seed", 7)))
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

"$tools/bin/python" - "$output/result.json" "$policy_entry" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
report = json.loads(path.read_text())
result = report["result"]
if result["episodes"] != 1 or result["total_steps"] <= 0:
    raise SystemExit("Expected one terminal episode with at least one executed policy step.")

candidate_details = {}
if sys.argv[2] in ("pi05_candidate_policy.py", "pi05_shared_candidate_policy.py", "pi05_scored_candidate_policy.py"):
    candidate_log = path.parent.parent / "candidates.jsonl"
    with candidate_log.open() as stream:
        requests = sum(1 for line in stream if line.strip())
    expected_requests = (result["total_steps"] + 4) // 5
    if requests != expected_requests:
        raise SystemExit(f"Expected {expected_requests} candidate log records, found {requests}.")
    candidate_details = {"candidate_requests": requests, "candidate_log": str(candidate_log)}

print(json.dumps({
    "episodes": result["episodes"],
    "total_steps": result["total_steps"],
    "record_success": report["record_success"],
    "result_file": str(path),
    **candidate_details,
}, indent=2))
PY
