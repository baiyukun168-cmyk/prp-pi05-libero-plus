#!/usr/bin/env bash
set -euo pipefail

root="${PRP_ROOT:-/root/prp}"
run_dir="${1:?Usage: run_candidate_replay.sh candidate-run-directory}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$root/kit-tools/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAGICK_HOME="$root/kit-tools"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export LIBERO_CONFIG_PATH="$run_dir/evaluation/worker-state/libero-config"
export PYTHONPATH="$root/libero-plus-runtime/LIBERO-plus:$root/RoboScope-Kit/packages/roboscope-kit/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
    for vendor in "$root"/nvidia-egl.*.json; do
        if [[ -f "$vendor" ]]; then
            export __EGL_VENDOR_LIBRARY_FILENAMES="$vendor"
            break
        fi
    done
fi
mkdir -p "$root/results/replay"
session="$(mktemp -d "$root/results/replay/candidate-replay.XXXXXX")"
echo "Replay output: $session/evaluation"
cp -- "$script_dir/../src/evaluation/candidate_replay.py" "$session/candidate_replay.py"
"$root/libero-plus-runtime/env/bin/python" "$session/candidate_replay.py" \
    --run-dir "$run_dir" --output-dir "$session/evaluation" \
    2>&1 | tee "$session/replay.log"
