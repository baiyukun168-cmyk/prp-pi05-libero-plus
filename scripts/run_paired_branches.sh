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
export PYTHONPATH="$script_dir/..:$root/libero-plus-runtime/LIBERO-plus:$root/RoboScope-Kit/packages/roboscope-kit/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
    for vendor in "$root"/nvidia-egl.*.json; do
        if [[ -f "$vendor" ]]; then
            export __EGL_VENDOR_LIBRARY_FILENAMES="$vendor"
            break
        fi
    done
fi
mkdir -p "$root/results/paired"
session="$(mktemp -d "$root/results/paired/branches.XXXXXX")"
echo "Paired output: $session/evaluation"
cp -- "$script_dir/../src/evaluation/paired_branches.py" "$script_dir/../src/evaluation/branch_model_server.py" "$session/"
"$root/libero-plus-runtime/env/bin/python" "$session/paired_branches.py" \
    --run-dir "$run_dir" --output-dir "$session/evaluation" --root "$root" "${@:2}" \
    2>&1 | tee "$session/paired.log"
