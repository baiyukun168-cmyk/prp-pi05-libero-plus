#!/usr/bin/env bash
set -euo pipefail
root="${PRP_ROOT:-/root/prp}"
source_run="${1:?Usage: run_inference_comparison.sh source-run observation-npz}"
observation="${2:?Specify an archived branch observation}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$root/results/inference"
session="$(mktemp -d "$root/results/inference/comparison.XXXXXX")"
echo "Inference output: $session"
cp -- "$script_dir/../src/evaluation/benchmark_candidates.py" "$script_dir/../src/policies/shared_prefix_sampler.py" "$session/"
nvidia-smi > "$session/gpu-before.txt"
env -u LD_LIBRARY_PATH \
    PYTHONPATH="$root/RoboScope-Kit/packages/roboscope-kit/src:$session" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$root/openpi/.venv/bin/python" "$session/benchmark_candidates.py" \
    --source-run "$source_run" --observation "$observation" \
    --checkpoint "$root/checkpoints/pi05_libero" --output "$session/benchmark.json" \
    --include-batch --include-shared-prefix --samples 60 2>&1 | tee "$session/benchmark.log"
nvidia-smi > "$session/gpu-after.txt"
