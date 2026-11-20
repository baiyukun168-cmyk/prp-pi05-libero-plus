#!/usr/bin/env bash
set -uo pipefail
root="${PRP_ROOT:-/root/prp}"
mkdir -p "$root/results/camera/jobs"
job="$(mktemp -d "$root/results/camera/jobs/camera419.XXXXXX")"
echo "JOB_DIR=$job"
echo "$$" > "$job/pid"
date -u +%FT%TZ > "$job/started-utc.txt"
cp -- "$root/configs/camera_419_scored.json" "$job/config.json"
nvidia-smi > "$job/gpu-before.txt"
for repository in openpi RoboScope-Kit; do
    "$root/kit-tools/bin/git" -C "$root/$repository" rev-parse HEAD >> "$job/code-versions.txt"
done
"$root/openpi/.venv/bin/python" -m pip freeze > "$job/openpi-packages.txt" 2>&1
"$root/libero-plus-runtime/env/bin/python" -m pip freeze > "$job/simulator-packages.txt" 2>&1
bash "$root/scripts/run_camera_scored.sh" \
    "$root/results/smoke/pi05-libero-plus-candidates.6qLk3g" "$job/config.json" \
    > "$job/run.log" 2>&1
status=$?
echo "$status" > "$job/exit-code.txt"
date -u +%FT%TZ > "$job/finished-utc.txt"
nvidia-smi > "$job/gpu-after.txt"
exit "$status"
