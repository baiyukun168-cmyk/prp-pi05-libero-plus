"""Continue isolated Kit throughput probes without changing existing experiments."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

out = Path(__file__).resolve().parent
previous = Path('/root/prp/results/official-timing/parallel.tDF9X5')
root = Path('/root/prp')
kit = root/'RoboScope-Kit'
def capture(cmd):
    return subprocess.check_output(cmd, text=True).strip()
if capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']):
    raise SystemExit('GPU busy; no probes started')
if 'nvidia-cuda-mps' in capture(['ps','-eo','comm']):
    raise SystemExit('Another MPS session exists; no probes started')
shutil.copy2(previous/'kit_parallel_policy.py', out/'kit_parallel_policy.py')
shutil.copy2(previous/'benchmark-manifest.json', out/'benchmark-manifest.json')
env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH=str(root/'kit-tools/lib'),
           MAGICK_HOME=str(root/'kit-tools'), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
           MUJOCO_EGL_DEVICE_ID='0', __EGL_VENDOR_LIBRARY_FILENAMES=str(next(root.glob('nvidia-egl.*.json'))),
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
           PYTHONPATH=str(kit/'packages/roboscope-kit/src'), XLA_PYTHON_CLIENT_PREALLOCATE='false',
           OPENPI_DATA_HOME=str(root/'openpi-cache'))
tests=[(8,1,False),(4,2,False),(8,2,False)]
if shutil.which('nvidia-cuda-mps-control'):
    tests.append((8,2,True))
experiment=json.loads((previous/'experiment.json').read_text())
experiment.update(started_utc=datetime.now(timezone.utc).isoformat(), previous_experiment=str(previous),
                  tests=[{'workers':k,'replicas':g,'mps':m} for k,g,m in tests],
                  cpu_affinity=len(os.sched_getaffinity(0)), gpu=capture(['nvidia-smi']))
for legacy_key in ('workers', 'replicas', 'mps'):
    experiment.pop(legacy_key, None)  # Per-trial tests above are authoritative.
(out/'experiment.json').write_text(json.dumps(experiment,indent=2))
trials=[]
for workers, replicas, mps in tests:
    name=f'k{workers}-g{replicas}'+('-mps' if mps else '')
    trial=out/name
    trial.mkdir()
    policy={'schema_version':2,'cwd':str(root/'openpi'),'command':['/usr/bin/env','-u','LD_LIBRARY_PATH',
        f'PYTHONPATH={kit}/packages/roboscope-kit/src', f'PARALLEL_TRIAL_DIR={trial}',
        str(root/'openpi/.venv/bin/python'), str(out/'kit_parallel_policy.py')]}
    (trial/'policy-config.json').write_text(json.dumps(policy,indent=2))
    command=[str(root/'kit-tools/bin/python'),'-m','roboscope_kit.runtime.benchmark',
        '--benchmark','libero','--suite','libero_10','--episodes','1',
        '--checkpoint-source',str(root/'checkpoints/pi05_libero'),
        '--benchmark-manifest',str(out/'benchmark-manifest.json'),
        '--policy-config',str(trial/'policy-config.json'),
        '--output-dir',str(trial/'evaluation'),'--concurrency',str(workers),
        '--replicas',str(replicas),'--run-cache','off']
    if mps: command.append('--cuda-mps')
    (trial/'command.json').write_text(json.dumps(command,indent=2))
    started=time.perf_counter()
    (out/'progress.json').write_text(json.dumps({'active':name,'completed_trials':trials},indent=2))
    with (trial/'cli.log').open('w') as log:
        process=subprocess.Popen(command,env=env,cwd=kit,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        (trial/'pid').write_text(str(process.pid))
        with (trial/'gpu-samples.jsonl').open('w') as gpu:
            while process.poll() is None:
                sample=capture(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'])
                gpu.write(json.dumps({'elapsed_s':time.perf_counter()-started,'memory_mib,utilization_pct':sample})+'\n')
                gpu.flush()
                time.sleep(1)
    summary={'name':name,'workers':workers,'replicas':replicas,'mps':mps,
             'exit_code':process.returncode,'total_wall_s':time.perf_counter()-started}
    result=trial/'evaluation/result.json'
    if result.exists():
        data=json.loads(result.read_text())
        summary.update(result=data.get('result'),phase_timings=data.get('phase_timings'),
                       resources=data.get('resources'),record_success=data.get('record_success'),
                       recovery=data.get('recovery'),cuda_mps=data.get('cuda_mps'))
    trials.append(summary)
    (out/'progress.json').write_text(json.dumps({'active':None,'completed_trials':trials},indent=2))
    (out/'summary.json').write_text(json.dumps(trials,indent=2))
    if process.returncode:
        # A failed capacity probe is evidence; do not push higher concurrency blindly.
        raise SystemExit(f'{name} failed; inspect cli.log; later probes not started')
(out/'finished.json').write_text(json.dumps({'finished_utc':datetime.now(timezone.utc).isoformat(),'trials':len(trials)}))
