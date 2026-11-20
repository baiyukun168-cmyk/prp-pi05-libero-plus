from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

out=Path(__file__).resolve().parent
root=Path('/root/prp')
kit=root/'RoboScope-Kit'
previous=root/'results/official-timing/parallel.tDF9X5'
camera=root/'results/camera/scored.bgRYYO'
def capture(cmd): return subprocess.check_output(cmd,text=True).strip()
if capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']):
    raise SystemExit('GPU busy; no evaluation started')
for name in ('candidate_selection.py','pi05_candidate_policy.py','pi05_scored_candidate_policy.py','shared_prefix_sampler.py'):
    shutil.copy2(camera/'policies'/name,out/name)
shutil.copy2(root/'results/smoke/pi05-libero-plus-candidates.6qLk3g/policy.py',out/'policy.py')
shutil.copy2(previous/'benchmark-manifest.json',out/'benchmark-manifest.json')
hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob('*.py')}
experiment={'started_utc':datetime.now(timezone.utc).isoformat(),'benchmark':'standard libero_10',
    'episodes_per_task':1,'workers':4,'replicas':1,'mps':False,'seed':7,'init_state_index':0,
    'checkpoint':str(root/'checkpoints/pi05_libero'),'policy':'existing scored candidate policy',
    'candidate_count':2,'draft_flow_steps':4,'fallback_flow_steps':10,'verification_time':.5,
    'min_score_gap':.05,'max_flow_residual':None,'executed_action_chunk':5,
    'source_camera_run':str(camera),'baseline_comparison':str(previous/'k4-g1'),
    'source_hashes':hashes,'warmup':'scored generation twice plus explicit full-step fallback; separate warmup log',
    'runtime_note':'Same reused simulator environment and standard LIBERO assets as prior parallel tests',
    'gpu':capture(['nvidia-smi'])}
(out/'experiment.json').write_text(json.dumps(experiment,indent=2))
env=os.environ.copy()
env.pop('PRP_MAX_FLOW_RESIDUAL',None)
env.update(PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(root/'kit-tools/lib'),
    MAGICK_HOME=str(root/'kit-tools'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',
    __EGL_VENDOR_LIBRARY_FILENAMES=str(next(root.glob('nvidia-egl.*.json'))),
    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
    PYTHONPATH=f'{out}:{kit}/packages/roboscope-kit/src',XLA_PYTHON_CLIENT_PREALLOCATE='false',
    OPENPI_DATA_HOME=str(root/'openpi-cache'),PRP_MIN_SCORE_GAP='.05')
policy={'schema_version':2,'cwd':str(root/'openpi'),'command':['/usr/bin/env','-u','LD_LIBRARY_PATH',
    f'PYTHONPATH={out}:{kit}/packages/roboscope-kit/src',f'SCORED_TRIAL_DIR={out}',
    str(root/'openpi/.venv/bin/python'),str(out/'kit_scored_policy.py')]}
(out/'policy-config.json').write_text(json.dumps(policy,indent=2))
command=[str(root/'kit-tools/bin/python'),'-m','roboscope_kit.runtime.benchmark',
    '--benchmark','libero','--suite','libero_10','--episodes','1',
    '--checkpoint-source',str(root/'checkpoints/pi05_libero'),
    '--benchmark-manifest',str(out/'benchmark-manifest.json'),
    '--policy-config',str(out/'policy-config.json'),'--output-dir',str(out/'evaluation'),
    '--concurrency','4','--replicas','1','--run-cache','off']
(out/'command.json').write_text(json.dumps(command,indent=2))
started=time.perf_counter()
with (out/'cli.log').open('w') as log:
    process=subprocess.Popen(command,env=env,cwd=kit,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    (out/'cli-pid').write_text(str(process.pid))
    with (out/'gpu-samples.jsonl').open('w') as gpu:
        while process.poll() is None:
            sample=capture(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'])
            gpu.write(json.dumps({'elapsed_s':time.perf_counter()-started,'memory_mib,utilization_pct':sample})+'\n')
            gpu.flush()
            time.sleep(1)
(out/'finished.json').write_text(json.dumps({'finished_utc':datetime.now(timezone.utc).isoformat(),
    'exit_code':process.returncode,'total_wall_s':time.perf_counter()-started},indent=2))
raise SystemExit(process.returncode)
