"""Single full Camera run in a fresh directory; preserve previous runs."""
from collections import Counter
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
pilot=root/'results/camera/kit-pilot.HlFDzm'
runtime=root/'libero-plus-runtime'
def capture(cmd): return subprocess.check_output(cmd,text=True).strip()
if capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']):
    raise SystemExit('GPU busy; full Camera evaluation not started')
for name in ('camera_scored_policy.py','kit_scored_policy.py','candidate_selection.py',
             'pi05_candidate_policy.py','pi05_scored_candidate_policy.py','shared_prefix_sampler.py','policy.py'):
    shutil.copy2(pilot/name,out/name)
source=json.loads((root/'configs/camera_419_scored.json').read_text())
metadata_path=runtime/'LIBERO-plus/libero/libero/benchmark/task_classification.json'
digest=hashlib.sha256(metadata_path.read_bytes()).hexdigest()
if digest!=source['metadata_sha256']:raise RuntimeError('Classification metadata changed')
metadata=json.loads(metadata_path.read_text())['libero_10']
chosen=sorted(source['tasks'],key=lambda row:row['task_id'])
ids=[r['task_id'] for r in chosen]
assert len(chosen)==419 and len(set(ids))==419
for row in chosen:
    actual=metadata[row['task_id']]
    assert int(actual['id'])-1==row['task_id'] and actual['name']==row['official_name']
    assert actual['category']=='Camera Viewpoints' and int(actual['difficulty_level'])==row['difficulty']
all_camera_ids=[int(r['id'])-1 for r in metadata if r['category']=='Camera Viewpoints']
assert ids==sorted(all_camera_ids), 'Configured IDs must cover all Camera entries in libero_10'
selection=dict(source,tasks=chosen,scope='All 419 libero_10 Camera entries, improved only, one episode each',
    selection={'method':'all Camera entries, no sampling','population':419,
               'difficulty_counts':dict(Counter(r['difficulty'] for r in chosen))})
(out/'camera-selection.json').write_text(json.dumps(selection,indent=2))
shutil.copy2(pilot/'benchmark-manifest.json',out/'benchmark-manifest.json')
experiment={'started_utc':datetime.now(timezone.utc).isoformat(),'benchmark':'libero_plus',
    'suite':'libero_10','category':'Camera Viewpoints','episodes':419,'episodes_per_task':1,
    'workers':4,'replicas':1,'mps':False,'seed':7,'init_state_index':0,
    'policy':'same scored improved policy and SDK wrapper as verified 25-entry Camera pilot',
    'draft_steps':4,'fallback_steps':10,'candidate_count':2,'verification_time':.5,
    'min_score_gap':.05,'max_flow_residual':None,'action_chunk_size':5,
    'metadata_sha256':digest,'pilot_run':str(pilot),'difficulty_counts':selection['selection']['difficulty_counts'],
    'no_merge':'Fresh full run; does not overwrite or merge previous partial/pilot results',
    'gpu':capture(['nvidia-smi']),'cpu_affinity':len(os.sched_getaffinity(0)),
    'source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob('*.py')}}
experiment['versions']={name:capture([str(root/'kit-tools/bin/git'),'-C',str(path),'rev-parse','HEAD']) for name,path in
    [('openpi',root/'openpi'),('kit',kit),('libero_plus',runtime/'LIBERO-plus')]}
(out/'experiment.json').write_text(json.dumps(experiment,indent=2))
for name,python in [('model',root/'openpi/.venv/bin/python'),('simulator',runtime/'env/bin/python'),('kit',root/'kit-tools/bin/python')]:
    command=[str(python),'-c', 'import importlib.metadata as m,json,sys; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}},indent=2))']
    (out/f'{name}-environment.json').write_text(capture(command))
env=os.environ.copy()
env.pop('PRP_MAX_FLOW_RESIDUAL',None)
env.update(PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(root/'kit-tools/lib'),
    MAGICK_HOME=str(root/'kit-tools'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',
    __EGL_VENDOR_LIBRARY_FILENAMES=str(next(root.glob('nvidia-egl.*.json'))),
    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
    PYTHONPATH=f'{out}:{kit}/packages/roboscope-kit/src',XLA_PYTHON_CLIENT_PREALLOCATE='false',
    OPENPI_DATA_HOME=str(root/'openpi-cache'),PRP_MIN_SCORE_GAP='.05',CAMERA_SELECTION=str(out/'camera-selection.json'))
policy={'schema_version':2,'cwd':str(root/'openpi'),'command':['/usr/bin/env','-u','LD_LIBRARY_PATH',
    f'PYTHONPATH={out}:{kit}/packages/roboscope-kit/src',f'SCORED_TRIAL_DIR={out}',
    str(root/'openpi/.venv/bin/python'),str(out/'camera_scored_policy.py')]}
(out/'policy-config.json').write_text(json.dumps(policy,indent=2))
command=[str(root/'kit-tools/bin/python'),str(out/'camera_full_cli.py'),
    '--benchmark','libero_plus','--suite','libero_10','--episodes','1',
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
