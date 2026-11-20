from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

out=Path(__file__).resolve().parent
root=Path('/root/prp')
kit=root/'RoboScope-Kit'
previous=root/'results/official-timing/improved-k4g1.3C3GFh'
runtime=root/'libero-plus-runtime'
def capture(cmd): return subprocess.check_output(cmd,text=True).strip()
if capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']):
    raise SystemExit('GPU busy; no pilot started')
for name in ('kit_scored_policy.py','candidate_selection.py','pi05_candidate_policy.py',
             'pi05_scored_candidate_policy.py','shared_prefix_sampler.py','policy.py'):
    shutil.copy2(previous/name,out/name)
source=json.loads((root/'configs/camera_419_scored.json').read_text())
metadata_path=runtime/'LIBERO-plus/libero/libero/benchmark/task_classification.json'
digest=hashlib.sha256(metadata_path.read_bytes()).hexdigest()
assert digest == source['metadata_sha256']
metadata=json.loads(metadata_path.read_text())['libero_10']
assert len(source['tasks']) == 419
groups=defaultdict(list)
for row in source['tasks']:
    actual=metadata[row['task_id']]
    assert int(actual['id'])-1 == row['task_id'] and actual['name']==row['official_name']
    assert actual['category']=='Camera Viewpoints' and int(actual['difficulty_level'])==row['difficulty']
    groups[row['difficulty']].append(row)
size=25
exact={d:size*len(rows)/419 for d,rows in groups.items()}
quota={d:int(value) for d,value in exact.items()}
for d in sorted(groups,key=lambda d:exact[d]-quota[d],reverse=True)[:size-sum(quota.values())]:
    quota[d]+=1
rng=random.Random(7)
chosen=[]
for d in sorted(groups): chosen.extend(rng.sample(groups[d],quota[d]))
chosen.sort(key=lambda row:row['task_id'])
selection=dict(source,tasks=chosen,scope='25 fixed stratified Camera pilot entries, improved only',
    sampling={'method':'proportional difficulty strata, largest remainder, without replacement',
              'seed':7,'population':419,'population_counts':{d:len(rows) for d,rows in groups.items()},
              'sample_counts':quota})
(out/'camera-selection.json').write_text(json.dumps(selection,indent=2))
manifest=json.loads((runtime/'benchmark-manifest.json').read_text())
manifest['runtime']={'python':str(runtime/'env/bin/python'),'benchmark_root':str(runtime/'LIBERO-plus'),
                     'config_path':str(runtime/'config/config.yaml')}
(out/'benchmark-manifest.json').write_text(json.dumps(manifest,indent=2))
experiment={'started_utc':datetime.now(timezone.utc).isoformat(),'benchmark':'libero_plus',
    'suite':'libero_10','category':'Camera Viewpoints','episodes':25,'episodes_per_task':1,
    'workers':4,'replicas':1,'mps':False,'seed':7,'init_state_index':0,
    'policy':'existing scored policy, same algorithm as previous 10-task improved probe',
    'draft_steps':4,'fallback_steps':10,'candidate_count':2,'verification_time':.5,
    'min_score_gap':.05,'max_flow_residual':None,'action_chunk_size':5,
    'sample':selection['sampling'],'metadata_sha256':digest,
    'task_selection':'External factory subclass uses native task_ids; Kit scheduling, worker and scoring unchanged',
    'source_run':str(previous),'gpu':capture(['nvidia-smi']),
    'source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob('*.py')}}
experiment['versions']={name:capture([str(root/'kit-tools/bin/git'),'-C',str(path),'rev-parse','HEAD']) for name,path in
    [('openpi',root/'openpi'),('kit',kit),('libero_plus',runtime/'LIBERO-plus')]}
(out/'experiment.json').write_text(json.dumps(experiment,indent=2))
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
command=[str(root/'kit-tools/bin/python'),str(out/'camera_subset_cli.py'),
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
