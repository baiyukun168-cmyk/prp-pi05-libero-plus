from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

out=Path(__file__).resolve().parent
root=Path('/root/prp')
kit=root/'RoboScope-Kit'
source=root/'results/camera/kit-full419.xnQoJI'
pilot=root/'results/camera/kit-pilot.HlFDzm'
def capture(command):return subprocess.check_output(command,text=True).strip()
if capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']):
    raise SystemExit('GPU busy; ablation not started')
snapshot=json.loads((source/'experiment.json').read_text())
for name in ('candidate_selection.py','pi05_candidate_policy.py','pi05_scored_candidate_policy.py','shared_prefix_sampler.py','policy.py'):
    raw=(source/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==snapshot['source_hashes'][name]
    (out/name).write_bytes(raw)
shutil.copy2(source/'benchmark-manifest.json',out/'benchmark-manifest.json')
selection=json.loads((pilot/'camera-selection.json').read_text())
assert len(selection['tasks'])==25
prompts={}
for row in selection['tasks']:
    path=root/'libero-plus-runtime/LIBERO-plus/libero/libero/bddl_files/libero_10'/(row['official_name'].split('_view_')[0]+'.bddl')
    match=re.search(r'\(:language\s+([^)]*)\)',path.read_text(),re.S)
    assert match
    prompts[f"task-{row['task_id']:04d}-episode-0000"]=' '.join(match.group(1).split())
selection['prompts']=prompts
selection['scope']='Fixed previous 25-entry Camera pilot, BDDL instructions, paired algorithm ablation'
(out/'task-selection.json').write_text(json.dumps(selection,indent=2))
schedule=[(7,'ten'),(7,'four'),(7,'scored'),(17,'scored'),(17,'four'),(17,'ten')]
experiment={'started_utc':datetime.now(timezone.utc).isoformat(),'source_run':str(source),
 'selection_source':str(pilot),'task_count':25,'total_episodes':150,'workers':4,'replicas':1,'mps':False,
 'environment_seed':7,'init_state_index':0,'noise_seeds':[7,17],'prompt_source':'bddl',
 'executed_action_chunk':5,'settle_steps':10,'max_action_steps':520,
 'arms':{'ten':'Official unmodified sample_actions, num_steps=10, explicit paired Gaussian noise',
         'four':'Official unmodified sample_actions, num_steps=4, same candidate-A noise',
         'scored':'Archived two-candidate shared-prefix sampler, 4 steps, FlowRes gap .05, fallback10'},
 'schedule':schedule,'counterbalancing':'reverse arm order on second noise seed',
 'limitation':'Two action-noise seeds on 25 repeated tasks; same environment seed and init index; not 50 independent tasks',
 'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.py')},
 'gpu':capture(['nvidia-smi'])}
experiment['versions']={name:capture([str(root/'kit-tools/bin/git'),'-C',str(path),'rev-parse','HEAD']) for name,path in
 [('openpi',root/'openpi'),('kit',kit),('libero_plus',root/'libero-plus-runtime/LIBERO-plus')]}
(out/'experiment.json').write_text(json.dumps(experiment,indent=2))
env=os.environ.copy()
env.pop('PRP_MAX_FLOW_RESIDUAL',None)
env.update(PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(root/'kit-tools/lib'),
 MAGICK_HOME=str(root/'kit-tools'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',
 __EGL_VENDOR_LIBRARY_FILENAMES=str(next(root.glob('nvidia-egl.*.json'))),
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
 PYTHONPATH=f'{out}:{kit}/packages/roboscope-kit/src',XLA_PYTHON_CLIENT_PREALLOCATE='false',
 OPENPI_DATA_HOME=str(root/'openpi-cache'),PRP_MIN_SCORE_GAP='.05',PROMPT_SELECTION=str(out/'task-selection.json'))
summaries=[]
for seed,mode in schedule:
    name=f'seed{seed}-{mode}'
    trial=out/name
    trial.mkdir()
    policy={'schema_version':2,'cwd':str(root/'openpi'),'command':['/usr/bin/env','-u','LD_LIBRARY_PATH',
       f'PYTHONPATH={out}:{kit}/packages/roboscope-kit/src',f'ABLATION_TRIAL_DIR={trial}',
       f'ABLATION_MODE={mode}',f'ABLATION_NOISE_SEED={seed}',
       str(root/'openpi/.venv/bin/python'),str(out/'ablation_policy.py')]}
    (trial/'policy-config.json').write_text(json.dumps(policy,indent=2))
    command=[str(root/'kit-tools/bin/python'),str(out/'ablation_cli.py'),
       '--benchmark','libero_plus','--suite','libero_10','--episodes','1',
       '--checkpoint-source',str(root/'checkpoints/pi05_libero'),
       '--benchmark-manifest',str(out/'benchmark-manifest.json'),
       '--policy-config',str(trial/'policy-config.json'),'--output-dir',str(trial/'evaluation'),
       '--concurrency','4','--replicas','1','--run-cache','off']
    (trial/'command.json').write_text(json.dumps(command,indent=2))
    (out/'progress.json').write_text(json.dumps({'active':name,'completed':summaries},indent=2))
    started=time.perf_counter()
    with (trial/'cli.log').open('w') as log:
        proc=subprocess.Popen(command,env=env,cwd=kit,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        (trial/'pid').write_text(str(proc.pid))
        with (trial/'gpu-samples.jsonl').open('w') as gpu:
            while proc.poll() is None:
                sample=capture(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'])
                gpu.write(json.dumps({'elapsed_s':time.perf_counter()-started,'sample':sample})+'\n')
                gpu.flush()
                time.sleep(1)
    summary={'name':name,'seed':seed,'mode':mode,'exit_code':proc.returncode,'total_wall_s':time.perf_counter()-started}
    result=trial/'evaluation/result.json'
    if result.exists():
        d=json.loads(result.read_text());summary.update(result=d['result'],phases=d['phase_timings'],
           recovery=d['recovery'],gpu=d['resources']['gpu'])
    summaries.append(summary)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    if proc.returncode:raise SystemExit(f'{name} failed; later trials not started')
(out/'progress.json').write_text(json.dumps({'active':None,'completed':summaries},indent=2))
(out/'finished.json').write_text(json.dumps({'finished_utc':datetime.now(timezone.utc).isoformat(),'trials':6,'episodes':150}))
