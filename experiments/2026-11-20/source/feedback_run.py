from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

out=Path(__file__).resolve().parent
root=Path('/root/prp');kit=root/'RoboScope-Kit'
previous=root/'results/camera/algorithm-ablation.sf8OV2'
source=root/'results/camera/kit-full419.xnQoJI'
def capture(command):return subprocess.check_output(command,text=True).strip()
if capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']):
    raise SystemExit('GPU busy; feedback experiment not started')
snapshot=json.loads((source/'experiment.json').read_text())
assert hashlib.sha256((source/'policy.py').read_bytes()).hexdigest()==snapshot['source_hashes']['policy.py']
shutil.copy2(source/'policy.py',out/'policy.py')
shutil.copy2(previous/'ablation_cli.py',out/'feedback_cli.py')
shutil.copy2(previous/'benchmark-manifest.json',out/'benchmark-manifest.json')
old=json.loads((previous/'task-selection.json').read_text())
selection={'suite':'libero_10','category':'Camera Viewpoints','tasks':old['tasks'],
    'prompts':old['prompts'],'metadata_sha256':old['metadata_sha256'],
    'source_selection':str(previous/'task-selection.json')}
assert len(selection['tasks'])==25 and len({r['task_id'] for r in selection['tasks']})==25
(out/'task-selection.json').write_text(json.dumps(selection,indent=2))
schedule=[(7,10,5),(7,8,5),(7,10,3),(7,8,3),(17,8,3),(17,10,3),(17,8,5),(17,10,5)]
experiment={'started_utc':datetime.now(timezone.utc).isoformat(),'source_run':str(source),
 'task_count':25,'total_episodes':200,'workers':4,'replicas':1,'mps':False,
 'generation_steps':[8,10],'execution_steps':[3,5],'action_horizon':10,
 'sampler':'official sample_actions, no candidate ranking or fallback',
 'prompt_source':'bddl','environment_seed':7,'init_state_index':0,'noise_seeds':[7,17],
 'noise_index':'physical action step, not floor(step/5) or request counter; common steps share Gaussian noise',
 'previous_result_comparison':'All four configurations rerun; previous ablation cannot be directly reused because noise indexing differs after step zero',
 'settle_steps':10,'max_action_steps':520,'schedule':schedule,
 'counterbalancing':'Reverse order on noise seed17','limitation':'25 repeated tasks, 2 action-noise seeds, fixed init0; not 50 independent tasks',
 'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.py')},
 'gpu':capture(['nvidia-smi'])}
experiment['versions']={name:capture([str(root/'kit-tools/bin/git'),'-C',str(path),'rev-parse','HEAD']) for name,path in
 [('openpi',root/'openpi'),('kit',kit),('libero_plus',root/'libero-plus-runtime/LIBERO-plus')]}
(out/'experiment.json').write_text(json.dumps(experiment,indent=2))
env=os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(root/'kit-tools/lib'),
 MAGICK_HOME=str(root/'kit-tools'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',
 __EGL_VENDOR_LIBRARY_FILENAMES=str(next(root.glob('nvidia-egl.*.json'))),
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
 PYTHONPATH=f'{out}:{kit}/packages/roboscope-kit/src',XLA_PYTHON_CLIENT_PREALLOCATE='false',
 OPENPI_DATA_HOME=str(root/'openpi-cache'),PROMPT_SELECTION=str(out/'task-selection.json'))
summaries=[]
for seed,generation,execution in schedule:
    name=f'seed{seed}-g{generation}-c{execution}';trial=out/name;trial.mkdir()
    policy={'schema_version':2,'cwd':str(root/'openpi'),'command':['/usr/bin/env','-u','LD_LIBRARY_PATH',
        f'PYTHONPATH={out}:{kit}/packages/roboscope-kit/src',f'FEEDBACK_TRIAL_DIR={trial}',
        f'GENERATION_STEPS={generation}',f'EXECUTION_STEPS={execution}',f'ACTION_NOISE_SEED={seed}',
        str(root/'openpi/.venv/bin/python'),str(out/'feedback_policy.py')]}
    (trial/'policy-config.json').write_text(json.dumps(policy,indent=2))
    command=[str(root/'kit-tools/bin/python'),str(out/'feedback_cli.py'),
        '--benchmark','libero_plus','--suite','libero_10','--episodes','1',
        '--checkpoint-source',str(root/'checkpoints/pi05_libero'),
        '--benchmark-manifest',str(out/'benchmark-manifest.json'),
        '--policy-config',str(trial/'policy-config.json'),'--output-dir',str(trial/'evaluation'),
        '--concurrency','4','--replicas','1','--run-cache','off']
    (trial/'command.json').write_text(json.dumps(command,indent=2))
    (out/'progress.json').write_text(json.dumps({'active':name,'completed':summaries},indent=2))
    start=time.perf_counter()
    with (trial/'cli.log').open('w') as log:
        proc=subprocess.Popen(command,env=env,cwd=kit,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        (trial/'pid').write_text(str(proc.pid))
        with (trial/'gpu-samples.jsonl').open('w') as gpu:
            while proc.poll() is None:
                sample=capture(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'])
                gpu.write(json.dumps({'elapsed_s':time.perf_counter()-start,'sample':sample})+'\n');gpu.flush()
                time.sleep(1)
    summary={'name':name,'seed':seed,'generation_steps':generation,'execution_steps':execution,
        'exit_code':proc.returncode,'total_wall_s':time.perf_counter()-start}
    result=trial/'evaluation/result.json'
    if result.exists():
        data=json.loads(result.read_text());summary.update(result=data['result'],phases=data['phase_timings'],
            recovery=data['recovery'],gpu=data['resources']['gpu'])
    summaries.append(summary)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    if proc.returncode:raise SystemExit(f'{name} failed; later trials not started')
(out/'progress.json').write_text(json.dumps({'active':None,'completed':summaries},indent=2))
(out/'finished.json').write_text(json.dumps({'finished_utc':datetime.now(timezone.utc).isoformat(),'trials':8,'episodes':200}))
