from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

out = Path(__file__).resolve().parent
root = Path('/root/prp')
source = root/'results/camera/kit-full419.xnQoJI'
kit = root/'RoboScope-Kit'
def capture(cmd): return subprocess.check_output(cmd, text=True).strip()
if capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']):
    raise SystemExit('GPU busy; diagnostic not started')
for name in ('camera_scored_policy.py','kit_scored_policy.py','candidate_selection.py',
             'pi05_candidate_policy.py','pi05_scored_candidate_policy.py','shared_prefix_sampler.py','policy.py'):
    shutil.copy2(source/name, out/name)
shutil.copy2(source/'benchmark-manifest.json', out/'benchmark-manifest.json')
full = json.loads((source/'full-summary.json').read_text())
rows = sorted(full['records'], key=lambda r:r['metadata']['task_id'])
chosen = [next(r for r in rows if r['success'] and r['difficulty']==d) for d in (1,2)]
bases = {r['base_task'] for r in chosen}
for row in rows:
    if not row['success'] and row['difficulty']==5 and row['base_task'] not in bases:
        chosen.append(row)
        bases.add(row['base_task'])
        if len(chosen)==6: break
assert len(chosen)==6
selection_rule = 'Two successful controls (difficulty 1/2), four failed difficulty-5 tasks from distinct other base tasks; diagnostic only, not an unbiased benchmark'
if os.environ.get('PROMPT_SELECTION_SOURCE'):
    previous_selection = json.loads(Path(os.environ['PROMPT_SELECTION_SOURCE']).read_text())
    ids = [r['task_id'] for r in previous_selection['tasks']]
    assert len(ids)==25 and len(set(ids))==25
    lookup = {r['metadata']['task_id']:r for r in rows}
    chosen = [lookup[task_id] for task_id in ids]
    selection_rule = 'Reuse previous fixed 25-entry stratified Camera pilot, independent of diagnostic outcomes'
prompts = {}
for row in chosen:
    path = root/'libero-plus-runtime/LIBERO-plus/libero/libero/bddl_files/libero_10'/(row['base_task']+'.bddl')
    match = re.search(r'\(:language\s+([^)]*)\)', path.read_text(), re.S)
    assert match
    prompts[row['record_id']] = ' '.join(match.group(1).split())
selection = {'tasks':[{'task_id':r['metadata']['task_id'],'difficulty':r['difficulty'],
    'previous_success':r['success'],'official_name':r['official_name']} for r in chosen],
    'bdll_prompts':prompts, 'expected_tasks':len(chosen), 'selection_rule':selection_rule}
(out/'task-selection.json').write_text(json.dumps(selection, indent=2))
(out/'experiment.json').write_text(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),
    'source_run':str(source),'conditions':['raw','bddl'],'workers':4,'replicas':1,'mps':False,
    'episodes_per_task':1,'seed':7,'init_state_index':0,'action_chunk_size':5,
    'changed_variable':'Language prompt only; existing scored algorithm unchanged',
    'hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob('*.py')}},indent=2))
env = os.environ.copy()
env.pop('PRP_MAX_FLOW_RESIDUAL',None)
env.update(PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(root/'kit-tools/lib'),
    MAGICK_HOME=str(root/'kit-tools'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',
    __EGL_VENDOR_LIBRARY_FILENAMES=str(next(root.glob('nvidia-egl.*.json'))),
    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
    PYTHONPATH=f'{out}:{kit}/packages/roboscope-kit/src',XLA_PYTHON_CLIENT_PREALLOCATE='false',
    OPENPI_DATA_HOME=str(root/'openpi-cache'),PRP_MIN_SCORE_GAP='.05',PROMPT_SELECTION=str(out/'task-selection.json'))
summaries=[]
for mode in ('raw','bddl'):
    trial=out/mode
    trial.mkdir()
    policy={'schema_version':2,'cwd':str(root/'openpi'),'command':['/usr/bin/env','-u','LD_LIBRARY_PATH',
        f'PYTHONPATH={out}:{kit}/packages/roboscope-kit/src',f'SCORED_TRIAL_DIR={trial}',f'PROMPT_MODE={mode}',
        str(root/'openpi/.venv/bin/python'),str(out/'prompt_diagnostic_policy.py')]}
    (trial/'policy-config.json').write_text(json.dumps(policy,indent=2))
    command=[str(root/'kit-tools/bin/python'),str(out/'prompt_diagnostic_cli.py'),
        '--benchmark','libero_plus','--suite','libero_10','--episodes','1',
        '--checkpoint-source',str(root/'checkpoints/pi05_libero'),
        '--benchmark-manifest',str(out/'benchmark-manifest.json'),'--policy-config',str(trial/'policy-config.json'),
        '--output-dir',str(trial/'evaluation'),'--concurrency','4','--replicas','1','--run-cache','off']
    (trial/'command.json').write_text(json.dumps(command,indent=2))
    (out/'progress.json').write_text(json.dumps({'active':mode,'completed':summaries},indent=2))
    start=time.perf_counter()
    with (trial/'cli.log').open('w') as log:
        process=subprocess.Popen(command,env=env,cwd=kit,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        (trial/'pid').write_text(str(process.pid))
        status=process.wait()
    summary={'mode':mode,'exit_code':status,'total_wall_s':time.perf_counter()-start}
    if (trial/'evaluation/result.json').exists():
        result=json.loads((trial/'evaluation/result.json').read_text())
        summary.update(result=result['result'],phases=result['phase_timings'],recovery=result['recovery'])
    summaries.append(summary)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    if status: raise SystemExit(f'{mode} failed; later condition not started')
(out/'progress.json').write_text(json.dumps({'active':None,'completed':summaries},indent=2))
(out/'finished.json').write_text(json.dumps({'finished_utc':datetime.now(timezone.utc).isoformat(),'conditions':2}))
