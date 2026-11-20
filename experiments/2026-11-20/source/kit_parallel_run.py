from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

out = Path(__file__).resolve().parent
root = Path('/root/prp')
kit = root/'RoboScope-Kit'
def capture(cmd):
    return subprocess.check_output(cmd,text=True).strip()
if capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']):
    raise SystemExit('GPU busy; no tests started')
env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH=str(root/'kit-tools/lib'),
           MAGICK_HOME=str(root/'kit-tools'), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
           MUJOCO_EGL_DEVICE_ID='0', __EGL_VENDOR_LIBRARY_FILENAMES=str(next(root.glob('nvidia-egl.*.json'))),
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
           PYTHONPATH=str(kit/'packages/roboscope-kit/src'), XLA_PYTHON_CLIENT_PREALLOCATE='false',
           OPENPI_DATA_HOME=str(root/'openpi-cache'))
manifest = json.loads((kit/'benchmarks/libero/benchmark-manifest.json').read_text())
cfg = out/'libero-config'
cfg.mkdir()
cleanroot = root/'openpi/third_party/libero/libero/libero'
paths={'benchmark_root':str(cleanroot),'bddl_files':str(cleanroot/'bddl_files'),
       'init_states':str(cleanroot/'init_files'),'assets':str(cleanroot/'assets'),
       'datasets':str(cleanroot.parent/'datasets')}
(cfg/'config.yaml').write_text('\n'.join(f'{k}: {v}' for k,v in paths.items())+'\n')
manifest['runtime']={'python':str(root/'libero-plus-runtime/env/bin/python'),
                     'benchmark_root':str(root/'openpi/third_party/libero'),
                     'config_path':str(cfg/'config.yaml')}
manifest['defaults']['episodes']=1
(out/'benchmark-manifest.json').write_text(json.dumps(manifest,indent=2))
versions={name:capture([str(root/'kit-tools/bin/git'),'-C',str(path),'rev-parse','HEAD'])
          for name,path in [('kit',kit),('openpi',root/'openpi'),('libero',root/'openpi/third_party/libero')]}
(out/'experiment.json').write_text(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),
    'versions':versions,'workers':[1,2,4],'replicas':1,'mps':False,'episodes_per_task':1,
    'suite':'standard libero_10','flow_steps':10,'executed_chunk_size':5,
    'rng':'official default key(0), independently reset and retained per episode',
    'warmup':'two dummy requests in load; excluded from evaluation phase',
    'runtime_note':'Existing simulator environment reused; standard LIBERO assets from OpenPI submodule, not a new pinned Kit build',
    'gpu':capture(['nvidia-smi'])},indent=2))
trials=[]
for workers in (1,2,4):
    trial=out/f'k{workers}-g1'
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
        '--output-dir',str(trial/'evaluation'),'--concurrency',str(workers),'--replicas','1','--run-cache','off']
    (trial/'command.json').write_text(json.dumps(command,indent=2))
    started=time.perf_counter()
    (out/'progress.json').write_text(json.dumps({'active_workers':workers,'completed_trials':trials},indent=2))
    with (trial/'cli.log').open('w') as log:
        process=subprocess.Popen(command,env=env,cwd=kit,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        (trial/'pid').write_text(str(process.pid))
        with (trial/'gpu-samples.jsonl').open('w') as gpu:
            while process.poll() is None:
                sample=capture(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'])
                gpu.write(json.dumps({'elapsed_s':time.perf_counter()-started,'memory_mib,utilization_pct':sample})+'\n')
                gpu.flush()
                time.sleep(1)
    summary={'workers':workers,'exit_code':process.returncode,'total_wall_s':time.perf_counter()-started}
    result=trial/'evaluation/result.json'
    if result.exists():
        data=json.loads(result.read_text())
        summary.update(result=data.get('result'),phase_timings=data.get('phase_timings'),
                       resources=data.get('resources'),record_success=data.get('record_success'))
    trials.append(summary)
    (out/'progress.json').write_text(json.dumps({'active_workers':None,'completed_trials':trials},indent=2))
    (out/'summary.json').write_text(json.dumps(trials,indent=2))
    if process.returncode:
        raise SystemExit(f'K={workers} failed; inspect cli.log; no later trials started')
(out/'finished.json').write_text(json.dumps({'finished_utc':datetime.now(timezone.utc).isoformat(),'trials':len(trials)}))
