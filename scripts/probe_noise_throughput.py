"""CPU-only comparison of exact consumed-frame noise and worker concurrency."""
import ast
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import probe_noise_acceleration as probe

SCOPE=None
IMAGE=None


def initialize(ready):
    global SCOPE,IMAGE
    definitions=[n for n in ast.parse(probe.SOURCE.read_text()).body
                 if isinstance(n,ast.FunctionDef) or isinstance(n,ast.ClassDef) and n.name=='MotionImage']
    SCOPE=dict(vars(probe))
    probe.wandlibrary.MagickMotionBlurImage.argtypes=(probe.ctypes.c_void_p,probe.ctypes.c_double,probe.ctypes.c_double,probe.ctypes.c_double)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(probe.SOURCE),'exec'),SCOPE)
    IMAGE=Image.fromarray(np.random.RandomState(314).randint(0,256,size=(256,256,3),dtype=np.uint8))
    probe.fast_glass(IMAGE,1)
    SCOPE['motion_blur'](IMAGE,1)
    ready.put(os.getpid())


def advance_rng(group,severity):
    if group==0:
        np.random.uniform(-45,45)
    elif group==3:
        decay=[3,2.8,2.5,2.2,2.,1.8,1.6,1.5,1.4,1.3][severity-1]
        SCOPE['plasma_fractal'](wibbledecay=decay)
    elif group==4:
        _,delta,iterations=probe.PARAMETERS[severity-1]
        count=iterations*(256-2*delta)*(256-2*delta)
        np.random.randint(-delta,delta,size=(count,2))


def job(args):
    noise,mode=args
    group,severity=(noise-1)//10,(noise-1)%10+1
    function=SCOPE[['motion_blur','gaussian_blur','zoom_blur','fog','glass_blur'][group]]
    if mode!='original' and group==4:function=probe.fast_glass
    np.random.seed(700+noise)
    states=[];output=None
    for frame in range(5):
        # Simulate intervening random draws by the rest of an environment.
        np.random.uniform(size=3)
        if mode=='on_demand' and frame!=4:
            advance_rng(group,severity)
        else:
            output=np.asarray(function(IMAGE,severity),dtype=np.uint8)
        state=np.random.get_state()
        states.append(hashlib.sha256(state[1].tobytes()+repr(state[2:]).encode()).hexdigest())
    return {'noise':noise,'consumed_image_sha256':hashlib.sha256(output.tobytes()).hexdigest(),
            'rng_after_each_frame':states}


def main():
    root=Path(__file__).resolve().parent
    context=mp.get_context('spawn')
    results=[];reference=None
    for mode,workers in [('original',4),('fast_glass',4),('fast_glass',8),('on_demand',4),('on_demand',8)]:
        ready=context.Queue();startup=time.perf_counter()
        with context.Pool(workers,initializer=initialize,initargs=(ready,)) as pool:
            pids=[ready.get(timeout=180) for _ in range(workers)]
            startup_s=time.perf_counter()-startup
            start=time.perf_counter()
            rows=pool.map(job,[(n,mode) for n in range(1,51)],chunksize=1)
            elapsed=time.perf_counter()-start
        ready.close();ready.join_thread()
        if reference is None:reference=rows
        assert rows==reference,(mode,workers)
        result={'mode':mode,'workers':workers,'startup_and_warmup_s':startup_s,
                'work_s':elapsed,'generated_frames':250 if mode!='on_demand' else 50,
                'consumed_frames':50,'all_consumed_images_and_rng_states_equal':True,'records':rows}
        results.append(result)
        (root/'throughput-progress.json').write_text(json.dumps(results,indent=2))
        print(mode,workers,'work_s',round(elapsed,3),'startup_s',round(startup_s,3),flush=True)
    report={'scope':'synthetic fixed images, all 50 noise variants, five frames per variant, only fifth frame consumed',
            'source_sha256':hashlib.sha256(probe.SOURCE.read_bytes()).hexdigest(),
            'simulation_or_model_run':False,'runs':results,
            'warning':'On-demand computation is a research prototype, not integrated into the native environment/worker.'}
    (root/'throughput-report.json').write_text(json.dumps(report,indent=2))
    print('THROUGHPUT_CHECK_OK',flush=True)


if __name__=='__main__':main()
