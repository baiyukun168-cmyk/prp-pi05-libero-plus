"""Isolated CPU experiments; never patch the simulator or start evaluation."""
import ast
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import statistics
import sys
import time
import ctypes

import cv2
import numba
from numba import njit
import numpy as np
from PIL import Image
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian
from wand.api import library as wandlibrary
from wand.image import Image as WandImage
from wand.resource import limits

SOURCE = Path('/root/prp/libero-plus-runtime/LIBERO-plus/libero/libero/envs/env_wrapper.py')
PARAMETERS = [(0.5,1,3),(0.7,1,3),(0.9,2,3),(1.,2,2),(1.1,3,2),
              (1.3,3,2),(1.5,4,2),(1.8,4,2),(2.2,5,1),(2.5,5,1)]


@njit(cache=False)
def shuffle(x, delta, iterations, offsets):
    """Preserve the upstream NumPy view-assignment behavior, including aliasing."""
    index = 0
    for _ in range(iterations):
        for h in range(x.shape[0]-delta, delta, -1):
            for w in range(x.shape[1]-delta, delta, -1):
                dx, dy = offsets[index, 0], offsets[index, 1]
                index += 1
                for channel in range(x.shape[2]):
                    x[h, w, channel] = x[h+dy, w+dx, channel]
                # In the original tuple assignment the second RHS is a view
                # of the already overwritten pixel, so the second copy is a no-op.
    return x


def fast_glass(image, severity):
    sigma, delta, iterations = PARAMETERS[severity-1]
    x = np.uint8(gaussian(np.array(image)/255., sigma=sigma, channel_axis=-1)*255)
    count = iterations * max(0,x.shape[0]-2*delta) * max(0,x.shape[1]-2*delta)
    offsets = np.random.randint(-delta, delta, size=(count,2))
    x = shuffle(x, delta, iterations, offsets)
    return np.clip(gaussian(x/255., sigma=sigma, channel_axis=-1),0,1)*255


def same_state(a, b):
    return a[0] == b[0] and np.array_equal(a[1],b[1]) and a[2:] == b[2:]


def run(function, image, severity, seed):
    np.random.seed(seed)
    start = time.perf_counter()
    value = function(image, severity)
    elapsed = (time.perf_counter()-start)*1000
    return value, np.random.get_state(), elapsed


def main():
    out = Path(sys.argv[1]);out.mkdir(exist_ok=False)
    tree = ast.parse(SOURCE.read_text())
    definitions = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
                   and n.name in ('MotionImage','motion_blur','glass_blur')]
    assert len(definitions)==3
    scope = dict(globals())
    wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,ctypes.c_double,ctypes.c_double,ctypes.c_double)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(SOURCE),'exec'),scope)
    original_glass, motion = scope['glass_blur'],scope['motion_blur']
    # Compile separately from all steady-state timings.
    start=time.perf_counter();fast_glass(Image.fromarray(np.zeros((16,16,3),np.uint8)),1)
    compile_s=time.perf_counter()-start
    cases=[]
    generator=np.random.RandomState(314)
    for size in (32,128,256):
        image=Image.fromarray(generator.randint(0,256,size=(size,size,3),dtype=np.uint8))
        for severity in range(1,11):
            for seed in (7,17):
                a,sa,ta=run(original_glass,image,severity,seed)
                b,sb,tb=run(fast_glass,image,severity,seed)
                row={'size':size,'severity':severity,'seed':seed,'pixels_equal':bool(np.array_equal(a,b)),
                     'rng_state_equal':same_state(sa,sb),'original_ms':ta,'fast_ms':tb,'speedup':ta/tb,
                     'max_abs_error':float(np.max(np.abs(a-b)))}
                cases.append(row)
                assert row['pixels_equal'] and row['rng_state_equal'],row
        print('GLASS_SIZE_DONE',size,flush=True)
    # Mixed severities in sequence exercise RNG continuity across function calls.
    image=Image.fromarray(generator.randint(0,256,size=(128,128,3),dtype=np.uint8))
    np.random.seed(91);expected=[original_glass(image,s) for s in (1,10,5,2,9)];sa=np.random.get_state()
    np.random.seed(91);actual=[fast_glass(image,s) for s in (1,10,5,2,9)];sb=np.random.get_state()
    sequence_match=all(np.array_equal(a,b) for a,b in zip(expected,actual)) and same_state(sa,sb)
    assert sequence_match
    motion_cases=[]
    image=Image.fromarray(generator.randint(0,256,size=(256,256,3),dtype=np.uint8))
    original_threads=limits['thread']
    try:
        for severity in (1,5,10):
            for seed in (7,17):
                limits['thread']=1;a,sa,ta=run(motion,image,severity,seed)
                for threads in (2,4):
                    limits['thread']=threads;b,sb,tb=run(motion,image,severity,seed)
                    motion_cases.append({'severity':severity,'seed':seed,'requested_threads':threads,
                        'actual_limit':limits['thread'],'pixels_equal':bool(np.array_equal(a,b)),
                        'rng_state_equal':same_state(sa,sb),'single_ms':ta,'multi_ms':tb,'speedup':ta/tb})
            print('MOTION_SEVERITY_DONE',severity,flush=True)
    finally:
        limits['thread']=original_threads
    report={'utc':datetime.now(timezone.utc).isoformat(),'kind':'isolated CPU microbenchmark; synthetic RGB images',
        'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'versions':{'numpy':np.__version__,'numba':numba.__version__},'compile_s':compile_s,
        'glass_cases':cases,'glass_sequence_match':sequence_match,'motion_cases':motion_cases,
        'scope':'No model, simulator episode, production patch or full benchmark was run.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    for size in (32,128,256):
        rows=[r for r in cases if r['size']==size]
        print('GLASS',size,'cases',len(rows),'mean_original_ms',statistics.mean(r['original_ms'] for r in rows),
              'mean_fast_ms',statistics.mean(r['fast_ms'] for r in rows),'all_exact',True,flush=True)
    for threads in (2,4):
        rows=[r for r in motion_cases if r['requested_threads']==threads]
        print('MOTION',threads,'median_speedup',statistics.median(r['speedup'] for r in rows),
              'all_exact',all(r['pixels_equal'] and r['rng_state_equal'] for r in rows),flush=True)
    print('REPORT',out/'report.json',flush=True)


if __name__=='__main__':main()
