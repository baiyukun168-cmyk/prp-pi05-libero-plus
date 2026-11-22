"""Check candidate noise function on real simulator images in an isolated process."""
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
from probe_noise_acceleration import fast_glass, same_state, SOURCE


def main():
    import libero.libero.envs.env_wrapper as wrapper
    from libero.libero import benchmark
    from roboscope_kit.runtime.benchmarks.libero.worker import create_task_runtime

    root=Path(__file__).resolve().parent
    before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    original=wrapper.glass_blur
    rows=[]
    fast_glass(Image.fromarray(np.zeros((16,16,3),np.uint8)),1)

    def checked(image, severity=1):
        state=np.random.get_state()
        start=time.perf_counter();a=original(image,severity);original_ms=(time.perf_counter()-start)*1000
        expected_state=np.random.get_state()
        np.random.set_state(state)
        start=time.perf_counter();b=fast_glass(image,severity);fast_ms=(time.perf_counter()-start)*1000
        row={'severity':severity,'shape':list(np.asarray(image).shape),
             'pixels_equal':bool(np.array_equal(a,b)),
             'rng_state_equal':same_state(expected_state,np.random.get_state()),
             'original_ms':original_ms,'fast_ms':fast_ms}
        rows.append(row)
        assert row['pixels_equal'] and row['rng_state_equal'],row
        return b

    wrapper.glass_blur=checked
    metadata=json.loads(Path('/root/prp/libero-plus-runtime/LIBERO-plus/libero/libero/benchmark/task_classification.json').read_text())['libero_10']
    suite=benchmark.get_benchmark_dict()['libero_10']()
    tasks=[]
    try:
        for noise in (41,45,50):
            task_id=next(i for i,r in enumerate(metadata) if r['category']=='Sensor Noise' and r['name'].endswith(f'_noise_{noise}'))
            random.seed(7);np.random.seed(7)
            runtime=create_task_runtime(suite,task_id,7)
            try:
                assert runtime.env.noise==noise
                runtime.env.reset()
                runtime.env.set_init_state(runtime.initial_states[0])
                for _ in range(6):
                    runtime.env.step(np.asarray([0.]*6+[-1.],dtype=np.float32))
                tasks.append({'task_id':task_id,'noise':noise,'steps':6})
            finally:
                runtime.env.close()
    finally:
        wrapper.glass_blur=original
    assert before==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    report={'kind':'three simulator tasks with fixed dummy actions; no policy or full episodes',
            'tasks':tasks,'calls':rows,'all_exact':True,'upstream_file_unchanged':True}
    (root/'environment-check.json').write_text(json.dumps(report,indent=2))
    print('ENV_CHECK_OK',len(rows),'calls',len(tasks),'tasks',flush=True)


if __name__=='__main__':main()
