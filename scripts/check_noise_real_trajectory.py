"""Gate full policy validation on paired real-environment fixed-action traces."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np


def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def main():
    import libero.libero.envs.env_wrapper as wrapper
    from libero.libero import benchmark
    from roboscope_kit.runtime.benchmarks.libero.worker import create_task_runtime
    from noise_acceleration import install_noise_acceleration,rng_hash

    root=Path(__file__).resolve().parent
    sha=hashlib.sha256(Path(wrapper.__file__).read_bytes()).hexdigest()
    metadata=json.loads(Path('/root/prp/libero-plus-runtime/LIBERO-plus/libero/libero/benchmark/task_classification.json').read_text())['libero_10']
    suite=benchmark.get_benchmark_dict()['libero_10']()
    records=[]
    for noise in (8,18,28,38,48):
        task=next(i for i,r in enumerate(metadata) if r['category']=='Sensor Noise' and r['name'].endswith(f'_noise_{noise}'))
        runtime=create_task_runtime(suite,task,7)
        runs={}
        try:
            for mode in ('original','lazy'):
                handle=install_noise_acceleration(wrapper,expected_source_sha256=sha,mode=mode)
                handle.context(f'task-{task:04d}-episode-0000')
                try:
                    random.seed(7);runtime.env.seed(7);np.random.seed(7)
                    runtime.env.reset();observation=runtime.env.set_init_state(runtime.initial_states[0])
                    states=[];images=[]
                    for step in range(16):
                        observation,_,success,_=runtime.env.step(np.asarray([0.]*6+[-1.],dtype=np.float32))
                        states.append({'physics':digest(runtime.env.env.sim.get_state().flatten()),'rng':rng_hash(),'success':bool(success)})
                        if step in (9,14):
                            images.append({'primary':digest(np.asarray(observation['agentview_image'],dtype=np.uint8)),
                                           'wrist':digest(observation['robot0_eye_in_hand_image'])})
                            assert rng_hash()==states[-1]['rng']
                    runs[mode]={'states':states,'images':images,'stats':dict(handle.stats)}
                finally:handle.restore()
        finally:runtime.env.close()
        record={'noise':noise,'task_id':task,'states_equal':runs['original']['states']==runs['lazy']['states'],
                'images_equal':runs['original']['images']==runs['lazy']['images'],'runs':runs}
        records.append(record)
        (root/'real-trajectory-check.json').write_text(json.dumps(records,indent=2))
        assert record['states_equal'] and record['images_equal'],record
        print('REAL_TRAJECTORY_OK',noise,flush=True)
    print('ALL_REAL_TRAJECTORIES_OK',flush=True)


if __name__=='__main__':main()
