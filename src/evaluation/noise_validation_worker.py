"""Native Kit worker with opt-in noise adapter and comparison audit."""
import hashlib
import os
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    import libero.libero.envs.env_wrapper as wrapper
    from noise_acceleration import install_noise_acceleration
    from roboscope_kit.runtime.benchmarks.libero import worker

    directory=Path(os.environ['PRP_NOISE_AUDIT_DIR']);directory.mkdir(exist_ok=True)
    handle=install_noise_acceleration(wrapper,
        expected_source_sha256=os.environ['PRP_NOISE_SOURCE_SHA256'],
        mode=os.environ['PRP_NOISE_MODE'],audit_path=directory/f'worker-{os.getpid()}.jsonl')
    original_episode=worker.run_episode;original_observation=worker.build_observation

    def observation(*args,**kwargs):
        value=original_observation(*args,**kwargs)
        handle.log('observation',step=value.step,
            primary=digest(value.features['images.primary']),
            wrist=digest(value.features['images.wrist']),
            state=digest(value.features['state.proprioception']))
        return value

    def episode(runtime,policy,index,**kwargs):
        handle.context(f'task-{runtime.task_id:04d}-episode-{index:04d}')
        original_step=runtime.env.step;count=0
        def step(action):
            nonlocal count
            result=original_step(action)
            handle.log('step',index=count,action=digest(action),
                       physics=digest(runtime.env.env.sim.get_state().flatten()),success=bool(result[2]))
            count+=1
            return result
        runtime.env.step=step
        try:return original_episode(runtime,policy,index,**kwargs)
        finally:runtime.env.step=original_step

    worker.run_episode=episode;worker.build_observation=observation
    try:worker.main()
    finally:handle.restore()


if __name__=='__main__':main()
