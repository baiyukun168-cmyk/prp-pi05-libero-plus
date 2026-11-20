"""Official model sampler; vary only generation steps and executed chunk size."""
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from roboscope_kit import ActionChunk,run
from policy import Pi05LiberoPlusPolicy
from feedback_noise import paired_noise


class FeedbackPolicy(Pi05LiberoPlusPolicy):
    def load(self,context):
        self.generation_steps=int(os.environ['GENERATION_STEPS'])
        self.action_chunk_size=int(os.environ['EXECUTION_STEPS'])
        self.noise_seed=int(os.environ['ACTION_NOISE_SEED'])
        assert self.generation_steps in (8,10) and self.action_chunk_size in (3,5)
        self.sample_kwargs={'num_steps':self.generation_steps}
        self.directory=Path(os.environ['FEEDBACK_TRIAL_DIR'])
        self.prompts=json.loads(Path(os.environ['PROMPT_SELECTION']).read_text())['prompts']
        self.initial_seen=set();self.hashes={};self.warming=True
        self.initial_log=(self.directory/'initial-inputs.jsonl').open('a',buffering=1)
        self.timing=(self.directory/'timing.jsonl').open('a',buffering=1)
        start=time.perf_counter()
        capabilities=super().load(context)
        load_s=time.perf_counter()-start
        state=np.zeros(16,dtype=np.float32);state[6]=1.
        dummy=SimpleNamespace(episode_id='warmup-only',step=0,features={
            'images.primary':np.zeros((256,256,3),np.uint8),
            'images.wrist':np.zeros((256,256,3),np.uint8),
            'state.proprioception':state,'language.instruction':'warmup'})
        start=time.perf_counter()
        for index in range(2):
            dummy.step=index*self.action_chunk_size
            self.act(dummy)
        self.warming=False
        self.timing.write(json.dumps({'type':'load','load_s':load_s,'warmup_s':time.perf_counter()-start,
            'generation_steps':self.generation_steps,'execution_steps':self.action_chunk_size,
            'noise_seed':self.noise_seed})+'\n')
        return capabilities

    def model_input(self,observation):
        element=super().model_input(observation)
        if observation.episode_id in self.prompts:
            element['prompt']=self.prompts[observation.episode_id]
            if observation.episode_id not in self.initial_seen:
                fingerprints={}
                for key in ('observation/image','observation/wrist_image','observation/state'):
                    value=np.ascontiguousarray(element[key])
                    fingerprints[key]={'shape':list(value.shape),'dtype':str(value.dtype),
                        'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
                self.initial_log.write(json.dumps({'episode_id':observation.episode_id,'step':observation.step,
                    'prompt':element['prompt'],'fingerprints':fingerprints})+'\n')
                self.initial_seen.add(observation.episode_id)
        return element

    def reset(self,context):
        self.hashes[context.episode_id]=hashlib.sha256()

    def act(self,observation):
        start=time.perf_counter()
        noise_seed,noise=paired_noise(self.noise_seed,observation.episode_id,observation.step,self.noise_shape)
        assert observation.step%self.action_chunk_size==0
        element=self.model_input(observation)
        before_prediction=time.perf_counter()
        actions=self.action_prefix(element,noise=noise)
        action=ActionChunk(actions)
        elapsed=(time.perf_counter()-start)*1000.
        if not self.warming:
            self.hashes[observation.episode_id].update(actions.tobytes())
            row={'type':'infer','episode_id':observation.episode_id,'step':observation.step,
                'generation_steps':self.generation_steps,'execution_steps':self.action_chunk_size,
                'noise_round':self.noise_seed,'noise_seed':noise_seed,'act_wall_ms':elapsed,
                'predict_ms':(time.perf_counter()-before_prediction)*1000.}
            if observation.step==0:
                row['noise_sha256']=hashlib.sha256(noise.tobytes()).hexdigest()
                row['first_actions']=actions.tolist()
            self.timing.write(json.dumps(row)+'\n')
        return action

    def end_episode(self,episode_id):
        digest=self.hashes.pop(episode_id,None)
        if digest is not None:self.timing.write(json.dumps({'type':'episode_end','episode_id':episode_id,
            'action_sha256':digest.hexdigest()})+'\n')


if __name__=='__main__':run(FeedbackPolicy())
