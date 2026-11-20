"""Three-arm diagnostic with official samplers and unchanged scored snapshot."""
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from roboscope_kit import ActionChunk, run
from pi05_scored_candidate_policy import Pi05ScoredCandidatePolicy


def candidate_a_noise(seed, episode_id, step, shape):
    episode_seed = int.from_bytes(hashlib.sha256(f'{seed}:{episode_id}'.encode()).digest()[:8], 'big')
    noise_seed = int(np.random.SeedSequence([episode_seed, step//5, 0]).generate_state(1,dtype=np.uint64)[0])
    return noise_seed, np.random.default_rng(noise_seed).standard_normal(shape,dtype=np.float32)


class AblationPolicy(Pi05ScoredCandidatePolicy):
    def load(self, context):
        self.mode = os.environ['ABLATION_MODE']
        assert self.mode in ('ten','four','scored')
        self.directory = Path(os.environ['ABLATION_TRIAL_DIR'])
        self.prompts = json.loads(Path(os.environ['PROMPT_SELECTION']).read_text())['prompts']
        self.initial_seen = set()
        self.hashes = {}
        self.initial_log = (self.directory/'initial-inputs.jsonl').open('a',buffering=1)
        self.timing_log = (self.directory/'timing.jsonl').open('a',buffering=1)
        self.warming = True
        started = time.perf_counter()
        capabilities = super().load(context)
        load_s = time.perf_counter()-started
        self.policy._sample_kwargs = {'num_steps':10 if self.mode=='ten' else 4}
        self.log_path = self.directory/'warmup-selection.jsonl'
        state = np.zeros(16,dtype=np.float32)
        state[6] = 1.
        dummy = SimpleNamespace(episode_id='warmup-only',step=0,features={
            'images.primary':np.zeros((256,256,3),dtype=np.uint8),
            'images.wrist':np.zeros((256,256,3),dtype=np.uint8),
            'state.proprioception':state,'language.instruction':'warmup'})
        started = time.perf_counter()
        for index in range(2):
            dummy.step=index*5
            self.act(dummy)
        if self.mode=='scored':
            previous = self.policy._sample_kwargs
            try:
                self.policy._sample_kwargs={'num_steps':10}
                self.action_prefix(self.model_input(dummy),noise=np.zeros(self.noise_shape,dtype=np.float32))
            finally:
                self.policy._sample_kwargs=previous
            self.fallback_seen=True
        self.log_path=self.directory/'selection.jsonl'
        self.warming=False
        self.timing_log.write(json.dumps({'type':'load','mode':self.mode,'seed':self.seed,
            'load_s':load_s,'warmup_s':time.perf_counter()-started})+'\n')
        return capabilities

    def model_input(self, observation):
        element=super().model_input(observation)
        if observation.episode_id in self.prompts:
            element['prompt']=self.prompts[observation.episode_id]
            if observation.episode_id not in self.initial_seen:
                fingerprints={}
                for key in ('observation/image','observation/wrist_image','observation/state'):
                    value=np.ascontiguousarray(element[key])
                    fingerprints[key]={'shape':list(value.shape),'dtype':str(value.dtype),
                        'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
                self.initial_log.write(json.dumps({'episode_id':observation.episode_id,
                    'step':observation.step,'prompt':element['prompt'],'fingerprints':fingerprints})+'\n')
                self.initial_seen.add(observation.episode_id)
        return element

    def reset(self, context):
        self.hashes[context.episode_id]=hashlib.sha256()

    def act(self, observation):
        start=time.perf_counter()
        noise_seed,noise=candidate_a_noise(self.seed,observation.episode_id,observation.step,self.noise_shape)
        if self.mode=='scored':
            action=super().act(observation)
            record=self.last_record
            # Cross-check against the archived scored algorithm's independently derived seed.
            if record['candidates'][0]['noise_seed']!=noise_seed:
                raise RuntimeError('Candidate A noise is not paired with official sampler arms')
            candidate_a=record['candidates'][0]['action_prefix']
            details={'fallback':record['fallback'],'selected_candidate_idx':record['selected_candidate_idx'],
                'preprocess_ms':record['preprocess_ms'],'generation_verification_ms':record['generation_verification_ms'],
                'fallback_ms':record['fallback_ms'],'cold_generation':record['cold_generation'],
                'cold_fallback':record['cold_fallback']}
        else:
            prepared=time.perf_counter()
            model_input=self.model_input(observation)
            prediction=time.perf_counter()
            actions=self.action_prefix(model_input,noise=noise)
            action=ActionChunk(actions)
            candidate_a=actions.tolist()
            details={'fallback':False,'selected_candidate_idx':0,
                'preprocess_ms':(prediction-prepared)*1000.,
                'predict_ms':(time.perf_counter()-prediction)*1000.}
        elapsed_ms=(time.perf_counter()-start)*1000.
        if not self.warming:
            self.hashes[observation.episode_id].update(action.actions.tobytes())
            row={'type':'infer','mode':self.mode,'seed':self.seed,'episode_id':observation.episode_id,
                'step':observation.step,'noise_seed':noise_seed,'act_wall_ms':elapsed_ms,**details}
            if observation.step==0:
                row['noise_a_sha256']=hashlib.sha256(noise.tobytes()).hexdigest()
                row['candidate_a_prefix']=candidate_a
                row['executed_prefix']=action.actions.tolist()
            self.timing_log.write(json.dumps(row)+'\n')
        return action

    def end_episode(self, episode_id):
        digest=self.hashes.pop(episode_id,None)
        if digest is not None:
            self.timing_log.write(json.dumps({'type':'episode_end','episode_id':episode_id,
                'action_sha256':digest.hexdigest()})+'\n')


if __name__=='__main__':
    run(AblationPolicy(log_path=Path(os.environ['ABLATION_TRIAL_DIR'])/'selection.jsonl',
        seed=int(os.environ['ABLATION_NOISE_SEED'])))
