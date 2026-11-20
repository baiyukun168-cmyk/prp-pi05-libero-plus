"""SDK entry point for the existing scored Policy, without changing its algorithm."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from roboscope_kit import BenchmarkName, run
from pi05_scored_candidate_policy import Pi05ScoredCandidatePolicy


class KitScoredPolicy(Pi05ScoredCandidatePolicy):
    def load(self, context):
        start = time.perf_counter()
        capabilities = super().load(context)
        load_s = time.perf_counter() - start
        trial = Path(os.environ['SCORED_TRIAL_DIR'])
        self.timing = (trial/'policy-timing.jsonl').open('a', buffering=1)
        self.action_hashes = {}
        # Compile generation/scoring and full-step fallback before scored episodes.
        self.log_path = trial/'warmup-selection.jsonl'
        state = np.zeros(16, dtype=np.float32)
        state[6] = 1.
        dummy = SimpleNamespace(episode_id='warmup-only', step=0, features={
            'images.primary': np.zeros((256,256,3), dtype=np.uint8),
            'images.wrist': np.zeros((256,256,3), dtype=np.uint8),
            'state.proprioception': state, 'language.instruction': 'warmup',
        })
        start = time.perf_counter()
        for index in range(2):
            dummy.step = index*5
            super().act(dummy)
        model_input = self.model_input(dummy)
        previous = self.policy._sample_kwargs
        try:
            self.policy._sample_kwargs = {'num_steps': 10}
            self.action_prefix(model_input, noise=np.zeros(self.noise_shape, dtype=np.float32))
        finally:
            self.policy._sample_kwargs = previous
        self.fallback_seen = True
        self.log_path = trial/'selection.jsonl'
        self.timing.write(json.dumps({'type':'load','load_s':load_s,
            'warmup_s':time.perf_counter()-start,'min_gap':self.min_gap,
            'max_residual':self.max_residual})+'\n')
        # LIBERO and LIBERO-Plus share the same canonical observation/action contract.
        b = BenchmarkName.LIBERO.value
        return replace(capabilities, supported_benchmarks=(b,), supported_suites={b:('libero_10',)})

    def reset(self, context):
        self.action_hashes[context.episode_id] = hashlib.sha256()

    def act(self, observation):
        start = time.perf_counter()
        action = super().act(observation)
        wall_ms = (time.perf_counter()-start)*1000.
        self.action_hashes[observation.episode_id].update(action.actions.tobytes())
        r = self.last_record
        self.timing.write(json.dumps({'type':'infer','episode_id':observation.episode_id,
            'step':observation.step,'policy_ms':r['policy_ms'],'act_wall_ms':wall_ms,
            'preprocess_ms':r['preprocess_ms'],'generation_verification_ms':r['generation_verification_ms'],
            'fallback_ms':r['fallback_ms'],'fallback':r['fallback'],
            'selected_candidate_idx':r['selected_candidate_idx'],
            'cold_generation':r['cold_generation'],'cold_fallback':r['cold_fallback']})+'\n')
        return action

    def end_episode(self, episode_id):
        digest = self.action_hashes.pop(episode_id, None)
        if digest is not None:
            self.timing.write(json.dumps({'type':'episode_end','episode_id':episode_id,
                'action_sha256':digest.hexdigest()})+'\n')


if __name__ == '__main__':
    run(KitScoredPolicy(log_path=Path(os.environ['SCORED_TRIAL_DIR'])/'selection.jsonl', seed=7))
