"""Same-observation FlowRes ranking with an explicit low-gap full-step fallback."""

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models.model import Observation
from roboscope_kit import ActionChunk, run

from candidate_selection import select_candidate
from pi05_candidate_policy import Pi05CandidatePolicy
from shared_prefix_sampler import create_shared_prefix_sampler


class Pi05ScoredCandidatePolicy(Pi05CandidatePolicy):
    def load(self, context):
        capabilities = super().load(context)
        self.scored_sampler = create_shared_prefix_sampler(self.policy._model, with_scores=True)
        self.min_gap = float(os.environ.get("PRP_MIN_SCORE_GAP", ".05"))
        limit = os.environ.get("PRP_MAX_FLOW_RESIDUAL")
        self.max_residual = float(limit) if limit else None
        self.generation_seen = False
        self.fallback_seen = False
        select_candidate([1., 2.], min_gap=self.min_gap, max_residual=self.max_residual)
        return capabilities

    def act(self, observation):
        started = time.perf_counter()
        model_input = self.model_input(observation)
        inputs = self.policy._input_transform(jax.tree.map(lambda value: value, model_input))
        inputs = jax.tree.map(lambda value: jnp.asarray(value)[None,...], inputs)
        episode_key = f"{self.seed}:{observation.episode_id}"
        episode_seed = int.from_bytes(hashlib.sha256(episode_key.encode()).digest()[:8],"big")
        replan_idx = observation.step//self.action_chunk_size
        seeds = [int(np.random.SeedSequence([episode_seed,replan_idx,index]).generate_state(1,dtype=np.uint64)[0]) for index in range(3)]
        noises = [np.random.default_rng(seed).standard_normal(self.noise_shape,dtype=np.float32) for seed in seeds]
        preprocess_ms = (time.perf_counter()-started)*1000
        sampling_start = time.perf_counter()
        cold_generation = not self.generation_seen
        outputs,scores = self.scored_sampler(Observation.from_dict(inputs),jnp.asarray(np.stack(noises[:2])),
                                            num_steps=4,verification_noise=jnp.asarray(noises[2]))
        jax.block_until_ready((outputs,scores))
        outputs,scores = np.asarray(outputs),np.asarray(scores)
        generation_verification_ms = (time.perf_counter()-sampling_start)*1000
        self.generation_seen = True
        prefixes=[]
        for output in outputs:
            data=self.policy._output_transform({"state":np.asarray(inputs["state"][0]),"actions":output})
            actions=np.asarray(data["actions"],dtype=np.float32)
            if actions.shape!=(10,7) or not np.isfinite(actions).all():
                raise ValueError("Invalid scored candidate action")
            prefixes.append(np.ascontiguousarray(actions[:5]))
        decision=select_candidate(scores,min_gap=self.min_gap,max_residual=self.max_residual)
        fallback_ms=0.
        cold_fallback = False
        if decision["fallback"]:
            cold_fallback = not self.fallback_seen
            fallback_start=time.perf_counter()
            previous=self.policy._sample_kwargs
            try:
                self.policy._sample_kwargs={"num_steps":10}
                chosen=self.action_prefix(model_input,noise=noises[0])
            finally:
                self.policy._sample_kwargs=previous
            fallback_ms=(time.perf_counter()-fallback_start)*1000
            self.fallback_seen = True
        else:
            chosen=prefixes[decision["selected_candidate_idx"]]
        record={"episode_id":observation.episode_id,"step":observation.step,"replan_idx":replan_idx,
                "episode_seed":episode_seed,"num_steps":4,"verification_time":.5,
                "verification_noise_seed":seeds[2],"score_kind":"mean_flow_residual_first5_dims7",
                "min_gap":self.min_gap,"max_residual":self.max_residual,**decision,
                "execution_source":"fallback_10" if decision["fallback"] else "candidate",
                "preprocess_ms":preprocess_ms,"generation_verification_ms":generation_verification_ms,
                "fallback_ms":fallback_ms,"policy_ms":(time.perf_counter()-started)*1000,
                "cold_generation":cold_generation,"cold_fallback":cold_fallback,
                "candidates":[{"candidate_idx":i,"noise_seed":seeds[i],
                               "score":float(scores[i]) if np.isfinite(scores[i]) else None,
                               "action_prefix":prefix.tolist()} for i,prefix in enumerate(prefixes)],
                "executed_action_prefix":chosen.tolist()}
        with self.log_path.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(record,allow_nan=False)+"\n")
        self.last_record = record
        return ActionChunk(chosen)


if __name__=="__main__":
    run(Pi05ScoredCandidatePolicy(log_path=Path(__file__).with_name("candidates.jsonl"),seed=int(sys.argv[1])))
