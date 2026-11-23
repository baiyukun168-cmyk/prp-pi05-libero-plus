"""Version-pinned OpenPI adapter sharing one encoded observation across candidates."""

import einops
from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import model as model_utils
from openpi.models.pi0 import make_attn_mask


def make_sampler(model):
    graph, weights = nnx.split(model)

    @jax.jit
    def sample(weights, observation, noise):
        resident = nnx.merge(graph, weights)
        observation = model_utils.preprocess_observation(None, observation, train=False)
        prefix, prefix_mask, prefix_ar = resident.embed_prefix(observation)
        positions = jnp.cumsum(prefix_mask, axis=1)-1
        _, cache = resident.PaliGemma.llm([prefix, None], mask=make_attn_mask(prefix_mask,prefix_ar), positions=positions)
        count = 2
        # Gemma scans layers on axis 0; its KV batch dimension is axis 1.
        cache = jax.tree.map(lambda value: jnp.repeat(value,count,axis=1), cache)
        observation = jax.tree.map(lambda value: jnp.repeat(value,count,axis=0), observation)
        prefix_mask = jnp.repeat(prefix_mask,count,axis=0)
        dt = -0.1

        def velocity(actions, time):
            suffix, mask, ar, cond = resident.embed_suffix(observation, actions, jnp.broadcast_to(time,count))
            prefix_attention = einops.repeat(prefix_mask,"b p -> b s p",s=suffix.shape[1])
            attention = jnp.concatenate([prefix_attention,make_attn_mask(mask,ar)],axis=-1)
            suffix_positions = jnp.sum(prefix_mask,axis=-1)[:,None]+jnp.cumsum(mask,axis=-1)-1
            (_, suffix_out), _ = resident.PaliGemma.llm([None,suffix], mask=attention,
                                                       positions=suffix_positions,kv_cache=cache,
                                                       adarms_cond=[None,cond])
            return resident.action_out_proj(suffix_out[:,-resident.action_horizon:])

        x0 = noise[0]
        def step(carry):
            i, x, previous, accepted, attempts = carry
            draft = x + dt * previous
            t = 1.0 + dt * i
            v = velocity(jnp.stack([x, draft]), jnp.stack([t, jnp.maximum(t+dt, 0.)]))
            corrected = x + dt * v[0]
            eligible = (i > 0) & (i < 9)
            error = jnp.sqrt(jnp.mean(jnp.square(corrected-draft)))
            take = eligible & (error <= 0.02)
            return (i+1+take.astype(int), jnp.where(take, draft+dt*v[1], corrected),
                    jnp.where(take, v[1], v[0]), accepted+take, attempts+eligible)
        _, x, _, accepted, attempts = jax.lax.while_loop(lambda c: c[0]<10, step,
            (0,x0,jnp.zeros_like(x0,dtype=jnp.bfloat16),0,0))
        return x, accepted, attempts
    return lambda obs, noise: sample(weights, obs, noise)

from feedback_policy import FeedbackPolicy
from roboscope_kit import run
import numpy as np
import json

class SpecPolicy(FeedbackPolicy):
    def action_prefix(self, model_input, *, noise=None):
        if not hasattr(self, 'spec'):
            self.spec = make_sampler(self.policy._model)
        inp = self.policy._input_transform(dict(model_input))
        inp = jax.tree.map(lambda a: jnp.asarray(a)[None,...], inp)
        action, accepted, attempts = jax.device_get(self.spec(model_utils.Observation.from_dict(inp), jnp.asarray(noise)[None,...]))
        out = self.policy._output_transform({'state': np.asarray(inp['state'][0]),'actions':action})
        a = np.asarray(out['actions'],dtype=np.float32)
        assert a.shape==(10,7) and np.isfinite(a).all()
        if not self.warming:
            self.timing.write(json.dumps({'type':'spec','accepted':int(accepted),'attempts':int(attempts),'threshold':0.02})+'\n')
        return np.ascontiguousarray(a[:5])

if __name__=='__main__': run(SpecPolicy())
