"""Version-pinned OpenPI adapter sharing one encoded observation across candidates."""

import einops
from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import model as model_utils
from openpi.models.pi0 import make_attn_mask


def create_shared_prefix_sampler(model, *, with_scores=False):
    graph, weights = nnx.split(model)

    @jax.jit
    def sample(weights, observation, noise, num_steps, verification_noise):
        resident = nnx.merge(graph, weights)
        observation = model_utils.preprocess_observation(None, observation, train=False)
        prefix, prefix_mask, prefix_ar = resident.embed_prefix(observation)
        positions = jnp.cumsum(prefix_mask, axis=1)-1
        _, cache = resident.PaliGemma.llm([prefix, None], mask=make_attn_mask(prefix_mask,prefix_ar), positions=positions)
        count = noise.shape[0]
        # Gemma scans layers on axis 0; its KV batch dimension is axis 1.
        cache = jax.tree.map(lambda value: jnp.repeat(value,count,axis=1), cache)
        observation = jax.tree.map(lambda value: jnp.repeat(value,count,axis=0), observation)
        prefix_mask = jnp.repeat(prefix_mask,count,axis=0)
        dt = -1.0/num_steps

        def velocity(actions, time):
            suffix, mask, ar, cond = resident.embed_suffix(observation, actions, jnp.broadcast_to(time,count))
            prefix_attention = einops.repeat(prefix_mask,"b p -> b s p",s=suffix.shape[1])
            attention = jnp.concatenate([prefix_attention,make_attn_mask(mask,ar)],axis=-1)
            suffix_positions = jnp.sum(prefix_mask,axis=-1)[:,None]+jnp.cumsum(mask,axis=-1)-1
            (_, suffix_out), _ = resident.PaliGemma.llm([None,suffix], mask=attention,
                                                       positions=suffix_positions,kv_cache=cache,
                                                       adarms_cond=[None,cond])
            return resident.action_out_proj(suffix_out[:,-resident.action_horizon:])

        def step(carry):
            actions, time = carry
            return actions+dt*velocity(actions,time),time+dt

        actions, _ = jax.lax.while_loop(lambda carry: carry[1] >= -dt/2, step,(noise,1.0))
        if with_scores:
            # Independent verification noise is shared by all candidates.
            point = .5*actions + .5*verification_noise[None,...]
            residual = velocity(point,jnp.asarray(.5))-(verification_noise[None,...]-actions)
            scores = jnp.mean(jnp.square(residual[:,:5,:7]),axis=(1,2))
            return actions, scores
        return actions

    def infer(observation, noise, *, num_steps=4, verification_noise=None):
        if observation.state.shape[0] != 1:
            raise ValueError("Shared-prefix adapter expects one observation")
        if with_scores and verification_noise is None:
            raise ValueError("Scoring requires independent verification noise")
        if verification_noise is None:
            verification_noise = jnp.zeros(noise.shape[1:],dtype=noise.dtype)
        return sample(weights,observation,noise,num_steps,verification_noise)

    return infer
