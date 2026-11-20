import hashlib
import numpy as np


def paired_noise(seed, episode_id, action_step, shape):
    """Physical-step indexing avoids repeated noise for a shorter execution chunk."""
    if action_step < 0:
        raise ValueError('action_step must be nonnegative')
    episode_seed=int.from_bytes(hashlib.sha256(f'{seed}:{episode_id}'.encode()).digest()[:8],'big')
    noise_seed=int(np.random.SeedSequence([episode_seed,action_step,0]).generate_state(1,dtype=np.uint64)[0])
    noise=np.random.default_rng(noise_seed).standard_normal(shape,dtype=np.float32)
    return noise_seed,noise
