import jax
import numpy as np
from openpi.policies import policy_config
from openpi.training import config
from openpi_client import image_tools
from roboscope_kit import ActionChunk, BenchmarkName, Policy, PolicyCapabilities, run


class Pi05LiberoPlusPolicy(Policy):
    action_chunk_size = 5
    sample_kwargs = None

    def load(self, context):
        print(f"JAX GPU: {jax.devices('gpu')}", flush=True)
        model_config = config.get_config("pi05_libero")
        self.noise_shape = (model_config.model.action_horizon, model_config.model.action_dim)
        self.policy = policy_config.create_trained_policy(
            model_config, context.checkpoint_path, sample_kwargs=self.sample_kwargs
        )
        benchmark = BenchmarkName.LIBERO_PLUS.value
        return PolicyCapabilities(
            supported_benchmarks=(benchmark,),
            required_features=(
                "images.primary", "images.wrist",
                "state.proprioception", "language.instruction",
            ),
            supported_suites={benchmark: ("libero_10", "libero_spatial", "libero_object", "libero_goal")},
            supported_action_chunk_sizes=(self.action_chunk_size,),
        )

    def model_input(self, observation):
        features = observation.features
        proprio = np.asarray(features["state.proprioception"], dtype=np.float32)
        # LIBERO sends position, an XYZW quaternion, then gripper qpos.
        w = np.clip(float(proprio[6]), -1.0, 1.0)
        sin_half_angle = np.sqrt(1.0 - w * w)
        rotation = np.zeros(3, dtype=np.float32)
        if sin_half_angle > 0.0:
            rotation = proprio[3:6] * np.float32(2.0 * np.arccos(w) / sin_half_angle)

        return {
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(features["images.primary"], 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(features["images.wrist"], 224, 224)
            ),
            "observation/state": np.concatenate((proprio[:3], rotation, proprio[7:9])),
            "prompt": features["language.instruction"],
        }
    def action_prefix(self, model_input, *, noise=None):
        actions = np.asarray(self.policy.infer(model_input, noise=noise)["actions"], dtype=np.float32)
        expected_shape = (self.noise_shape[0], 7)
        if actions.shape != expected_shape or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite OpenPI actions of shape {expected_shape}, got {actions.shape}")
        return np.ascontiguousarray(actions[:self.action_chunk_size])

    def act(self, observation):
        return ActionChunk(self.action_prefix(self.model_input(observation)))


if __name__ == "__main__":
    run(Pi05LiberoPlusPolicy())
