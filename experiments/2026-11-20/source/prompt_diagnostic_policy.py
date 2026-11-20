"""Single-variable diagnostic: original Camera prompt versus BDDL instruction."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from roboscope_kit import run
from camera_scored_policy import CameraScoredPolicy


class PromptDiagnosticPolicy(CameraScoredPolicy):
    def load(self, context):
        self.prompts = json.loads(Path(os.environ['PROMPT_SELECTION']).read_text())['bdll_prompts']
        self.mode = os.environ['PROMPT_MODE']
        self.initial_seen = set()
        self.input_log = (Path(os.environ['SCORED_TRIAL_DIR'])/'initial-inputs.jsonl').open('a', buffering=1)
        return super().load(context)

    def model_input(self, observation):
        element = super().model_input(observation)
        if observation.episode_id not in self.prompts:
            return element  # Shared dummy warmup, identical in both conditions.
        original = element['prompt']
        if self.mode == 'bddl':
            element['prompt'] = self.prompts[observation.episode_id]
        if observation.episode_id not in self.initial_seen:
            fingerprints = {}
            for key in ('observation/image', 'observation/wrist_image', 'observation/state'):
                value = np.ascontiguousarray(element[key])
                fingerprints[key] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                                     'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
            self.input_log.write(json.dumps({'episode_id': observation.episode_id,
                'step': observation.step, 'mode': self.mode, 'original_prompt': original,
                'actual_prompt': element['prompt'], 'fingerprints': fingerprints})+'\n')
            self.initial_seen.add(observation.episode_id)
        return element


if __name__ == '__main__':
    run(PromptDiagnosticPolicy(log_path=Path(os.environ['SCORED_TRIAL_DIR'])/'selection.jsonl', seed=7))
