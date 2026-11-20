from dataclasses import replace
import os
from pathlib import Path

from roboscope_kit import BenchmarkName, run
from kit_scored_policy import KitScoredPolicy


class CameraScoredPolicy(KitScoredPolicy):
    def load(self, context):
        capabilities = super().load(context)
        name = BenchmarkName.LIBERO_PLUS.value
        return replace(capabilities, supported_benchmarks=(name,), supported_suites={name:('libero_10',)})


if __name__ == '__main__':
    run(CameraScoredPolicy(log_path=Path(os.environ['SCORED_TRIAL_DIR'])/'selection.jsonl', seed=7))
