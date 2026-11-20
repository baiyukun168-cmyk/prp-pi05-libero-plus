"""Expose the archived scored algorithm to the requested LIBERO-Plus suite."""

from dataclasses import replace
import json
import os
from pathlib import Path

from roboscope_kit import BenchmarkName,run
from kit_scored_policy import KitScoredPolicy


class CameraSuiteScoredPolicy(KitScoredPolicy):
    def load(self,context):
        capabilities=super().load(context)
        selection=json.loads(Path(os.environ["CAMERA_SELECTION"]).read_text())
        name=BenchmarkName.LIBERO_PLUS.value
        return replace(capabilities,supported_benchmarks=(name,),supported_suites={name:(selection["suite"],)})


if __name__=="__main__":
    run(CameraSuiteScoredPolicy(log_path=Path(os.environ["SCORED_TRIAL_DIR"])/"selection.jsonl",seed=7))
