"""Native Kit entry restricted to the fixed full Camera selection for one suite."""

import json
import os
from pathlib import Path

from roboscope_kit import BenchmarkName
from roboscope_kit.runtime.benchmark import cli
from roboscope_kit.runtime.benchmarks.libero_plus.factory import LiberoPlusBenchmarkFactory


selection=json.loads(Path(os.environ["CAMERA_SELECTION"]).read_text())
suite=selection["suite"]
ids=tuple(row["task_id"] for row in selection["tasks"])
counts={"libero_spatial":376,"libero_object":396,"libero_goal":408}
if suite not in counts or len(ids)!=counts[suite] or len(set(ids))!=len(ids):
    raise ValueError("Remaining Camera selection has an unexpected population")
if selection["category"]!="Camera Viewpoints" or selection["modes"]!=["scored"]:
    raise ValueError("Only the scored Camera policy is authorized")


class AllCameraFactory(LiberoPlusBenchmarkFactory):
    def create(self,request,manifest):
        if request.suite!=suite or request.quick_test or request.episodes!=1:
            raise ValueError("Suite and episode count must match the archived Camera selection")
        adapter=self._create(request,manifest,task_ids=ids,episodes=1)
        items=adapter.work_items()
        if tuple(item.payload["task_id"] for item in items)!=ids:
            raise RuntimeError("Kit workload differs from fixed Camera selection")
        return adapter

    def create_probe(self,*args,**kwargs):
        raise RuntimeError("Camera evaluation uses fixed 4/1 concurrency")


def selected_factory(name):
    if name is not BenchmarkName.LIBERO_PLUS:
        raise ValueError("Only LIBERO-Plus is supported")
    return AllCameraFactory()


cli.benchmark_factory=selected_factory
raise SystemExit(cli.main())
