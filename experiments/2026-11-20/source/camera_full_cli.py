"""Run exactly all 419 configured Camera entries through the native Kit runtime."""
import json
import os
from pathlib import Path

from roboscope_kit import BenchmarkName
from roboscope_kit.runtime.benchmark import cli
from roboscope_kit.runtime.benchmarks.libero_plus.factory import LiberoPlusBenchmarkFactory

selection = json.loads(Path(os.environ['CAMERA_SELECTION']).read_text())
ids = tuple(row['task_id'] for row in selection['tasks'])
if len(ids) != 419 or len(set(ids)) != 419 or selection['category'] != 'Camera Viewpoints':
    raise ValueError('Full Camera selection must contain exactly 419 unique entries')


class AllCameraFactory(LiberoPlusBenchmarkFactory):
    def create(self, request, manifest):
        if request.suite != 'libero_10' or request.quick_test or request.episodes != 1:
            raise ValueError('Full Camera run uses libero_10, one episode per entry')
        adapter = self._create(request, manifest, task_ids=ids, episodes=1)
        items = adapter.work_items()
        if len(items) != 419 or tuple(item.payload['task_id'] for item in items) != ids:
            raise RuntimeError('Camera queue differs from fixed 419-entry manifest')
        return adapter

    def create_probe(self, *args, **kwargs):
        raise RuntimeError('Full run uses fixed 4/1; autotune is disabled')


def selected_factory(name):
    if name is not BenchmarkName.LIBERO_PLUS:
        raise ValueError('This entry only supports LIBERO-Plus')
    return AllCameraFactory()


cli.benchmark_factory = selected_factory
raise SystemExit(cli.main())
