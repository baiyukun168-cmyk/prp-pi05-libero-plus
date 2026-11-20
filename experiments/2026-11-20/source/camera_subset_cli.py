"""Select an explicit Camera subset; use Kit's unchanged scheduling and workers."""
import json
import os
from pathlib import Path

from roboscope_kit import BenchmarkName
from roboscope_kit.runtime.benchmark import cli
from roboscope_kit.runtime.benchmarks.libero_plus.factory import LiberoPlusBenchmarkFactory

selection = json.loads(Path(os.environ['CAMERA_SELECTION']).read_text())
ids = tuple(row['task_id'] for row in selection['tasks'])
assert len(ids) == 25 and len(set(ids)) == 25


class SelectedCameraFactory(LiberoPlusBenchmarkFactory):
    def create(self, request, manifest):
        if request.suite != 'libero_10' or request.quick_test or request.episodes != 1:
            raise ValueError('This pilot only supports libero_10, one episode per task')
        adapter = self._create(request, manifest, task_ids=ids, episodes=1)
        items = adapter.work_items()
        if len(items) != 25 or tuple(item.payload['task_id'] for item in items) != ids:
            raise RuntimeError('Selected work items do not match fixed Camera manifest')
        return adapter

    def create_probe(self, *args, **kwargs):
        raise RuntimeError('Pilot uses fixed 4/1; autotune is disabled')


def selected_factory(name):
    if name is not BenchmarkName.LIBERO_PLUS:
        raise ValueError('This entry only supports LIBERO-Plus')
    return SelectedCameraFactory()


# Local entrypoint binding only: no upstream source file is edited.
cli.benchmark_factory = selected_factory
raise SystemExit(cli.main())
