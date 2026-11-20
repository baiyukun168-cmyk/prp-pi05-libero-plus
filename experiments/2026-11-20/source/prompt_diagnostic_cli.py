import json
import os
from pathlib import Path
from roboscope_kit import BenchmarkName
from roboscope_kit.runtime.benchmark import cli
from roboscope_kit.runtime.benchmarks.libero_plus.factory import LiberoPlusBenchmarkFactory

selection = json.loads(Path(os.environ['PROMPT_SELECTION']).read_text())
ids = tuple(r['task_id'] for r in selection['tasks'])
expected = int(selection.get('expected_tasks', 6))
assert expected in (6,25) and len(ids) == expected and len(set(ids)) == expected


class DiagnosticFactory(LiberoPlusBenchmarkFactory):
    def create(self, request, manifest):
        assert request.suite == 'libero_10' and request.episodes == 1 and not request.quick_test
        adapter = self._create(request, manifest, task_ids=ids, episodes=1)
        assert len(adapter.work_items()) == expected
        return adapter

    def create_probe(self, *args, **kwargs):
        raise RuntimeError('Diagnostic uses fixed concurrency')


def factory(name):
    assert name is BenchmarkName.LIBERO_PLUS
    return DiagnosticFactory()


cli.benchmark_factory = factory
raise SystemExit(cli.main())
