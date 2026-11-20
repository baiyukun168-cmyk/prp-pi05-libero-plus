import json
import os
from pathlib import Path
from roboscope_kit import BenchmarkName
from roboscope_kit.runtime.benchmark import cli
from roboscope_kit.runtime.benchmarks.libero_plus.factory import LiberoPlusBenchmarkFactory

selection=json.loads(Path(os.environ['PROMPT_SELECTION']).read_text())
ids=tuple(r['task_id'] for r in selection['tasks'])
assert len(ids)==25 and len(set(ids))==25


class AblationFactory(LiberoPlusBenchmarkFactory):
    def create(self,request,manifest):
        assert request.suite=='libero_10' and request.episodes==1 and not request.quick_test
        adapter=self._create(request,manifest,task_ids=ids,episodes=1)
        assert tuple(i.payload['task_id'] for i in adapter.work_items())==ids
        return adapter

    def create_probe(self,*args,**kwargs):
        raise RuntimeError('Fixed diagnostic does not autotune')


def factory(name):
    assert name is BenchmarkName.LIBERO_PLUS
    return AblationFactory()


cli.benchmark_factory=factory
raise SystemExit(cli.main())
