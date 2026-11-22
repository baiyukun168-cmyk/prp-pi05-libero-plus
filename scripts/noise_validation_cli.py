"""Select fixed Noise tasks and route workers through the isolated adapter."""
from dataclasses import replace
import json
import os
from pathlib import Path

from roboscope_kit import BenchmarkName
from roboscope_kit.runtime.benchmark import cli
from roboscope_kit.runtime.benchmarks.libero_plus.factory import LiberoPlusBenchmarkFactory
from roboscope_kit.runtime.benchmarks.libero_plus.adapter import LiberoPlusBenchmarkAdapter

selection=json.loads(Path(os.environ['PROMPT_SELECTION']).read_text())
ids=tuple(r['task_id'] for r in selection['tasks'])
assert len(ids)==len(set(ids))==10 and selection['category']=='Sensor Noise'


class NoiseAdapter(LiberoPlusBenchmarkAdapter):
    def worker_process(self,context):
        spec=super().worker_process(context)
        assert spec.command[1:3]==('-m','roboscope_kit.runtime.benchmarks.libero.worker')
        env=dict(spec.env)
        for key in ('PRP_NOISE_MODE','PRP_NOISE_AUDIT_DIR','PRP_NOISE_SOURCE_SHA256'):
            env[key]=os.environ[key]
        return replace(spec,command=(*spec.command[:2],'noise_validation_worker',*spec.command[3:]),env=env)


class Factory(LiberoPlusBenchmarkFactory):
    def create(self,request,manifest):
        assert request.suite==selection['suite'] and request.episodes==1 and not request.quick_test
        adapter=self._create(request,manifest,task_ids=ids,episodes=1)
        return NoiseAdapter(adapter.config)

    def create_probe(self,*args,**kwargs):raise RuntimeError('Fixed concurrency; no autotuning')


def factory(name):
    assert name is BenchmarkName.LIBERO_PLUS
    return Factory()


cli.benchmark_factory=factory
raise SystemExit(cli.main())
