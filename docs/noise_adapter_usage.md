# Noise evaluator adapter and storage boundaries

The opt-in adapter defers expensive LIBERO+ image noise until a consumer reads the image. Every logical frame still consumes the original NumPy random sequence. The glass-blur pixel loop is compiled with Numba and preserves the pinned upstream view-assignment behavior. Model weights, model sampling settings, and original upstream source files are unchanged.

The adapter checks the upstream noise source SHA-256, protects RNG during JIT warmup and deferred materialization, and caches materialized images. It is intended for single-threaded RGB simulator workers using the validated upstream and dependency versions. Video consumers also trigger materialization.

## Code

- `src/adapters/noise_acceleration.py`: optional evaluator adapter.
- `src/evaluation/noise_validation_worker.py`: native Kit worker wrapper with step and observation auditing.
- `scripts/probe_noise_acceleration.py`: isolated CPU correctness and timing probe.
- `scripts/probe_noise_throughput.py`: consumed-frame and worker-concurrency probe.
- `scripts/check_noise_real_trajectory.py`: real-environment fixed-action comparison.
- `tests/cloud_noise_acceleration_contract.py`: explicitly staged cloud integration contract.

These cloud probes expect the existing project layout under `/root/prp`; copy the adapter and required helper modules beside the staged entry point, use the simulator interpreter and its LIBERO/Kit PYTHONPATH, and supply the source hash and private output directory. They are not a self-contained one-command installer. Do not enable the adapter solely on a microbenchmark: verify source/dependency versions, exact consumed pixels/RNG and longer fixed-action traces, then test complete episodes.

No exact full-policy-trajectory equivalence or improved success rate is asserted. Low-precision model outputs may differ across separate runs; distinguish evaluator throughput from model-call latency. Record concurrency and cold-start scope separately.

## Storage

- Public GitHub: reusable code, configuration schemas, methods and build definitions.
- Private Hugging Face Model: authorized checkpoint assets and provenance.
- Private Hugging Face Dataset: raw runs, per-episode results, logs and source snapshots.
- Container registry: built, tested runtime images, referenced by immutable digest.
- Team Wiki: results entered by the project owner.

The current private dataset is https://huggingface.co/datasets/byk666666/prp-libero-plus-data . A new final snapshot has been prepared locally, but it has NOT yet been uploaded; the previous published snapshot remains separate. No raw results, numerical result tables, model weights or credentials are included in this public code update.

The existing core test suite also expects the upstream benchmark task_classification.json in results/paired/. Supply that fixture locally from the pinned LIBERO+ checkout; it is intentionally excluded from the public upload.

