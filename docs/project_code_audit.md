# PRP π0.5 项目代码审查

审查日期 2026-11-17（北京时间）。范围为 P0，只读盘点工作区并审查上游公开源码。本次新增两份 Markdown 报告，未运行模型或仿真。

## 工作区与框架

- 评测框架使用 RoboScope-Kit，π0.5 模型入口采用 Physical Intelligence 的 OpenPI。审查所用关键源码版本为 RoboScope-Kit `a7f756c173e621c42f8be18524c9bcfde91da526`（包版本 `0.8.0b66`）和 OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`。RoboScope-Kit 公开示例中，StarVLA 和 X-VLA 提供 LIBERO-Plus 接入参考，尚未找到可直接运行的 π0.5-LIBERO-Plus 示例。[RoboScope-Kit README](https://github.com/MINT-SJTU/RoboScope-Kit/tree/a7f756c173e621c42f8be18524c9bcfde91da526) · [OpenPI](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)

## 评测入口与文件

| 环节 | 已核对的入口与行为 |
|---|---|
| 策略接口 | `Policy.load(SessionContext) -> PolicyCapabilities` 读取 `context.checkpoint_path` 并声明 benchmark、观测特征及支持的 chunk 长度。 `reset(EpisodeContext)`、`act(Observation) -> ActionChunk`、`end_episode(episode_id)` 覆盖 episode 生命周期。 `run(policy)` 启动本地 socket 服务。 |
| 策略启动 | `--policy-config` 指向声明 `schema_version: 2`、`command`、`cwd` 的 JSON。Runtime 每个 replica 启动一个策略进程，并通过 RPC 加载 checkpoint、协商动作契约。 |
| 模拟器启动 | `benchmarks/libero_plus/build-runtime.sh <install-dir>` 准备 simulator prefix 和 manifest。 `python -m roboscope_kit.runtime.benchmark --benchmark libero_plus` 经 `cli.py`、`libero/factory.py` 和 `libero/adapter.py` 启动 `libero/worker.py`。每个 Worker 持有自己的仿真环境。 |
| 任务配置 | 协议 `libero-plus-v1`。Manifest 支持 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`，分别有 2402、2518、2591、2519 个任务。默认每个任务 1 个 episode、环境 seed 7、sample seed 7、10 步 settle、EGL。源码固定 LIBERO-Plus `4976dc30028e805ff8094b55501d532c48fec182`，资产固定 `dd2bd61b7d9a6fef1abc52d606e983b41886a149`。 |
| 结束与成功 | Worker 应用 native LIBERO 动作直到成功、环境终止或达到 horizon。以 `task_done` 判成功。 `--quick-test` 固定 task ID 0、1 个 episode、1 Worker、1 replica，并关闭视频。 |
| 结果位置 | 每次指向新的 `--output-dir`。主要文件为 `result.json`、`manifest.json`、`metrics.json`、`events.jsonl`、`progress.json`、`queue/`、`workers/`、`servers/`，有错误时检查 `errors.jsonl`。本项目拟将 P1 放在 `results/smoke/<run-id>/`，P2 放在 `results/baseline/<run-id>/`，目前尚无实跑结果。 |

核对文件见 [Policy API](https://github.com/MINT-SJTU/RoboScope-Kit/blob/a7f756c173e621c42f8be18524c9bcfde91da526/packages/roboscope-kit/src/roboscope_kit/policy.py) · [benchmark CLI](https://github.com/MINT-SJTU/RoboScope-Kit/blob/a7f756c173e621c42f8be18524c9bcfde91da526/packages/roboscope-kit/src/roboscope_kit/runtime/benchmark/cli.py) · [LIBERO factory、adapter 和 Worker](https://github.com/MINT-SJTU/RoboScope-Kit/tree/a7f756c173e621c42f8be18524c9bcfde91da526/packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/libero) · [LIBERO-Plus 构建和 manifest](https://github.com/MINT-SJTU/RoboScope-Kit/tree/a7f756c173e621c42f8be18524c9bcfde91da526/benchmarks/libero_plus) · [评测协议](https://github.com/MINT-SJTU/RoboScope-Kit/blob/a7f756c173e621c42f8be18524c9bcfde91da526/packages/roboscope-kit/LIBERO_EVALUATION_PROTOCOL.md)

## π0.5 的端到端调用链

`Observation → Preprocessing → π0.5 Inference → Action Chunk → Environment Execution → Evaluation`

1. **加载原始权重。** OpenPI CLI 用 `scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=...`。独立 RoboScope Policy 可在 `load()` 直接调用同一个 `create_trained_policy(get_config("pi05_libero"), checkpoint_dir)`，复用推理和转换代码。官方源为 `gs://openpi-assets/checkpoints/pi05_libero`。构造器以 `model.safetensors` 是否存在选择 PyTorch 或 JAX。JAX 从 `params/` 还原 BF16 权重，PyTorch loader 将选定参数转换为 BF16。默认从 checkpoint 的 `assets/<asset_id>/norm_stats.json` 读取配套归一化统计，配置中的 LIBERO asset ID 为 `physical-intelligence/libero`。[模型服务入口](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/serve_policy.py) · [checkpoint loader](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy_config.py) · [模型与数据配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py)
2. **从仿真取 Observation。** RoboScope LIBERO Worker 发送已经翻转两个空间轴的主视角和腕部图像，均为 256×256 RGB `uint8`，并提供 `language.instruction` 及 16 维 `state.proprioception`。Policy 直接使用 Kit 已翻转的图像，并复用 OpenPI 官方客户端的 resize-with-pad 到 224×224 和 uint8 转换。映射为 `observation/image`、`observation/wrist_image`、`prompt`。状态取末端位置 3 维、四元数转换到轴角 3 维、夹爪 qpos 2 维，组成 OpenPI 的 `observation/state`，沿用官方四元数约定，不传尾部 7 维关节位置。[RoboScope Worker](https://github.com/MINT-SJTU/RoboScope-Kit/blob/a7f756c173e621c42f8be18524c9bcfde91da526/packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/libero/worker.py) · [OpenPI 官方 LIBERO 客户端](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py)
3. **预处理与特征。** `LiberoInputs` 将两路相机映射到模型图像槽位，补零并掩码缺失的第三路相机。Policy 应用 checkpoint 对应的 quantile 归一化、图像 resize 与语言 tokenization，构造加 batch 维的 `Observation`。SigLIP 图像编码与 PaliGemma 语言嵌入构成前缀。这里 `pi05_libero` 明确设置 `pi05=True, discrete_state_input=False`，所以状态进入接口和变换链，但不拼到语言 token。模型的连续 state token 分支只在 `not self.pi05` 时启用。因此此配置的状态数值不作为动作生成的条件特征，状态数组仍参与输入形状等处理。[LIBERO 输入输出变换](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/libero_policy.py) · [训练/推理配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py) · [π0.5 JAX 模型](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)
4. **Flow Matching 动作采样。** `Policy.infer()` 调用 `sample_actions()`，先计算前缀 KV cache，再从形状为 `[B, 10, 32]` 的高斯噪声开始。Action expert 依次在当前噪声动作和时间条件下预测速度场，使用 `dt = -1 / num_steps` 从 1 向 0 做 Euler 更新，源码默认 10 步。每次更新处理整段连续动作。语言 token 用于条件前缀，动作采样的计算边界是速度场更新。[JAX sampler](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py) · [PyTorch sampler](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/pi0_pytorch.py) · [模型维度配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py)
5. **Action Chunk 到仿真。** OpenPI 做输出反归一化并取每步前 7 维，得到 10×7 的 native 动作。官方 LIBERO 客户端默认每次只执行前 5 步然后重规划。若要复现此执行节奏，Kit Policy 应声明长度 5，返回前 5 步 `float32` 动作。Worker 逐步调用 `env.step`，当前 chunk 耗尽才请求新观测，结束时写 terminal record，评测 runtime 汇总成功率。[OpenPI 原生 rollout](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py) · [RoboScope 动作契约](https://github.com/MINT-SJTU/RoboScope-Kit/blob/a7f756c173e621c42f8be18524c9bcfde91da526/packages/roboscope-kit/LIBERO_EVALUATION_PROTOCOL.md#4-libero-action-contract)

未来的接入位置是独立 `Policy.act()` 内部的动作采样调用，介于预处理与输出反归一化之间。基线保持原始 `infer() → sample_actions()`，候选实现通过独立策略复用输入输出契约。具体算法和改动等原始 Baseline 跑通后再决定。

## 延迟、显存与吞吐

- **框架已有的延迟。** `worker.py` 在构造 Observation 后，对 `PolicyClient.act()` 路径计墙钟时间，累加 `inference_wait_s`。它覆盖请求/连接、编解码和服务端处理等，adapter 将其与 `episode_elapsed_s` 汇总成 `inference_wait_ratio`。 `metrics.json` 是运行阶段耗时。OpenPI `policy_timing.infer_ms` 在输出同步或拷回 CPU 前结束计时，无显式 GPU 同步。这些字段都无法直接填充项目要求的同步模型 Mean/P50/P95。[Worker](https://github.com/MINT-SJTU/RoboScope-Kit/blob/a7f756c173e621c42f8be18524c9bcfde91da526/packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/libero/worker.py) · [OpenPI infer](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy.py)
- **模型计时建议。** 在 Policy 进程用单调高精度时钟逐请求记录 `preprocess_ms`（图像和状态转换、tokenization、输入传输，等输入就绪）、`predict_ms`（完整 `sample_actions`，含前缀与去噪）、`postprocess_ms`（结果回传、反归一化与切片）。采样起点排空已有 GPU 工作，结束时 PyTorch 用 `torch.cuda.synchronize(device)`，JAX 对输出用 `jax.block_until_ready`。固定 warm-up，记录首次编译/冷启动，稳态另算 Mean/P50/P95 和样本数。保留每个 chunk 的原始记录。另记请求 RPC 时间、episode 用时及框架 steps/s、episodes/s。若记录模型吞吐，用完成的 chunks / 稳态秒数并注明并发和 batch。[JAX 异步执行](https://docs.jax.dev/en/latest/async_dispatch.html) · [PyTorch CUDA 同步](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)
- **框架已有的显存。** `GpuSampler` 默认每 0.5 s 查询 `nvidia-smi memory.used`，`result.json` 内包括 GPU 采样数、`resources.gpu.peak_memory_bytes` 和各卡 `gpu_mem_mib_max`。采样峰值覆盖整卡、渲染和同卡进程，短尖峰可能漏采。 `policy_process.py` 的 `replica_vram_bytes` 将加载前后整卡用量差均分给同卡 replicas，是加载占用估计。保存原始采样、空闲占用、PID、设备、窗口与单位。无采样时记为缺失。[采样实现](https://github.com/MINT-SJTU/RoboScope-Kit/blob/a7f756c173e621c42f8be18524c9bcfde91da526/packages/roboscope-kit/src/roboscope_kit/runtime/benchmark/resources.py) · [Replica 显存估算](https://github.com/MINT-SJTU/RoboScope-Kit/blob/a7f756c173e621c42f8be18524c9bcfde91da526/packages/roboscope-kit/src/roboscope_kit/runtime/benchmark/policy_process.py)
- **Policy 峰值建议。** PyTorch 在测量窗口前同步并调用 `reset_peak_memory_stats`，结束同步后报告 `max_memory_allocated` 和 `max_memory_reserved`，保留是否包含已驻留权重。JAX 记录 XLA 预分配设置，有可用的后端 allocator 统计时报告其峰值字段及口径，否则报告设备/进程采样峰值并标注采样限制。至少分别保留模型侧与系统侧指标。统一 MiB/GiB（2 的幂）或 GB（10 的幂）换算规则再填表。[PyTorch CUDA 内存管理](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management) · [JAX GPU 内存分配](https://docs.jax.dev/en/latest/gpu_memory_allocation.html) · [NVIDIA NVML](https://docs.nvidia.com/deploy/nvml-api/latest/)

## 从零复现所需资源

| 资源 | 来源及核验 |
|---|---|
| RoboScope-Kit 与 OpenPI 源码 | 使用上文记录的两个提交作为起始版本，分别记录实际 Git SHA，OpenPI 需要初始化已固定的 submodules。 |
| π0.5 LIBERO 权重和归一化统计 | 官方 `gs://openpi-assets/checkpoints/pi05_libero`，采用配置 `pi05_libero`。下载完整 checkpoint 后检查实际权重格式以及 `assets/physical-intelligence/libero/norm_stats.json`。 |
| PaliGemma tokenizer | OpenPI 从 `gs://big_vision/paligemma_tokenizer.model` 获取，和 checkpoint 一起记入资产清单。 |
| LIBERO-Plus 评测源码与资产 | Kit 构建脚本固定 LIBERO-Plus commit 和资产 revision。资产包约 6.4 GB，下载后按脚本记录的 SHA-256 验证。任务定义、初始状态与仿真资产用于评测，当前任务无需训练集。 |
| 运行时依赖 | 准备独立 OpenPI uv 环境、Kit 控制环境和 LIBERO-Plus 模拟器环境。目标机需验证 EGL 渲染。 |


资源依据见 [OpenPI 安装和 checkpoint](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md) · [Tokenizer 源码](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/tokenizer.py) · [LIBERO-Plus 构建与依赖锁定](https://github.com/MINT-SJTU/RoboScope-Kit/tree/a7f756c173e621c42f8be18524c9bcfde91da526/benchmarks/libero_plus)

## P1 在目标机上的单任务验证

先执行 [环境报告](environment_report.md) 中的只读命令。复用已有源码、checkpoint、归一化统计和资产。选择并记录实际 backend、精度、去噪步数、chunk 执行步数及 seed。准备独立 π0.5 Policy 和对应进程 JSON 后，在已具备 Kit CLI 的 Linux 环境中执行以下命令模板。变量由目标机实际路径提供，具体 `libero_10` 是待确认的 smoke suite 候选。

~~~bash
: "${PI05_CHECKPOINT_DIR:?Set to existing local checkpoint directory}"
: "${LIBERO_PLUS_MANIFEST:?Set to existing benchmark-manifest.json}"
: "${PI05_POLICY_CONFIG:?Set to π0.5 RoboScope policy JSON}"
mkdir -p results/smoke
run_dir="$(mktemp -d results/smoke/pi05-libero-plus.XXXXXX)"
python -m roboscope_kit.runtime.benchmark \
  --benchmark libero_plus --suite libero_10 --quick-test \
  --checkpoint-source "$PI05_CHECKPOINT_DIR" \
  --benchmark-manifest "$LIBERO_PLUS_MANIFEST" \
  --policy-config "$PI05_POLICY_CONFIG" \
  --output-dir "$run_dir"
~~~

验收看一次实际的 `load → reset → act → env.step → terminal record`，检查 `result.json`、`events.jsonl`、Worker 和 replica 日志。Episode 允许以正常失败结束，仍须验证实际调用和执行过动作。正式 Baseline 在 smoke 后固定任务、并发和版本，以新的输出目录运行，归档每次运行的完整配置、原始延迟/显存、日志及环境信息。单任务实测见 [验证报告](smoke_test_report.md)。
