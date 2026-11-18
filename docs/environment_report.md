# RTX 4090 复现环境

单任务验证使用 Ubuntu 22.04.5、NVIDIA RTX 4090（24564 MiB）和 NVIDIA 驱动 595.80。策略侧 JAX 识别 GPU，模拟器使用 EGL 渲染。运行配置见 [Smoke Test 报告](smoke_test_report.md)。

## 目标 Linux 机器的依赖

| 进程 | 上游要求与实机待验证项 |
|---|---|
| 评测控制进程 | RoboScope-Kit 包要求 Python ≥3.11。记录版本和 checkout commit。构建脚本要求 `KIT_PYTHON` 指向能导入 Kit 的解释器。 |
| LIBERO-Plus Worker | 使用上游构建脚本定义的独立 Linux x86_64 Conda/Micromamba prefix。固定 CPython 3.10.20、LIBERO 0.1.0（固定源码）、MuJoCo 3.2.3、robosuite 1.4.0、bddl 1.0.1、NumPy 1.24.4、Torch 2.5.1+cpu。模拟器渲染走 manifest 的 EGL，仍要验证 NVIDIA/EGL 可用性。 |
| π0.5 Policy | 独立的 OpenPI uv 环境，审查 commit 的 `pyproject.toml` 要求 Python ≥3.11，模型核心依赖包括 `jax[cuda12]==0.5.3`、`torch==2.7.1`、`transformers==4.53.2` 和 workspace 内的 `openpi-client`。按实际 checkpoint/backend 验证 CUDA、设备、精度、tokenizer 资产及归一化统计。策略进程还需同版 Kit API。 |
| 仿真资产 | 构建脚本依赖 `conda` 或 `micromamba`、`git`、`curl`、`unzip`。优先寻找已构建 runtime 或已有 `LIBERO_PLUS_ASSETS_ZIP`。脚本在无可复用资产时可能下载约 6.4 GB 归档，需为归档及解压留空间。 |

环境依据见 [RoboScope-Kit 包配置](https://github.com/MINT-SJTU/RoboScope-Kit/blob/a7f756c173e621c42f8be18524c9bcfde91da526/packages/roboscope-kit/pyproject.toml) · [LIBERO-Plus 构建与依赖锁定](https://github.com/MINT-SJTU/RoboScope-Kit/tree/a7f756c173e621c42f8be18524c9bcfde91da526/benchmarks/libero_plus) · [OpenPI 包配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/pyproject.toml)
