# PRP π0.5 推理加速实验

在 Windows 工作区维护实验代码、配置和报告，在租用的 Ubuntu RTX 4090 上运行 OpenPI 与 LIBERO-Plus。先复现官方 π0.5 LIBERO checkpoint，通过单任务 Smoke Test，再记录本项目实测 Baseline。

## 项目目录

| 目录 | 用途 |
|---|---|
| `docs/` | [代码审查](docs/project_code_audit.md)和[环境报告](docs/environment_report.md) |
| `scripts/` | 实验入口 |
| `results/` | 本地查看或归档的实验输出 |
| `third_party/` | 可选的本地上游源码 checkout |

## 固定的上游源码

| 项目 | 起始提交 |
|---|---|
| [RoboScope-Kit](https://github.com/MINT-SJTU/RoboScope-Kit/tree/a7f756c173e621c42f8be18524c9bcfde91da526) | `a7f756c173e621c42f8be18524c9bcfde91da526` |
| [OpenPI](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479) | `215abfb217dbac7d5f1273282331b9b1866c0479` |

如果本机没有源码 checkout，在项目根目录用 PowerShell 执行：

~~~powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone --filter=blob:none --no-checkout https://github.com/MINT-SJTU/RoboScope-Kit.git third_party/RoboScope-Kit
git -C third_party/RoboScope-Kit checkout --detach a7f756c173e621c42f8be18524c9bcfde91da526
git clone --filter=blob:none --no-checkout https://github.com/Physical-Intelligence/openpi.git third_party/openpi
git -C third_party/openpi checkout --detach 215abfb217dbac7d5f1273282331b9b1866c0479
git -C third_party/openpi submodule update --init --recursive
~~~

每个仓库的 clone 成功后再执行对应的 checkout。已有 checkout 时，用 `git -C <目录> status --short --branch` 和 `git -C <目录> rev-parse HEAD` 核对状态与版本。

## 实验进度

云端 RTX 4090 已加载原始 π0.5 并通过 LIBERO-Plus 单任务 Smoke Test。见[验证报告](docs/smoke_test_report.md)和[启动脚本](scripts/run_pi05_libero_plus_smoke.sh)。本次完成 1 个 Episode、250 步，任务成功。正式 Baseline 待测。见[结果摘要](docs/runs/smoke_4090.json)。
