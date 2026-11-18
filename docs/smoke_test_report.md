# π0.5 / LIBERO-Plus 单任务验证

2026-11-18 在 Ubuntu 22.04.5、NVIDIA RTX 4090 上运行官方 OpenPI `pi05_libero` checkpoint。RoboScope-Kit 通过 EGL 驱动 LIBERO-Plus `libero_10` task 0。每次执行动作块前 5 步，使用 1 Worker 和 1 replica，仿真 seed 为 7。

本次成功完成 **1 个 Episode、250 个仿真步**。RoboScope 记录的评测用时为 52.50 秒，包含仿真和策略调用。模型延迟及正式 Baseline 待测。见[实测结果摘要](runs/smoke_4090.json)。

在 Linux 上按[环境说明](environment_report.md)准备 OpenPI、RoboScope-Kit、模拟器与 checkpoint。项目根目录应包含 `openpi/`、`RoboScope-Kit/`、`kit-tools/`、`libero-plus-runtime/` 和 `checkpoints/pi05_libero/`。运行[启动脚本](../scripts/run_pi05_libero_plus_smoke.sh)时设置实际路径。

~~~bash
PRP_ROOT=/path/to/prp MAGICK_HOME=/path/to/prp/kit-tools \
  bash scripts/run_pi05_libero_plus_smoke.sh
~~~
