"""Render public comparison tables from the audited summary, without raw logs."""
import json
from pathlib import Path
import shutil
import sys


def duration(seconds):
    seconds = round(seconds)
    return f'{seconds//3600}h {(seconds%3600)//60}m {seconds%60}s'


def main():
    source, destination = map(Path, sys.argv[1:])
    destination.mkdir(parents=True, exist_ok=True)
    s = json.loads((source/'summary.json').read_text())
    o = s['overview']
    call_count = sum(x['call_count'] for x in s['speculative'].values())
    accepted = sum(x['accepted'] for x in s['speculative'].values())
    attempts = sum(x['attempts'] for x in s['speculative'].values())
    rows = s['comparison']
    lines = ['# π0.5 两位置近似推测采样：七维完整评测', '',
             '分析日期：2026-11-23。结论：本次观察到约一成的推理加速；整体成功率基本持平，但未证明统计等价或非劣效；没有充分证据支持降低显存或硬件要求。', '',
             '## 1. 比较对象与实验条件', '',
             '比较的是本项目在 RTX 4090 上实测的官方 π0.5 与两位置近似推测采样，不是老师提供的参考表。老师原表保持原样，其硬件、配置及 Avg. 统计口径未确认，因此不作公平提升基准。', '',
             '- 同一个 pi05_libero 原始 checkpoint，未训练、压缩或下载新权重。OpenPI JAX BF16，10 步流采样网格，每轮执行 5 个动作。',
             '- 七维、四个子套件，共 10,030 个任务变体；每变体一次 Episode，环境与物理动作步噪声种子均为 7，采用真实 BDDL 指令。',
             '- 前五维 4 Worker，Noise/Layout 8 Worker，均为 1 Replica。两版 Noise 均沿用已有按需加噪适配器，因此本表的差异不是把环境加噪优化混作采样器收益。',
             '- 记录的上游版本：OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`；Kit `a7f756c173e621c42f8be18524c9bcfde91da526`；LIBERO-Plus `4976dc30028e805ff8094b55501d532c48fec182`。',
             '- 两版都只在 RTX 4090 上运行；关机后的改进版续跑不是同一不间断进程。没有低显存 GPU 或其他硬件实测。', '',
             '## 2. 明确的进步：观察到推理延迟降低', '',
             f"全部正式、已完成 Episode 的真实调用加权均值：**{o['baseline_mean_ms']:.2f} → {o['speculative_mean_ms']:.2f} ms**，耗时减少 **{o['latency_reduction_pct']:.2f}%**，调用加速约 **{o['baseline_mean_ms']/o['speculative_mean_ms']:.3f}×**。",
             '按任务等权平均每个 Episode 的调用均值也得到相同方向，排除了单纯由长 Episode 调用更多造成的主要权重偏差；但没有消除动作轨迹、运行时间段与调度差异。', '',
             '| 维度 | 原版 Mean ms | 改进版 Mean ms | 耗时减少 | 原版 P95 ms | 改进版 P95 ms |',
             '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['dimension'].title()} | {r['baseline_mean_ms']:.2f} | {r['speculative_mean_ms']:.2f} | {r['latency_reduction_pct']:.2f}% | {r['baseline_p95_ms']:.2f} | {r['speculative_p95_ms']:.2f} |")
    lines += ['', '计时口径：`act_wall_ms` 为 Policy 侧调用墙钟耗时，包含输入处理、预测和返回动作处理，不含加载与预热，不是完整 Episode 耗时或客户端排队时间。采样器通过 `jax.device_get` 等待动作与统计量落到主机，计时包含该同步；不能将 GPU 异步提交时间当作完成时间。P50/P95 使用排序后线性插值。`act_wall_ms - predict_ms` 仅代表预测外开销，不能单独称为纯预处理时间。', '',
              '## 3. 成功率：总体没有提升', '',
              '| 维度 | 原版成功/总数 | 改进版成功/总数 | 原版成功率 | 改进版成功率 | 差值（百分点） |',
              '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['dimension'].title()} | {r['baseline_success']}/{r['episodes']} | {r['speculative_success']}/{r['episodes']} | {r['baseline_success_pct']:.2f}% | {r['speculative_success_pct']:.2f}% | {r['success_delta_pp']:+.3f} |")
    lines += [f"| 合计（按 Episode 加权） | {o['baseline_success']}/10030 | {o['speculative_success']}/10030 | {100*o['baseline_success']/10030:.3f}% | {100*o['speculative_success']/10030:.3f}% | {100*(o['speculative_success']-o['baseline_success'])/10030:+.3f} |", '',
              f"逐任务配对发现：原版失败→改进版成功 **{o['paired_wins']} 条**；原版成功→改进版失败 **{o['paired_losses']} 条**。净差只有 -2 条，但合计 570 条成败发生变化，不能把总体接近解释为动作或结果逐任务无损等价。", '',
              'Layout 净增加 10 次成功、Camera 净增加 2 次；Noise 净减少 12 次、Light 净减少 2 次；其余总数相同，但任务成败不一定相同。每个任务只有一个种子/一次 Episode，扰动变体也并非独立基础任务；这些差异不能证明某维度有稳定收益。CSV 中的逐维精确 McNemar 双侧 p 值仅作探索性配对描述，未校正多重比较，也不能代替多种子与按基础任务分组的统计。', '',
              '## 4. 整场耗时与显存', '',
              f"记录的正式评测阶段合计：**{duration(o['baseline_evaluation_s'])} → {duration(o['speculative_evaluation_s'])}**，约 **{o['evaluation_speedup']:.3f}×**。加上记录的加载和环境准备阶段：**{duration(o['baseline_active_total_s'])} → {duration(o['speculative_active_total_s'])}**，减少约 **{(o['baseline_active_total_s']-o['speculative_active_total_s'])/60:.1f} 分钟**。", '',
              '这些是阶段计时之和，不是从日历开始到结束的时间。改进版包含首次 Noise 部分批次和重启后的加载，排除整夜暂停；中断前未完成的工作仍占运行时间，所以整场耗时不是严格无中断 A/B 测试。两版执行总步数分别为 1,904,274 和 1,903,760，任务轨迹不完全相同。', '',
              '| 维度 | 原版峰值 GiB | 改进版峰值 GiB | 说明 |',
              '|---|---:|---:|---|']
    for r in rows:
        note = '仅为已完成分段的观测下界；缺少中断前分段的最终峰值' if r['speculative_gpu_peak_incomplete'] else '同 Worker/Replica；整卡观测'
        lines.append(f"| {r['dimension'].title()} | {r['baseline_gpu_peak_gib']:.3f} | {r['speculative_gpu_peak_gib']:.3f} | {note} |")
    lines += ['', f"原版全局峰值约 **{o['baseline_gpu_peak_gib']:.3f} GiB**；改进版已完成分段的最高观测约 **{o['speculative_gpu_peak_gib_lower_bound']:.3f} GiB**。总体接近，不能声称显著省显存；Noise 的不完整观测尤其不能用来计算可靠的显存降幅。数据来自框架的整卡采样，包含模型、仿真和驱动，非模型专属分配量。", '',
              '## 5. 加速来自哪里', '',
              '原版按 10 个时间位置串行求流速度。改进版用上一步速度外推下一状态，将当前状态和预测状态组成 batch=2，用同一个动作专家并行求两处速度；对目标单步纠正状态与预测状态计算全动作张量 RMS 误差，≤0.02 且处于可接受的中间网格位置时推进两步，否则采用目标纠正值推进一步。固定阈值在正式评测前给定，没有根据正式结果调参。', '',
              f"正式已完成任务共 **{call_count:,} 次调用**，接受 **{accepted:,}/{attempts:,}（{100*accepted/attempts:.4f}%）**。按照代码中的计数，每次接受减少一次串行网格迭代，平均串行专家调用约从 10 次降至 **{10-accepted/call_count:.3f} 次**；但每次计算两个位置，不能据此宣称 FLOPs 减少 40% 或整体加速 1.67×。观测前缀在一个采样调用中复用，原版也有前缀缓存；没有增加或训练独立 Draft Model。", '',
              '高接受率只是当前 RMS 阈值下的局部接受率，不是成功置信度、准确率保证或精确分布保持证明。误差统计覆盖完整动作张量，未证明该阈值在可执行动作维度和不同任务上都校准良好。该方法是连续动作流的近似推测采样实现，不是传统 Token 级无损推测解码，也未宣称复现了某篇特定论文。', '',
              '## 6. 完整性、恢复与复现', '',
              '- 两版各 10,030 条唯一完成记录，任务 ID、episode_index=0、任务配置字段和 BDDL 指令逐一一致。完成记录都有对应 episode_end；28 个原版完整分段及改进版完整分段均无记录的 OOM/Worker 失败。',
              '- 改进版第一次 Noise/libero_10 已完成的 359 条，与重启后剩余 90 条严格不相交，合并为 449 条；旧的 8 条未完成 Episode 从新队列重跑。延迟只纳入完成 Episode 的调用，排除旧未完成调用。',
              '- 首次 Noise 部分分段没有完整 result.json，GPU 峰值缺失；其已有完成记录和阶段指标保留，未伪造完整峰值或声称该部分的恢复统计完整。独立 smoke 不计入正式结果。',
              '- 分析使用 Python 标准库，运行 `python scripts/analyze_two_position.py BASELINE.tar.gz SPECULATIVE.tar.gz OUT_DIR`，再运行 `python scripts/render_two_position_report.py OUT_DIR reports/2026-11-23-two-position`。',
              f"- 采样器 SHA-256：`{s['protocol']['sampler_sha256']}`。", '',
              '归档校验值：', '',
              f"- 原版：`{s['archive_sha256']['baseline']}`",
              f"- 改进版：`{s['archive_sha256']['speculative']}`", '',
              '原始日志、逐 Episode 配对 CSV、配置和源码快照放私有 [HF Dataset 实验目录](https://huggingface.co/datasets/byk666666/prp-libero-plus-data/tree/main/experiments/2026-11-23-two-position)。公开仓库仅放报告、汇总表、分析脚本及独立采样器。具体上传 revision 另存发布收据。', '',
              '## 7. 下一步优先级', '',
              '先将本轮定位为“观察到约一成延迟改善、成功率总体接近、硬件门槛未降低”的可复现原型。随后对发生翻转的 570 条任务及代表性稳定任务做预先固定的多种子配对实验，重点检查 Noise 的退化；预留验证集校准阈值，明确非劣效容忍度，再尝试更激进的并行位置或降低显存。正式测试集不用于反复挑阈值。', '']
    (destination/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    shutil.copyfile(source/'summary.json',destination/'summary.json')
    shutil.copyfile(source/'comparison.csv',destination/'comparison.csv')
    print(destination/'report.md')


if __name__ == '__main__':
    main()
