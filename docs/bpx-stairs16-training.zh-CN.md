# 16 厘米台阶训练

2026-11-05：当前目标已改为 [15 cm 台阶](bpx-stairs15-training.zh-CN.md)。以下为历史16 cm方案。

历史目标为 **16 cm**。使用 `bpx_stairs16_teacher` 训练感知教师，评估合格后使用 `bpx_stairs16_blind` 蒸馏盲学生。

课程为 4、6、8、10、12、14、16 cm 七级，最高级精确为 16 cm；踏面深度 30 cm，上楼/下楼各占一半，初始级别为 4～8 cm。完整梯段晋级条件、30 Nm 力矩上限、20 rad/s 速度上限与教师 235 维/盲学生 900 维输入沿用现有设置。

从兼容教师检查点恢复（`max_iterations` 是额外训练轮数）：

```bash
./bpx.sh train --task bpx_stairs16_teacher --num_envs 512 \
  --resume --checkpoint /absolute/path/model_300.pt \
  --max_iterations 5700 --seed 2026 --run_name stairs16_v1 \
  --headless --device cuda:0
```

512 环境在 RTX 2080 SUPER 上约 1.4 秒/轮，5700 轮约 2 小时 15 分钟，每 100 轮保存。恢复网络与优化器，仿真 episode 和地形课程从新环境开始。

评估目标同步改为 16 cm（先测教师，不能据此宣称盲学生或实机达标）：

```bash
python scripts/evaluate_stairs.py --task bpx_stairs16_teacher \
  --checkpoint /absolute/path/final_model.pt --step_height 0.16 \
  --num_envs 64 --speed 0.30 --seed 2026 \
  --output /absolute/path/stairs16-forward.json --device cuda:0
```

用 `--backward` 测倒退，用 `--robust` 测带噪声与随机扰动的表现；重复种子 2026、2027、2028，并分别报告上行/下行首轮完整梯段通过率。
