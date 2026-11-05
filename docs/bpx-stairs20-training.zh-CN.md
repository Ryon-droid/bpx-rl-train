# 20 厘米台阶：教师优化与盲学生

2026-11-05：当前训练目标已改为 [16 cm 台阶](bpx-stairs16-training.zh-CN.md)，请使用 `bpx_stairs16_teacher` / `bpx_stairs16_blind`。以下保留 20 cm 方案供参考。

目标是阶高 **20 厘米**，当前按踏面深 **30 厘米**训练。实际楼梯踏面深度、阶沿和摩擦不同，需要相应地形与评估。这里定义训练目标，不代表已经具备 20 厘米通过能力。

## 独立任务与课程

`bpx_stairs20_teacher` 保持 235 维教师输入，`bpx_stairs20_blind` 保持 900 维本体历史输入，两者都输出 12 维动作。原有混合地形和低台阶任务保留。

- 地形网格 9 行，阶高精确为 4、6、8、10、12、14、16、18、20 厘米。对 Isaac Lab 行内随机难度量化，确保最高行确实是 20 厘米。
- 一半为从中心低平台出发的上行台阶，一半为从中心高平台出发的下行台阶；包含正向和倒退指令，训练默认 70% 纵向、10% 横向、5% 转向、10% 组合、5% 站立。
- 从 4–8 厘米开始。只有完成整段、到达外围平台且速度跟踪达标的无失败回合才能升级；原地站稳或只走到第一阶不能升级。
- 到外围平台边界结束该回合，避免跨入相邻地形。该边界是时间截断；同时发生物理失败时仍判失败。最高级通过后按 Isaac Lab 默认行为返回随机等级，继续保留较低难度训练。
- 纵向速度不超过 0.40 m/s，横向不超过 0.15 m/s，转向不超过 0.30 rad/s。减少身体竖向速度、腿部姿态偏差和动作变化惩罚，允许高台阶所需的身体升降与屈腿。
- 保留 30 Nm 力矩上限、关节范围、自碰撞、物理跌倒判断，并明确设置 URDF 对应的 20 rad/s 关节速度上限；不通过提高电机能力获得虚假的通过效果。

BPX URDF 总质量约 15.16 kg，大腿和小腿各约 23 cm。20 cm 已是有挑战的阶高，最终要结合电机实测、碰撞几何和实机受控测试判断。

## 从已有教师开始微调

```bash
./bpx.sh contract --task bpx_stairs20_teacher --headless --device cuda:0
./bpx.sh contract --task bpx_stairs20_blind --headless --device cuda:0
./bpx.sh train --task bpx_stairs20_teacher --num_envs 512 \
  --max_iterations 6000 --seed 2026 --run_name stairs20_v1 \
  --init_checkpoint /absolute/path/to/rough_teacher/model_4999.pt \
  --headless --device cuda:0
```

`--init_checkpoint` 仅加载兼容的模型与观测归一化权重，重新建立优化器，并将当前训练迭代归零；保存到新的实验目录。与恢复同一次运行的 `--resume` 不同，不能与之同时使用，也不能用于学生蒸馏。原有 45 维模型不能初始化该 235 维教师。

6000 次迭代只是首轮优化预算，不是能力达标保证。RTX 2080 SUPER 上 512 环境短试训约 1.4 秒/迭代，预计约 2 小时 20 分钟，实际以日志为准。检查点每 100 次迭代保存。

## 单次台阶通过评估

```bash
python scripts/evaluate_stairs.py --task bpx_stairs20_teacher \
  --checkpoint /absolute/path/model.pt --step_height 0.20 \
  --num_envs 64 --speed 0.30 --seed 2026 \
  --output /absolute/path/stairs20-forward.json --device cuda:0
```

加 `--backward` 测倒退，加 `--robust` 保留训练的观测噪声、摩擦/质量随机化和推扰。重复多个种子，并分别测试 8、12、16、20 cm。测试保留小幅有种子的起始位置、朝向和关节姿态变化。

每只狗仅记录第一次尝试。到达目标方向的平台、横向偏离小于 0.75 m、完成对应高度变化且没有物理失败才算成功；超过时间仍未完成判未通过。完成后自动重置的轨迹不计入第二次机会。上行/下行分别统计。固定指令的 `play --eval` 可以补充检查跟踪误差，但其失败/重置次数不能替代整段通过率。

## 盲学生与实机

先评估教师在 20 cm 上的正向、倒退、上行和下行能力。只有教师表现可接受后，再用其检查点启动 `bpx_stairs20_blind --teacher_checkpoint PATH` 蒸馏，并在相同测试下比较。盲学生只能根据触地后的本体反馈适应；教师成绩不等于盲学生成绩，更不等于实机能力。
