# 15 厘米台阶：诊断、专项教师微调、盲学生

> 历史配置说明：本文中的统一 30 N·m / 20 rad/s 为原实验参数。2026-11-06 运行配置已按电机规格更正；本文历史结果不改写，详见[电机规格核对](bpx-motor-spec-alignment.zh-CN.md)。

2026-11-05：当前目标已调整为[17cm](bpx-stairs17-training.zh-CN.md)。以下保留15cm试验方案；该轮最终教师出现上楼回退，未通过资格评估。训练有地形感知，最终部署使用无相机的历史观测学生。已有16 cm教师不能直接作为盲狗部署。

## 任务与地形

- `bpx_stairs15_teacher`：普通课程；`bpx_stairs15_focus_teacher`：已有教师的专项微调；`bpx_stairs15_blind`：盲学生蒸馏。
- 实际梯级为4、6、8、10、12、14、15 cm，踏面30 cm，上下楼地形各50%。普通课程初始4～8 cm。
- 专项微调初始覆盖全部七档；每次重置有50%概率重采样12/14/15 cm，其余沿用完整梯段晋降级规则。在下楼地形，纯纵向命令中80%为倒退；上楼仍前后各半。纵向、侧向、转向、混合、站立比例仍为70/10/5/10/5%。高台阶覆盖率是采样统计，不能当作通过率。
- 保留30 Nm力矩、20 rad/s关节速度、原终止条件、原奖励、质量/摩擦/重心/推力及观测噪声。教师235维，盲学生900维（45维×20帧），12维动作。

## 先诊断

使用原16 cm教师在15 cm完整梯段上建立基线。`evaluate_stairs.py`支持 `--condition nominal|noise|material|mass|com|push|robust`；单因素只启用对应项，robust启用全部训练随机化。根部和关节重置使用独立固定随机序列，并保存实际初始状态、质量、质心、材料参数及10 Hz姿态轨迹。须验证初始状态一致后，才作配对比较。material同时改变静/动摩擦和恢复系数，不应称为纯摩擦辨识。

```bash
python scripts/evaluate_stairs.py --task bpx_stairs15_teacher \
  --checkpoint /absolute/path/teacher.pt --step_height 0.15 \
  --num_envs 64 --speed 0.30 --seed 2026 --condition nominal \
  --output /absolute/path/baseline-forward.json --device cuda:0
```

加 `--backward` 测倒退。每机器人只统计首次尝试；必须沿命令方向走完3.3 m，横向偏移小于0.75 m，满足垂直进度且没有身体碰地或姿态终止。默认20 s；低速时自动延长至至少 `6/speed` 秒。固定高度评估会清除七档课程，确保全部梯级为指定高度。

## 首轮微调

从原16 cm最终教师加载权重和观测归一化，使用新优化器，固定学习率1e-4。不要使用 `--resume` 恢复旧优化器。

```bash
./bpx.sh train --task bpx_stairs15_focus_teacher --num_envs 512 \
  --init_checkpoint /absolute/path/stairs16/model_5999.pt \
  --max_iterations 2000 --seed 2026 --run_name stairs15_focus_v1 \
  --headless --device cuda:0
```

RTX 2080 SUPER实测旧任务约1.3 s/轮，2000轮预计约45分钟，另加启动和约10～15分钟复测；每次启动须重新告知预计时间，并依据当前速度修正。每100轮保存，最终预期`model_1999.pt`。实际用时以运行日志为准。

## 复测与后续

在15 cm、±0.30 m/s下，分别测nominal与robust，种子2026/2027/2028，每种子32上楼+32下楼。按条件、前进/倒退、上楼/下楼分别合计96次。每一组通过率至少95%才允许进入下一阶段，另列物理失败和漂移。再复测8/12 cm，检查低台阶能力有无退步。`scripts/summarize_stair_suite.py`检查报告完整性并输出达标结论；失败或缺报告不得自动进入学生训练。

教师通过后才进行盲学生蒸馏，先用短跑测速度并估时，再正式训练与同条件评估。当前未证明盲学生或实机15 cm能力。电机辨识结果验证后，更新执行器模型并做针对性微调、回归评估；无需因为有了辨识数据就默认从头重训。硬件辨识应按既有辨识执行方案独立进行。
