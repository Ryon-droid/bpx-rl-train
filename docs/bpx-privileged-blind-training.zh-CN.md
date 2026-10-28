# 训练用地形特权信息，部署用盲策略

适用条件：目前机器人只有 IMU、关节编码器和速度指令，相机后续安装。先在仿真中使用地形真值训练教师，再训练仅依赖本体观测历史的学生，最后部署学生。仿真地形真值由射线扫描生成，不需要真实相机或 RGB 渲染。

## 两份 ETH 论文如何取舍

《Learning Quadrupedal Locomotion over Challenging Terrain》正文 Fig. 4、Teacher/Student policy 小节采用特权教师、历史学生两阶段：教师可访问地形、接触、摩擦等仿真真值，学生只用本体历史；原文还使用 TCN、潜变量监督、PMTG/IK 和自适应地形课程。

《Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning》3.2 节把 108 个地形高度测量直接输入策略；4.3 节实机高度图来自 LiDAR。因此不能把该策略的高度输入在部署时删掉或全部置零，就当作盲策略。可以借鉴它的 PPO、并行仿真、课程和域随机化。

本次采用前一篇的两阶段思路，复用项目固定的 Isaac Lab 2.3.0 / RSL-RL 3.0.1。实现是**20 帧历史 MLP + 官方在线动作蒸馏基线**，不是论文 TCN、潜变量损失与 PMTG 的完整复现。原文补充 S9 也说明仅模仿动作的局限，实际能力需要独立验证；学生达不到教师表现时，再考虑 TCN 和潜变量监督。

## 当前代码与新增任务

原有三个任务已有非对称 Actor–Critic：Actor 45 维本体输入；Critic 使用这 45 维加 3 维真值线速度。它们没有地形扫描、历史或教师蒸馏，保留作为对照。

新增四个独立任务：

| 任务 | 训练方式 | 执行动作的网络输入 | 用途 |
|---|---|---|---|
| `bpx_rough_teacher` | PPO | 45 本体 + 3 真值线速度 + 187 高度 = 235 维 | 混合地形教师，只用于仿真 |
| `bpx_stairs_teacher` | PPO | 同上 | 台阶教师，只用于仿真 |
| `bpx_rough_blind` | 在线蒸馏 | 45 × 20 = 900 维本体历史 | 混合地形盲学生 |
| `bpx_stairs_blind` | 在线蒸馏 | 同上 | 台阶盲学生 |

教师/学生均输出项目现有顺序的 12 个关节位置残差。高度扫描范围 1.6 × 1.0 m，分辨率 0.1 m，包含边界共 17 × 11 = 187 点，跟随机身 yaw。它是训练真值，不是未来相机输出格式。

蒸馏时学生执行动作、教师在学生实际访问的状态上给出动作标签，优化动作 MSE。学生输入映射只有 `proprio_history`；`policy + privileged + terrain` 只供冻结教师打标签。历史仍包含原有观测噪声，重置由 Isaac Lab 历史缓冲处理，地形难度沿用现有课程，不自动恢复教师训练时的课程进度。

## 执行顺序

以下命令在仓库根目录、已安装项目依赖的 `bpx_train` 环境执行。`1024` 环境、`5000`/`3000` 迭代是起始设置，不是已验证的收敛预算。先跑少量环境确认显存、射线和维度，再扩大训练。

```bash
conda activate bpx_train
python scripts/list_envs.py
./bpx.sh contract --task bpx_rough_teacher --headless --device cuda:0
./bpx.sh contract --task bpx_rough_blind --headless --device cuda:0

# 1. 从头训练地形教师。
./bpx.sh train --task bpx_rough_teacher \
  --num_envs 1024 --max_iterations 5000 --seed 2026 \
  --run_name privileged_v1 --headless --device cuda:0
```

教师保存在 `logs/bpx_rough_teacher/<运行目录>/model_<编号>.pt`。先用同任务播放与定量评估，确认能处理目标地形，再蒸馏。将下列绝对路径替换为实际产生的检查点；不要使用仓库自带的旧 45 维平地/台阶模型作为新教师。

```bash
TEACHER=/absolute/path/to/logs/bpx_rough_teacher/RUN/model_4999.pt
./bpx.sh play --task bpx_rough_teacher --checkpoint "$TEACHER" \
  --eval --robust --num_envs 32 --headless --device cuda:0

# 2. 新建盲学生训练；不要附加 --resume。
./bpx.sh train --task bpx_rough_blind --teacher_checkpoint "$TEACHER" \
  --num_envs 1024 --max_iterations 3000 --seed 2026 \
  --run_name blind_history_v1 --headless --device cuda:0
```

学生保存在 `logs/bpx_rough_blind/<运行目录>/`。学生断点续训使用 `--resume --checkpoint "$STUDENT"`，不再传 `--teacher_checkpoint`。新教师和旧 45 维模型输入尺寸不同，不能直接完整恢复旧模型/优化器；本方案从头训练教师。

```bash
STUDENT=/absolute/path/to/logs/bpx_rough_blind/RUN/model_2999.pt
./bpx.sh play --task bpx_rough_blind --checkpoint "$STUDENT" \
  --eval --robust --num_envs 32 --headless --device cuda:0

# 3. 只导出学生用于部署。
./bpx.sh export --task bpx_rough_blind --checkpoint "$STUDENT" \
  --output_dir outputs/bpx_rough_blind --device cuda:0

# 4. 复用相同历史排列做 MuJoCo 平地接口验证。
./bpx.sh mujoco --policy outputs/bpx_rough_blind/bpx_rough_blind_900.pt \
  --history_length 20 --headless --eval
```

专攻台阶时，将教师任务改为 `bpx_stairs_teacher`、学生任务改为 `bpx_stairs_blind`，并使用对应的教师检查点。混合地形更适合作为通用行走起点；不要仅凭训练 reward 或台阶单任务结果断言实机通用性。

## 部署接口必须同步修改

学生导出文件为 `bpx_rough_blind_900.pt` / `.onnx`，旁边的 JSON 记录输入、时间步长和历史排列。官方 ONNX 导出默认 batch 为 1。部署端不能继续把 `[1,45]` 直接送进这个模型。

每个 20 ms 控制周期组装一次原来的 45 维观测，按原来的关节顺序、裁剪和缩放处理；将 20 帧送入学生。只需要 IMU、本体关节信息、速度指令和上一动作，不需要真值机身线速度、足端力、相机或高度图。

**历史按观测项分块，块内时间从旧到新**，与 Isaac Lab 2.3.0 一致，不能简单把 20 个完整 45 维帧直接展平。顺序是：20 帧角速度、20 帧重力、20 帧指令、20 帧关节位置、20 帧关节速度、20 帧上一动作；各块维度分别为 60、60、60、240、240、240。

`scripts/policy_contract.py` 的 `ProprioHistory` 是不依赖 Isaac 的参考实现，MuJoCo 已接入相同逻辑：

```python
from policy_contract import ProprioHistory

history = ProprioHistory(length=20)
# 控制启动/重启/仿真回合重置时调用，不能跨回合保留历史。
history.reset()
# 以下每个控制 tick 只调用一次；frame_45 是已经缩放后的原策略观测。
obs_900 = history.append(frame_45)
actions_12 = policy(obs_900[None, :])
```

启动时用第一帧填充完整历史；之后滚动更新。模型之外保存历史，模型本身不保存隐藏状态。SDK 接线不在本训练仓库中，实机控制程序必须同步接入此接口；原有 45 维部署程序不能原样加载新学生。

## 验证范围与后续工作

测试覆盖历史时间/分块顺序与重启、教师权重加载、学生更新而教师参数不变，以及改动地形/真值速度不改变学生动作。完整仿真还必须运行上面的 `contract`、短训练、教师/学生同条件评估。CPU 网络测试不代表台阶通过率或 sim-to-real 已验证。

2026-10-28 本地验证记录：

- `python -m pytest -q tests`：18 项通过，包含原有回归测试与新增盲学生测试。
- 使用 Isaac Lab 2.3.0 原始 `configclass` / RSL-RL 配置源码检查新配置的实例化和序列化；仿真对象用桩替代，不代表已启动 Isaac。
- 使用实际 RSL-RL 3.0.1 与本仓库的完整网络配置，在合成 CPU 观测环境执行一轮蒸馏、保存/恢复、学生 TorchScript/ONNX 导出及推理一致性检查，通过。
- 用上述合成测试权重执行 1 秒 MuJoCo 平地播放，验证 900 维历史接口；该权重不是训练成品，不能用于部署。
- 测试运行时为 Python 3.12 / PyTorch 2.14.1+cpu，与训练栈的 Python 3.11 / PyTorch 2.7.0 不同；ONNX 测试使用旧导出路径以对应 PyTorch 2.7.0。最终须在项目固定训练环境中重新跑 `contract`、训练和评估。
- 当前本机没有 Isaac Lab 和可用的 NVIDIA 训练环境，未运行真实地形训练，也未验证收敛、复杂地形表现或实机能力。

评估至少覆盖台阶高度、上/下台阶、前/后/横向、斜坡、推扰与噪声；分别统计不重置的通过率、跌倒率、跟踪误差和动作/力矩峰值。现有 `play --eval` 是跟踪与失败计数工具，回合会自动重置，其计数不能直接当作台阶通过率。MuJoCo 目前只有平地，也不能验证复杂地形能力。

盲策略通过触地后的本体反馈适应地形，无法保证提前看见远处障碍、深沟或选择视觉落脚点。相机后续安装后，应新增与相机测量、延迟、噪声匹配的感知策略及训练流程；保留这个盲学生作为可评估的基础策略。不能把图像直接接到 900 维本体历史接口上。
