# BPX 参数辨识软件准备

2026-11-06。此目录对应吊架完成前的软件准备，采用 PACE 的“固定基座、轨迹匹配、49 维 CMA-ES”思路。现有强化学习任务和参数不自动更改，原训练保持暂停。

## 当前可用内容

- `bpx_sysid/core.py`：SDK/策略关节名称映射，URDF 限位，绝对角度扫频规划，参数版本与单位。
- `bpx_sysid/capture.py`：厂商 SDK 只读普通状态采集，记录去重后的 q/dq/tau、时间戳、IMU、温度与电源诊断。
- `bpx_sysid/data.py`：实际采集日志检查，统一时间网格，指令零阶保持、反馈插值，拒绝长间断/未知前馈/增益不匹配。
- `bpx_sysid/mujoco_backend.py`：CPU 固定机身重放与已知参数合成数据；用于快速软件测试。
- `bpx_sysid/isaac_backend.py`：Isaac Lab 2.3 / PhysX 固定机身重放，按候选环境写入 armature、黏性/静动态关节摩擦，再从物理引擎读回验证。
- `bpx_sysid/optimize.py`：49 维 CMA-ES；每代保存实际最佳候选，最后对最佳候选和分布均值分别重放。拟合阶段流式累计误差，不保留全体候选的整段轨迹。

参数顺序为 armature(12)、黏性摩擦(12)、库仑摩擦(12)、编码器偏置(12)、全局延迟(1)。所有数组按具名关节转换。`q_encoder=q_physics-bias`；延迟同时记录整数物理步和秒。`command` 与 `effort` 模式明确区分。

这是独立的 BPX 适配工具，不是对上游完整实验的复现声明。这里显式计算位置 PD 和峰值扭矩饱和，不把额定速度冒充空载速度，也未虚构 DC 电机斜线包络。MuJoCo 与 PhysX 的接触/摩擦实现存在差异，跨引擎重放不应要求误差为零。正式拟合优先使用与后续训练一致的 PhysX。

## 尚需吊架和实机联调的内容

**当前不含自动向实机下发扫频的程序。** 只读采集可以先检查 SDK、单位、方向、网络和频率；固定机身后，再确认本机增益单位、看门狗/失联行为、厂商停止方式及机械碰撞范围，接入受控扫频下发和同包高频记录。不能把生成的离线轨迹直接当作实机动作许可。

只读程序使用 `RequestRobotState`，不构造 `JointLevelControl`。后者即使不发送关节命令，也可能切换主机控制模式。普通状态请求上限 200 Hz；代码不把轮询频率当作实际反馈率。时间戳前后夹读只能降低缓存撕裂风险，不是原子网络包记录。HighRate 的接收侧采集和命令实际发送时刻仍需在实机联调时适配。

电机扭矩采用髋 20 Nm、膝 40 Nm；额定速度 12/10 rad/s 仅作为保守运行设置。PD=30/1、姿态 `[0×4,0.8×4,-1.5×4]` 是仿真起点，不代表已确认的实机试验增益或吊架姿态。

## 无需机器狗的快速验证

在仓库根目录、独立的 Python 环境执行。CPU 工具无需 Isaac 或 torch；GPU 工具使用已有 Isaac 环境。可安装 `requirements-sysid.txt`，不要为此升级原来的 Isaac/PyTorch 环境。

```bash
python -m pip install -r requirements-sysid.txt

# 20 秒 chirp + 前后保持，仅规划，不联网，不运动。
./bpx.sh sysid plan --output outputs/sysid/plan \
  --joint fl_hip_pitch_joint --amplitude 0.05 --f0 0.1 --f1 1.0

# 用已知参数生成固定机身 MuJoCo 数据，再独立重放真值。
./bpx.sh sysid synthetic --output outputs/sysid/synthetic --duration 4
./bpx.sh sysid inspect --data outputs/sysid/synthetic
./bpx.sh sysid replay --data outputs/sysid/synthetic \
  --parameters outputs/sysid/synthetic/truth.json --output outputs/sysid/replay-truth

# 短跑仅验证优化链路，不代表 49 个参数已经可辨识/恢复。
./bpx.sh sysid fit --data outputs/sysid/synthetic \
  --bounds calibration/templates/synthetic-bounds.json \
  --population 8 --generations 3 --output outputs/sysid/fit-smoke
```

`outputs` 每个实验必须使用新目录，避免覆盖旧数据。轨迹输出 CSV、NPZ、控制模型约定和角度/速度/加速度检查结果；尚未进行腿间/腿身碰撞检查。默认仅单关节、小幅低频起步；完整 49 维拟合需要覆盖全部关节的多段独立实验。

## 固定机身 PhysX 检查与拟合

以下启动 GPU 仿真，完全不连接机器狗。若 NVIDIA 许可提示尚未处理，请先按环境安装流程处理许可。

```bash
# 验证参数实际写入物理引擎与重放链路；数据来自 MuJoCo，允许跨引擎误差。
./bpx.sh sysid-isaac --mode check --data outputs/sysid/synthetic \
  --parameters outputs/sysid/synthetic/truth.json \
  --output outputs/sysid/physx-check --headless --device cuda:0

# 在 PhysX 里用已知参数生成合成数据。
./bpx.sh sysid-isaac --mode synthetic --data outputs/sysid/synthetic \
  --parameters outputs/sysid/synthetic/truth.json \
  --output outputs/sysid/physx-synthetic --headless --device cuda:0

# 先少量候选检查单代，再决定正式规模/代数，并先报告预计耗时。
./bpx.sh sysid-isaac --mode fit --data outputs/sysid/physx-synthetic \
  --parameters outputs/sysid/physx-synthetic/truth.json \
  --bounds calibration/templates/synthetic-bounds.json \
  --population 8 --generations 3 --seed 2026 \
  --output outputs/sysid/physx-fit-smoke --headless --device cuda:0
```

示例边界文件明确标记 `synthetic_only=true`，无法用于实机数据。实机边界需要根据响应、CAD 和厂商资料另建并说明依据；只切换标记不能使示例范围变成实测范围。首轮不做 4096 环境或未经测速的长时拟合。

## 吊架完成后的首次调试

1. 固定机身，腿部悬空，检查支撑、碰撞余量和厂商停止方式。
2. 在实际接狗的电脑安装对应架构/Python 版本的厂商 SDK（导入名 `bpx_sdk`）。先只读采集，不进入关节控制模式：

```bash
./bpx.sh sysid collect-readonly --ip <机器狗实际IP> \
  --seconds 20 --rate 100 --output outputs/sysid/read-only.jsonl
```

`192.168.1.10` 是已有训练机地址，不自动当作机器狗 IP。采集结束显示实际不同反馈帧数/秒、重复轮询和不一致读取数量；即使无反馈也保留诊断日志，并以失败状态退出。

3. 确认关节顺序、rad 单位、实际增益、控制固件和基座稳定性。用原生采样率保存各传感器的新鲜度；IMU 不能代替机械固定。
4. 再接入受控单关节小幅激励，记录真实命令发送和反馈；通过后扩展扫频与多关节实验。第一版要求目标速度和前馈力矩为零。
5. 留出完整独立实验作验证，不随机拆时间点；原始日志、固件和模型哈希独立归档。

## 接入实际激励日志的数据格式

输入 JSONL 应来自同一采集会话，所有时间用同一主机单调时钟。元数据必须包含真实 `session_id`，拆分实验片段时保留它。以下是结构说明，数组均需 12 个值，不是直接运行的日志：

```text
metadata: kind, source="hardware", session_id, joint_names,
          time_basis="host_monotonic_ns", sequence_kind="packet_counter",
          capture_semantics="sequence_bracketed_cache", firmware, SDK版本
feedback: kind, host_monotonic_ns, seq, q, dq, tau
command:  kind, host_monotonic_ns, sent_ok, q_des, dq_des, tau_ff, kp, kd
```

如果采用原子接收侧记录，需另行适配/确认 `capture_semantics`；当前预处理器只接受已实现的缓存夹读格式。普通毫秒设备时间戳使用 `sequence_kind="device_timestamp_ms"`，不能按相邻差值统计丢包。`sent_ok` 必须对应实际发送事件；SDK 缓存更新返回值不能冒充驱动已收到命令。若 SDK 未暴露发送事件，需要在发送/接收侧补日志。

```bash
# contract 文件必须按实际增益、固定姿态、物理步和延迟模式核对。
./bpx.sh sysid prepare --raw outputs/sysid/excitation.jsonl \
  --contract /absolute/path/to/reviewed-session-contract.json \
  --max-gap 0.03 --output outputs/sysid/prepared
```

不把目标轨迹与反馈各自移动到“最重合”再拟合延迟。当前主机缓存时间基的延迟包含轮询和缓存不确定性；需更精确的时序时，应先建立设备—主机时钟映射和同包采集。预处理不做自动抗混叠滤波；当原始信号存在接近/超过重采样 Nyquist 的成分时，应先验证带宽并设计保相位的降采样处理。

## 结果与后续训练

拟合输出 `candidate_*.json`、`best.json`、`mean.json`、两套独立重放轨迹和 `report.json`。报告含逐关节 RMSE、最大误差、数据/URDF/代码哈希、实际仿真引擎、控制增益、延迟模式及步长。延迟搜索的连续值按向下取整成为实际物理步，文件中只记录有效整数步和秒数。

在留出实验上用 `sysid replay` 或 `sysid-isaac --mode replay` 复测。参数文件记录拟合数据和会话；不同数据哈希不代表独立实验，必须核对会话及实验条件。所有输出默认为 `calibrated=false`、`real_robot_ready=false`，不依据优化器收敛自动宣告可部署。

辨识通过后才讨论将延迟、偏置、惯量和摩擦接回教师/学生训练；目前工具不会自动改写训练执行器或实机零位，也不会恢复暂停的训练。

## 参考

- PACE 项目：https://pace.filipbjelonic.com/
- 上游源码：https://github.com/leggedrobotics/pace-sim2real ，分析参考提交 `7ad0f45db5ed01103543ddc25a812fef61a998ab`。
- 实施方案：工作区 `苏黎世联邦大学方案/BPX-PACE复现与参数辨识实施方案.md`。

## 本次软件验证记录（2026-11-06）

- 全仓库 42 项 CPU 测试通过，含 9 项辨识回归测试。
- 独立环境按 `requirements-sysid.txt` 安装后，9 项辨识测试通过；CPU MuJoCo 3.3.7 与现有环境的 3.17.0 均已检查。
- 默认单关节离线 chirp 峰值速度约 0.284 rad/s、加速度约 1.702 rad/s²；这些是轨迹规划量，不是实测性能。
- MuJoCo 已知参数数据的独立真值重放误差为零；8 候选、3 代 CMA-ES 短跑完成，验证了导出和最佳候选/均值的独立重放。
- RTX 2080 SUPER 上，Isaac Lab 2.3 / Isaac Sim 5.1 已验证固定基座、12 关节、零重复 PD 驱动、参数写入和物理引擎读回；4 候选、1 代 GPU 拟合完成，单代约 3.55 秒，仅是该小规模软件检查的耗时。
- PhysX 合成数据的同引擎真值独立重放误差为零。错误 warmup 参数以失败状态退出；成功路径使用 Isaac 官方快速清理方式，结果文件关闭后正常退出。
- 没有连接或驱动机器狗，没有实机标定结果，没有恢复原强化学习训练；验证结束后 GPU 只剩桌面进程。
