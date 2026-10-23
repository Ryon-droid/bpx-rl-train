# MIRRORME 强化学习训练

[English](README.md) | [简体中文](README.zh-CN.md)

这是一个面向 MIRRORME BPX 四足机器人的开源强化学习训练基线。工程采用 Isaac Lab 官方外部项目结构，使用 Manager-Based Environment，并通过官方 RSL-RL 集成训练 PPO 策略。

本仓库只包含仿真训练、策略评估和模型导出，不包含机器人 SDK 接入或实机部署内容。

## 任务与命名

工程注册平地、多地形和独立台阶三个任务：

| 名称 | 值 | 含义 |
|---|---|---|
| 任务 ID | `bpx_flat` | 平地全向速度跟踪任务 |
| 任务 ID | `bpx_rough` | 平地、起伏、缓坡和低台阶混合训练 |
| 任务 ID | `bpx_stairs` | 独立的金字塔台阶行走任务 |
| 实验名称 | `bpx_flat` | `logs/` 下的 RSL-RL 日志目录名 |
| 实验名称 | `bpx_rough` | 多地形任务的默认日志目录名 |
| 实验名称 | `bpx_stairs` | 独立台阶任务的日志目录名 |
| 运行目录 | `YYYY-MM-DD_HH-MM-SS[_run_name]` | 一次独立训练 |
| 检查点 | `model_<iteration>.pt` | RSL-RL 保存的模型，例如 `model_2999.pt` |
| 导出模型 | `bpx_flat_45.pt` / `bpx_flat_45.onnx` | TorchScript 和 ONNX Actor |
| 导出模型 | `bpx_rough_45.pt` / `bpx_rough_45.onnx` | 多地形任务的 Actor 导出 |
| 导出模型 | `bpx_stairs_45.pt` / `bpx_stairs_45.onnx` | 独立台阶任务的 Actor 导出 |

策略跟踪机身坐标系速度指令 `[vx, vy, wz]`，支持平地、轻度复杂地形和台阶。三个任务共用 45 维 Actor、48 维 Critic 和 12 维动作接口。任务中没有轮关节、高度扫描器、历史编码器、DWAQ/CENet 模块、循环网络或自定义 RL Runner。

## 1. 环境要求

本教程固定使用以下软件栈：

| 组件 | 版本 |
|---|---:|
| 操作系统 | Ubuntu 22.04 LTS，x86-64 |
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.0 |
| PyTorch / torchvision | 2.7.0 / 0.22.0，CUDA 12.8 wheel |
| RSL-RL | 3.0.1，由 Isaac Lab 的 `rsl-rl` extra 安装 |

Isaac Sim 训练需要兼容的 NVIDIA GPU 和驱动。NVIDIA 对 Isaac Lab 2.3.0 软件栈的建议是至少 32 GB 内存和 16 GB 显存。pip 安装还要求 glibc 2.35 或更高版本，Ubuntu 22.04 满足该要求。

## 2. 克隆与安装

克隆仓库即可，BPX 的 URDF 和网格资源已直接包含在仓库中：

```bash
git clone https://github.com/Ryon-droid/bpx-rl-train.git mirrorme_rl_train
cd mirrorme_rl_train
```

运行日志和缓存按 `.gitignore` 排除。已选定的训练检查点、导出策略及训练配置保存在 [`pretrained/`](pretrained/)，普通克隆即可获取，不需要 Git LFS 或额外下载模型。已有仓库执行 `git pull --ff-only` 即可更新。

### 自动创建干净环境

在仓库根目录执行：

```bash
./bpx.sh setup
```

该命令会创建名为 `bpx_train` 的新 Conda 环境，安装固定版本的 Isaac Sim、Isaac Lab、PyTorch、RSL-RL 和本项目，最后运行无界面的接口检查。为避免破坏已有环境，脚本不会复用或修改同名 Conda 环境。

如需指定另一个新环境名称：

```bash
./bpx.sh setup my_bpx_environment
```

如果只安装依赖、暂时不启动 Isaac Sim：

```bash
./bpx.sh setup my_bpx_environment --skip-contract
```

安装完成后激活环境：

```bash
conda activate bpx_train
```

Isaac Sim 第一次启动时可能下载扩展缓存，并要求接受 NVIDIA Omniverse 许可协议。请自行阅读并回答终端提示；安装脚本不会代替用户接受协议。

### 手动安装

以下步骤用于故障排查或需要逐步控制安装过程的情况：

```bash
conda create -n bpx_train python=3.11 -y
conda activate bpx_train
python -m pip install --upgrade pip
python -m pip install "setuptools<81" wheel
python -m pip install --no-build-isolation flatdict==4.0.1
python -m pip install "isaaclab[isaacsim,rsl-rl]==2.3.0" \
  --extra-index-url https://pypi.nvidia.com
python -m pip install -U torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
./bpx.sh install
```

Isaac Lab 2.3.0 固定依赖旧版源码包 `flatdict==4.0.1`。先使用兼容的构建工具和 `--no-build-isolation` 安装它，可以避免新版 setuptools 引起的构建失败。

检查安装版本和 GPU 可见性：

```bash
python -c "from importlib.metadata import version; import torch; print('Isaac Lab:', version('isaaclab')); print('Isaac Sim:', version('isaacsim')); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

开始 GPU 训练前，`CUDA available` 必须为 `True`。

## 预训练模型：播放与继续训练

仓库包含两套独立训练的策略。两者均使用随机种子 `2026`、`1024` 个并行环境，训练 `3000` 次迭代，最终保存的迭代编号为 `2999`。每套模型及配套文件约 4.6 MB。

| 任务 | 原始训练运行 | 检查点 | 训练配置 |
|---|---|---|---|
| 平地行走 | `2026-10-02_16-11-56_baseline_v1` | [bpx_flat/model_2999.pt](pretrained/bpx_flat/model_2999.pt) | [环境配置](pretrained/bpx_flat/params/env.yaml)、[PPO 配置](pretrained/bpx_flat/params/agent.yaml) |
| 独立台阶 | `2026-10-22_12-03-41_stairs_from_scratch` | [bpx_stairs/model_2999.pt](pretrained/bpx_stairs/model_2999.pt) | [环境配置](pretrained/bpx_stairs/params/env.yaml)、[PPO 配置](pretrained/bpx_stairs/params/agent.yaml) |

每套包含完整 RSL-RL 检查点（模型与优化器状态）、`exported/` 下的 TorchScript 和 ONNX Actor、`params/` 下的配置快照，以及 `reports/` 下的历史评估 JSON/CSV。[manifest.json](pretrained/manifest.json) 记录原始路径、文件大小和 SHA-256 校验值。Isaac 播放与续训使用 `model_2999.pt`，推理或 MuJoCo 使用导出的 Actor。

台阶策略在独立的 `bpx_stairs` 任务中从随机初始化开始训练。地形为 100% 金字塔台阶，配置台阶高度范围 2～10 cm、踏面宽 30 cm、中央平台宽 3 m；训练速度范围为前后 ±0.5 m/s、横向 ±0.2 m/s、转向 ±0.5 rad/s。Actor 不含地形高度扫描输入。对应任务代码为 [bpx_stairs_env_cfg.py](source/mirrorme_rl_train/mirrorme_rl_train/tasks/locomotion/bpx_stairs_env_cfg.py)。

完成安装并激活 `bpx_train` 后，在仓库根目录执行：

```bash
# 平地模型播放
./bpx.sh play --task bpx_flat \
  --checkpoint pretrained/bpx_flat/model_2999.pt \
  --device cpu --num_envs 1 --seed 2026 --real_time

# 独立台阶模型播放
./bpx.sh play --task bpx_stairs \
  --checkpoint pretrained/bpx_stairs/model_2999.pt \
  --device cpu --num_envs 20 --seed 2026 --real_time
```

没有桌面显示的服务器，将 `--real_time` 换成 `--headless`；使用 GPU 仿真/推理时将设备改为 `--device cuda:0`。播放脚本也会在检查点旁导出 Actor，可能重写随仓库提供的导出文件。

在兼容的 GPU 上继续训练：

```bash
./bpx.sh train --task bpx_flat --resume \
  --checkpoint pretrained/bpx_flat/model_2999.pt \
  --num_envs 1024 --max_iterations 1000 --seed 2026 \
  --run_name flat_resume --headless --device cuda:0

./bpx.sh train --task bpx_stairs --resume \
  --checkpoint pretrained/bpx_stairs/model_2999.pt \
  --num_envs 1024 --max_iterations 1000 --seed 2026 \
  --run_name stairs_resume --headless --device cuda:0
```

这里的 `--max_iterations 1000` 表示额外训练 1000 次迭代。新的模型和配置快照保存在 `logs/<任务名>/` 下的新运行目录中。续训恢复模型、优化器状态和迭代计数，不恢复仿真状态、地形课程进度或随机数生成器状态。

随附 YAML 是当次训练的原始配置存档，保留历史绝对路径和 Python 专用 YAML 标签。训练/播放实际加载注册的 Python 任务与 PPO 配置及命令行覆盖项；指定检查点**不会自动加载**旁边的 YAML。修改配置或复现训练时，请对照这两份快照。

### 随附的历史评估结果

以下报告来自原训练记录，本次上传没有重新运行仿真。均使用种子 `2026`，运行一轮十阶段指令序列，共 100 秒；跟踪统计排除阶段切换/重置后的前 2 秒。

| 报告 | 环境数量 | 条件 / 初始地形等级 | 物理失败次数 | Episode 超时次数 |
|---|---:|---|---:|---:|
| [平地带扰动](pretrained/bpx_flat/reports/2026-10-07_14-51-37_721833_robust/summary.json) | 16 | 随机化 + 每 3～5 秒 XY 速度推扰 | 0 | 80 |
| [台阶标称条件](pretrained/bpx_stairs/reports/2026-10-22_12-57-06_201909_nominal/summary.json) | 20 | 无扰动；等级 0～1 | 14 | 90 |
| [台阶带扰动](pretrained/bpx_stairs/reports/2026-10-22_13-00-11_185805_robust/summary.json) | 20 | 随机化 + 每 8～15 秒推扰；等级 0～1 | 52 | 83 |
| [台阶较宽等级范围](pretrained/bpx_stairs/reports/2026-10-22_13-05-06_654630_nominal/summary.json) | 20 | 无扰动；等级 0～9 | 18 | 88 |

推扰幅度为 XY 每轴最多 ±0.6 m/s。播放冻结课程升级/降级；0～9 的评估是在该范围采样初始等级，并非全部使用等级 9。固定评估序列含超出台阶训练范围的速度。失败后会自动重置，超时是正常回合结束，因此这些计数不能视为台阶通过率；台阶模型在这些记录条件下仍有失败。各报告旁的 `phases.csv` 保存跟踪误差和分阶段统计。下文 MuJoCo 场景为平地，不能用于验证台阶通过能力。

## 3. 验证工程

不启动仿真器，先检查任务注册：

```bash
python scripts/list_envs.py
```

预期输出：

```text
bpx_flat
bpx_rough
bpx_stairs
```

随后启动一个无界面环境并验证训练接口：

```bash
./bpx.sh contract --headless
```

检查会验证 45 维 Actor 观测、48 维 Critic 观测、12 维动作以及 12 个策略关节名称，成功时输出：

```text
[PASS] Actor 45-D, Critic 48-D, Action 12-D runtime contract.
```

## 4. 工程结构

```text
mirrorme_rl_train/
├── bpx.sh                         # 统一命令入口
├── pretrained/                    # 已发布的平地/台阶模型、配置与评估报告
├── resources/
│   ├── mirrorme.py                # BPX Articulation 配置
│   └── BPX/                       # 随仓库提供的机器人 URDF 和网格资源
├── scripts/
│   ├── setup_environment.sh       # 创建干净 Conda 环境
│   ├── inspect_contract.py        # 运行时维度和关节名称检查
│   ├── train.py                   # PPO 训练
│   ├── play.py                    # 检查点评估
│   └── export_policy.py           # 导出 TorchScript/ONNX
└── source/mirrorme_rl_train/
    └── mirrorme_rl_train/tasks/locomotion/
        ├── bpx_flat_env_cfg.py
        ├── bpx_rough_env_cfg.py
        ├── bpx_stairs_env_cfg.py
        ├── agents/rsl_rl_ppo_cfg.py
        └── mdp/rewards.py
```

训练加载 `resources/BPX/bpx/urdf/bpx.urdf`。不要改变资源内部目录结构，否则 URDF 的相对网格路径会失效。本地实验可以通过 `MIRRORME_BPX_ASSET_ROOT` 覆盖资源根目录。

## 5. 训练接口

### Actor 观测：45 维

```text
0:3    机身角速度 × 0.25
3:6    投影重力
6:9    [vx, vy, wz] 指令 × [2.0, 2.0, 0.25]
9:21   关节位置 - 默认关节位置
21:33  关节速度 × 0.05
33:45  上一步原始动作
```

训练时会对角速度、重力、关节位置和关节速度加入观测噪声。

### Critic 观测：48 维

Critic 接收 45 维 policy 观测，再加上仅仿真可用的 3 维机身线速度。Actor 不使用观测归一化，Critic 开启观测归一化。

### 动作：12 维

```text
q_target = q_default + 0.25 × clipped_action
```

策略关节顺序：

```text
FL/FR/HL/HR 髋关节 roll，
FL/FR/HL/HR 髋关节 pitch，
FL/FR/HL/HR 膝关节
```

每条腿的默认关节位置分别为 hip roll `0.0`、hip pitch `0.8`、knee `-1.5` 弧度。仿真执行器参数为力矩上限 `30 Nm`、刚度 `30`、阻尼 `1`、armature `0.003`。

## 6. 环境与 PPO 配置

| 设置 | 数值 |
|---|---:|
| 仿真时间步 | `0.005 s` |
| 控制降采样 | `4` |
| 策略频率 | `50 Hz` |
| Episode 长度 | `20 s` |
| 默认环境数量 | `4096` |
| 指令重采样间隔 | `5–9 s` |
| 静止环境比例 | `20%` |
| 前后速度 | `[-1.5, 1.5] m/s` |
| 横向速度 | `[-1.0, 1.0] m/s` |
| 偏航角速度 | `[-2.0, 2.0] rad/s` |

Actor 网络为 `45 → 256 → 256 → 128 → 12`，Critic 网络为 `48 → 512 → 256 → 128 → 1`，两者均使用 ELU 激活函数。

默认 PPO 参数：

| 设置 | 数值 |
|---|---:|
| 每个环境的 rollout 长度 | `24` 步 |
| 每次更新的训练轮数 | `5` |
| Mini-batch 数量 | `4` |
| PPO clip 参数 | `0.2` |
| 折扣因子 / GAE lambda | `0.99 / 0.95` |
| 学习率 | `1e-3`，自适应 KL 调度 |
| 熵系数 | `0.01` |
| 保存间隔 | `200` 次迭代 |
| 默认最大迭代次数 | `10000` |

速度跟踪奖励中，XY 线速度权重为 `+2.0`，偏航角速度权重为 `+1.0`。其他奖励项包括机身稳定性、机身高度、足端腾空和抬脚高度、滑动、接触力、非足端碰撞、关节力矩、机械功率、加速度、动作变化率、关节限位、腿部姿态、静止姿态和终止惩罚。

域随机化包括摩擦和恢复系数、机身质量和质心、初始位姿和速度、关节状态、观测噪声以及水平推力扰动。

### 电机标定与站姿调整计划

后续优先校准关节输出侧的力矩—转速包络，并比较不同站姿下的持续力矩/电流与温升，避免长期深蹲带来的关节负担。当前固定 `30 Nm` 上限和 `0.48 m` 平地高度目标均为现有仿真设置，不代表实测最优值。悬吊可测响应与延迟，完整输出能力曲线需要加载测量或有明确条件的厂商数据。DR 先保留常规类别，范围围绕实测值确定。

测量、拟合、仿真接入和站高选择方法见 [BPX 电机标定、站姿与复杂地形训练调整](docs/bpx-actuator-calibration.zh-CN.md)。本次仅补充方案，未改动训练参数或已发布模型。

## 7. 训练

先运行小规模冒烟测试：

```bash
./bpx.sh train --task bpx_flat --num_envs 64 --max_iterations 20 --headless
```

然后开始较大规模训练：

```bash
./bpx.sh train \
  --task bpx_flat \
  --num_envs 1024 \
  --max_iterations 3000 \
  --run_name baseline \
  --headless
```

训练结果保存在：

```text
logs/bpx_flat/YYYY-MM-DD_HH-MM-SS[_run_name]/
├── model_*.pt
├── params/env.yaml
├── params/agent.yaml
└── events.out.tfevents.*
```

使用 TensorBoard 查看训练过程：

```bash
tensorboard --logdir logs/bpx_flat
```

从指定运行继续训练：

```bash
./bpx.sh train \
  --task bpx_flat \
  --resume \
  --load_run 2026-07-23_10-00-00_baseline \
  --checkpoint model_2000.pt \
  --num_envs 1024 \
  --max_iterations 1000 \
  --headless
```

也可以通过 `--resume --checkpoint /absolute/path/model_2000.pt` 直接指定绝对路径。

### 多地形训练

```bash
./bpx.sh train --task bpx_rough --num_envs 64 --max_iterations 20 --headless
```

默认混合平地 20%、随机起伏 30%（-2～2 cm）、缓坡 30% 和低台阶 20%（2～8 cm）。课程要求完整行走回合达到有效位移且速度跟踪良好才升级，行走失败或跟踪较差时降级；站立和纯转向回合保持等级。初始等级为 0～1；随机起伏幅度不随等级改变。速度范围为前后 ±0.7 m/s、横向 ±0.45 m/s、转向 ±0.8 rad/s。每个 20 秒回合采样前后（30%）、横移（25%）、转向（15%）、组合（15%）或站立（15%），运动方向正负随机均衡。多地形任务关闭基于世界坐标高度的机身高度与抬脚奖励，减弱机身水平约束。

训练日志默认保存在 `logs/bpx_rough/`；可以使用 `--resume --checkpoint /absolute/path/flat_model.pt` 从平地模型继续训练。`play`、`export` 和 `contract` 均支持 `--task bpx_rough`，播放时冻结地形等级升级/降级，导出名称为 `bpx_rough_45.pt` / `bpx_rough_45.onnx`。使用 20 个播放环境可以覆盖所有地形列；`--eval` 仍使用现有固定速度序列，其中部分速度高于多地形训练范围。

完整训练、播放和导出命令见 [仿真指令.md](仿真指令.md#6-多地形训练与评估)。MuJoCo 当前场景仍是平地。

## 8. 评估与导出

打开交互式评估窗口：

```bash
./bpx.sh play --checkpoint /absolute/path/model_2999.pt
```

评估脚本还支持 `--num_envs`、`--video`、`--video_length`、`--real_time`，以及 `--headless`、`--device` 等 Isaac Lab 启动参数。

### 带扰动的定量评估

普通播放会关闭观测噪声、摩擦/质量/质心随机化和推扰。添加 `--robust` 保留这些训练扰动；添加 `--eval` 运行固定指令序列并保存统计：

```bash
./bpx.sh play \
  --checkpoint pretrained/bpx_flat/model_2999.pt \
  --eval --robust --num_envs 16 --eval_cycles 1 --real_time
```

每轮依次测试站立、前进、停车、后退、左移、右移、左转、右转、组合运动和停车。具体速度在 `scripts/play_evaluation.py` 的 `EVALUATION_PHASES` 中修改。默认每阶段 10 秒、两轮，共 200 秒仿真时间；上面的单轮命令运行 100 秒。阶段切换时不重置机器人，以测试启停和转向过渡；发生物理终止或达到 20 秒 episode 上限时，环境自动重置并继续当前指令。

提高推扰频率以观察恢复能力：

```bash
./bpx.sh play \
  --checkpoint pretrained/bpx_flat/model_2999.pt \
  --eval --robust --num_envs 16 --eval_cycles 1 \
  --push_interval_s 3 5 --push_velocity 0.6 --real_time
```

这里的推扰是直接改变机身 XY 速度，每轴变化范围 `[-0.6, 0.6] m/s`，不是以牛顿为单位的外力。摩擦、质量、质心在环境启动时随机化；需要多个 `--seed` 和足够的环境数量来覆盖不同组合。

终端按阶段输出速度 RMSE、物理终止/超时次数、峰值倾角和估计力矩饱和比例。完整结果保存在 checkpoint 旁的 `evaluations/<时间>_robust/summary.json` 和 `phases.csv`，包括接触足端滑动速度、静止阶段实际移动速度、动作变化和力矩 RMS。统计在自动重置之前采样，失败次数包含整个阶段；跟踪均值默认排除阶段切换和每次重置后的前 2 秒。ImplicitActuator 的力矩是估计值，峰值和饱和统计只在 50 Hz 控制边界采样。

使用相同 checkpoint、`--seed`、环境数和阶段参数，去掉 `--robust` 运行标称条件评估，比较两次 CSV。支持 `--eval_phase_s`、`--eval_cycles`、`--eval_warmup_s` 和 `--eval_output_dir`；添加 `--headless` 可批量评估。`--eval --video` 的 `--video_length` 只限制录像长度，评估仍运行完整序列。Ctrl+C 会保存已完成阶段和当前阶段的部分结果，并将 `completed_schedule` 标为 `false`。

如果启动时提示找不到 `libtorch_cuda_linalg.so`，在激活训练环境后设置动态库路径，再运行命令：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

不进入交互式播放循环，直接导出模型：

```bash
./bpx.sh export --checkpoint /absolute/path/model_2999.pt
```

默认会在检查点旁边创建 `exported/bpx_flat_45.pt` 和 `exported/bpx_flat_45.onnx`。可以通过 `--output_dir PATH` 指定其他输出目录。

### MuJoCo 跨仿真测试（可选）

在现有 BPX Python 环境中安装可选依赖，然后启动：

```bash
python -m pip install -r requirements-mujoco.txt
./bpx.sh mujoco \
  --policy pretrained/bpx_flat/exported/bpx_flat_45.pt \
  --command 0.7 0 0 --duration 60
```

这里加载的是导出的 **TorchScript actor**，不能直接传 `model_2999.pt` 训练检查点。MuJoCo 物理和策略推理均在 CPU 上执行，不启动 Isaac；窗口模式需要桌面显示。`--command VX VY WZ` 分别设置机身坐标系前向速度、横向速度（m/s）和转向角速度（rad/s）。窗口内 W/S、A/D、Q/E 分别以 0.1 增减这三个指令，空格停车，R 重置，P 推扰。`--eval` 模式禁用键盘改速度，手动重置/推扰仍会记录。

运行十阶段指令序列并施加世界坐标系 XY 速度推扰：

```bash
./bpx.sh mujoco \
  --policy pretrained/bpx_flat/exported/bpx_flat_45.pt \
  --eval --eval_cycles 1 --push_interval_s 3 5 --push_velocity 0.6
```

`--eval` 使用 `EVALUATION_PHASES` 的速度，覆盖 `--command`。添加 `--headless` 可快速定量测试；窗口模式添加 `--no-real-time` 可取消实时限速。去掉 `--push_interval_s` 即可作无推扰对照；`--friction` 设置固定滑动摩擦系数。报告默认位于策略文件旁的 `mujoco_evaluations/<时间>/summary.json` 和 `phases.csv`，也可用 `--output_dir` 指定。速度 RMSE 默认排除阶段切换和重置后的前 2 秒（`--warmup_s`）；失败、推扰和峰值统计覆盖所有步骤。20 秒超时单独计数，不算倾倒。关窗口或 Ctrl+C 会保存部分结果，并标记 `completed_schedule: false`。

场景从 BPX URDF 生成，保留质量、惯量、碰撞体和 OBJ 外观。关节名称映射、45 维观测、动作缩放 0.25、50 Hz 策略、200 Hz 物理、PD 30/1 和 30 Nm 力矩限制与训练接口对齐；速度先从世界系转换到 torso 连杆坐标系，避免误用 MuJoCo 的惯性主轴坐标系。PD 力矩每个物理步显式计算，积分和接触求解仍与 Isaac 的隐式驱动不同。这里的扰动仅包含速度增量，不含观测噪声、质量/质心/电机随机化，测试通过不等于可以直接上实机。

安装 MuJoCo 后可运行 CPU 接口和统计测试：`python -m unittest discover -s tests -v`。

## 9. 常见问题

| 现象 | 检查方法 |
|---|---|
| 找不到 BPX 网格或 URDF | 检查 `resources/BPX/bpx/` 是否完整，以及 `MIRRORME_BPX_ASSET_ROOT` 是否指向正确资源 |
| 环境列表中没有 `bpx_flat` | 激活正确环境后执行 `./bpx.sh install` |
| 缺少 Isaac Lab 或 Isaac Sim 模块 | 确认已经激活 `bpx_train`，然后重新运行版本检查 |
| CUDA 不可用 | 检查 `nvidia-smi`、NVIDIA 驱动、GPU 可见性和 CUDA 12.8 PyTorch wheel |
| `flatdict` 构建失败 | 使用文档中的 `setuptools<81` 和 `--no-build-isolation` 命令 |
| Isaac Sim 第一次启动时暂停 | 完成许可协议提示，并等待扩展缓存下载完成 |
| 长时间训练一启动就失败 | 先运行 `./bpx.sh contract --headless` 和 64 环境冒烟测试 |

## 参考资料与许可证

- [Isaac Lab 2.3.0 安装文档](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html)
- [Isaac Lab pip 包安装文档](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/isaaclab_pip_installation.html)
- [BPX 模型资源](https://github.com/mirrormerobotics/BPX)

本项目采用 [BSD 3-Clause License](LICENSE)。NVIDIA Isaac Sim 及其 pip 包适用 NVIDIA 自身的许可条款。
