# MIRRORME RL Train

[English](README.md) | [简体中文](README.zh-CN.md)

An open reinforcement-learning training baseline for the MIRRORME BPX quadruped. It follows the Isaac Lab external-project layout, uses a Manager-Based Environment, and trains PPO policies through the official RSL-RL integration.

This repository covers simulation training, evaluation, and policy export only. Robot SDK integration and physical-robot deployment are intentionally out of scope.

Optional privileged-teacher / blind-student tasks are also available: train `bpx_rough_teacher` or `bpx_stairs_teacher` with simulated terrain, then distill `bpx_rough_blind` or `bpx_stairs_blind` using 20 proprioceptive frames. See the [training and deployment-interface guide (Chinese)](docs/bpx-privileged-blind-training.zh-CN.md). The original three tasks retain the 45-D interface described below.

## Task and naming

The project registers flat-ground, mixed-terrain and dedicated stair tasks:

| Name | Value | Meaning |
|---|---|---|
| Task ID | `bpx_flat` | Flat-ground omnidirectional velocity tracking |
| Task ID | `bpx_rough` | Mixed flat ground, bumps, mild slopes and low stairs |
| Task ID | `bpx_stairs` | Dedicated pyramid-stair locomotion |
| Experiment name | `bpx_flat` | RSL-RL log directory under `logs/` |
| Experiment name | `bpx_rough` | Default mixed-terrain log directory |
| Experiment name | `bpx_stairs` | Dedicated stair log directory |
| Run directory | `YYYY-MM-DD_HH-MM-SS[_run_name]` | One training run |
| Checkpoint | `model_<iteration>.pt` | Saved RSL-RL checkpoint, for example `model_2999.pt` |
| Exported policy | `bpx_flat_45.pt` / `bpx_flat_45.onnx` | TorchScript and ONNX actor exports |
| Exported policy | `bpx_rough_45.pt` / `bpx_rough_45.onnx` | Mixed-terrain actor exports |
| Exported policy | `bpx_stairs_45.pt` / `bpx_stairs_45.onnx` | Dedicated stair actor exports |

The command target is body-frame velocity `[vx, vy, wz]` on flat ground, mild rough terrain or stairs. All three tasks share the 45-D Actor, 48-D Critic and 12-D action interface. There are no wheel joints, height scanners, history encoders, DWAQ/CENet modules, recurrent networks, or custom RL runners.

## 1. Requirements

The published setup is pinned to the following stack:

| Component | Version |
|---|---:|
| Operating system | Ubuntu 22.04 LTS, x86-64 |
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.0 |
| PyTorch / torchvision | 2.7.0 / 0.22.0, CUDA 12.8 wheels |
| RSL-RL | 3.0.1, installed by the Isaac Lab `rsl-rl` extra |

Isaac Sim training requires a compatible NVIDIA GPU and driver. NVIDIA recommends at least 32 GB of RAM and 16 GB of VRAM for the Isaac Lab 2.3.0 stack. Pip installation also requires glibc 2.35 or newer; Ubuntu 22.04 satisfies this requirement.

## 2. Clone and install

Clone the repository. The BPX URDF and mesh assets are included directly:

```bash
git clone https://github.com/Ryon-droid/bpx-rl-train.git mirrorme_rl_train
cd mirrorme_rl_train
```

Runtime logs and caches are excluded by `.gitignore`. Selected trained checkpoints, exported actors and their training configurations are committed under [`pretrained/`](pretrained/). A normal clone includes these files; Git LFS and separate model downloads are not required. For an existing checkout, run `git pull --ff-only` to get them.

### Automatic clean-environment setup

From the repository root, run:

```bash
./bpx.sh setup
```

This creates a new Conda environment named `bpx_train`, installs the pinned Isaac Sim, Isaac Lab, PyTorch, RSL-RL, and project packages, and then launches the headless runtime contract check. It refuses to modify an existing environment.

Use a different new environment name when needed:

```bash
./bpx.sh setup my_bpx_environment
```

Use `--skip-contract` only when installing dependencies without launching Isaac Sim:

```bash
./bpx.sh setup my_bpx_environment --skip-contract
```

After setup finishes:

```bash
conda activate bpx_train
```

The first Isaac Sim launch may download extension caches and ask you to accept the NVIDIA Omniverse license agreement. Read and answer that prompt directly; the setup script does not accept the agreement on your behalf.

### Manual installation

Use these commands only when troubleshooting or when you need control over each installation step:

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

Isaac Lab 2.3.0 pins the legacy `flatdict==4.0.1` source package. Preinstalling it with a compatible build toolchain avoids failures caused by newer setuptools releases.

Verify the installed versions and GPU visibility:

```bash
python -c "from importlib.metadata import version; import torch; print('Isaac Lab:', version('isaaclab')); print('Isaac Sim:', version('isaacsim')); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available` must be `True` before GPU training.

## Pretrained models: play and resume

Two independent trained policies are included. Both used seed `2026`, `1024` parallel environments and `3000` training iterations; the final saved iteration is `2999`. Each bundle is approximately 4.6 MB.

| Task | Original run | Checkpoint | Training configuration |
|---|---|---|---|
| Flat walking | `2026-10-02_16-11-56_baseline_v1` | [bpx_flat/model_2999.pt](pretrained/bpx_flat/model_2999.pt) | [Environment](pretrained/bpx_flat/params/env.yaml), [PPO](pretrained/bpx_flat/params/agent.yaml) |
| Dedicated stairs | `2026-10-22_12-03-41_stairs_from_scratch` | [bpx_stairs/model_2999.pt](pretrained/bpx_stairs/model_2999.pt) | [Environment](pretrained/bpx_stairs/params/env.yaml), [PPO](pretrained/bpx_stairs/params/agent.yaml) |

Each bundle contains the full RSL-RL checkpoint (model and optimizer state), `exported/` TorchScript and ONNX actors, `params/` configuration snapshots, and `reports/` historical evaluation JSON/CSV files. [manifest.json](pretrained/manifest.json) records original paths, byte sizes and SHA-256 hashes. Use `model_2999.pt` for Isaac playback and continued training; use the exported actor for inference or MuJoCo.

The stair policy was trained from random initialization on the independent `bpx_stairs` task. Its terrain is 100% pyramid stairs with a configured step-height range of 2–10 cm, 30 cm treads and a 3 m central platform. Training command bounds are forward ±0.5 m/s, lateral ±0.2 m/s and yaw ±0.5 rad/s. The actor has no terrain height scan. The task implementation is [bpx_stairs_env_cfg.py](source/mirrorme_rl_train/mirrorme_rl_train/tasks/locomotion/bpx_stairs_env_cfg.py).

After installing and activating `bpx_train`, run these commands from the repository root:

```bash
# Flat-ground playback
./bpx.sh play --task bpx_flat \
  --checkpoint pretrained/bpx_flat/model_2999.pt \
  --device cpu --num_envs 1 --seed 2026 --real_time

# Dedicated stair playback
./bpx.sh play --task bpx_stairs \
  --checkpoint pretrained/bpx_stairs/model_2999.pt \
  --device cpu --num_envs 20 --seed 2026 --real_time
```

On a server without a display, replace `--real_time` with `--headless`; use `--device cuda:0` for GPU simulation/inference. Playback also exports the actor beside the checkpoint and may rewrite the bundled exports.

Continue training either policy on a compatible GPU:

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

Here `--max_iterations 1000` requests 1000 additional training iterations. New checkpoints and configuration snapshots are written to a new run under `logs/<task>/`. Resume restores saved model/optimizer state and the iteration counter; it does not restore simulator state, terrain curriculum or random-number-generator state.

The bundled YAML files are original archival snapshots, including historical absolute paths and Python-specific YAML tags. Train/play load the registered Python task/PPO configuration and command-line overrides; specifying a checkpoint does **not** automatically load its saved YAML. Compare these snapshots when changing settings or reproducing a run.

### Included historical evaluations

These reports came from the original runs; this publication did not rerun simulation. All used seed `2026` and one 100-second, ten-phase command cycle, with two seconds of tracking warmup after phase changes/resets.

| Report | Environments | Conditions / initial terrain levels | Physical failures | Episode timeouts |
|---|---:|---|---:|---:|
| [Flat robust](pretrained/bpx_flat/reports/2026-10-07_14-51-37_721833_robust/summary.json) | 16 | Randomization + XY velocity kicks every 3–5 s | 0 | 80 |
| [Stairs nominal](pretrained/bpx_stairs/reports/2026-10-22_12-57-06_201909_nominal/summary.json) | 20 | No disturbances; levels 0–1 | 14 | 90 |
| [Stairs robust](pretrained/bpx_stairs/reports/2026-10-22_13-00-11_185805_robust/summary.json) | 20 | Randomization + kicks every 8–15 s; levels 0–1 | 52 | 83 |
| [Stairs nominal, wider levels](pretrained/bpx_stairs/reports/2026-10-22_13-05-06_654630_nominal/summary.json) | 20 | No disturbances; levels 0–9 | 18 | 88 |

Velocity kicks range up to ±0.6 m/s per XY axis. Playback freezes curriculum progression; the 0–9 evaluation samples initial levels across that range, rather than testing only level 9. The fixed command suite includes speeds outside the stair training bounds. Failures trigger resets and timeouts are normal episode endings, so these counts are not stair-traversal success rates. The stair checkpoint still has failures under the recorded conditions. See each report's adjacent `phases.csv` for tracking errors and per-phase statistics. The MuJoCo scene below is flat and does not evaluate stair traversal.

## 3. Verify the project

Check task registration without starting the simulator:

```bash
python scripts/list_envs.py
```

Expected output:

```text
bpx_flat
bpx_rough
bpx_stairs
```

Then launch one headless environment and validate the training interface:

```bash
./bpx.sh contract --headless
```

A successful check verifies 45 Actor observations, 48 Critic observations, 12 actions, and all 12 policy joint names, then prints:

```text
[PASS] Actor 45-D, Critic 48-D, Action 12-D runtime contract.
```

## 4. Repository layout

```text
mirrorme_rl_train/
├── bpx.sh                         # command entry point
├── pretrained/                    # published flat/stair models, configs and reports
├── resources/
│   ├── mirrorme.py                # BPX articulation configuration
│   └── BPX/                       # bundled robot URDF and mesh assets
├── scripts/
│   ├── setup_environment.sh       # clean Conda setup
│   ├── inspect_contract.py        # runtime dimension/name check
│   ├── train.py                   # PPO training
│   ├── play.py                    # checkpoint evaluation
│   └── export_policy.py           # TorchScript/ONNX export
└── source/mirrorme_rl_train/
    └── mirrorme_rl_train/tasks/locomotion/
        ├── bpx_flat_env_cfg.py
        ├── bpx_rough_env_cfg.py
        ├── bpx_stairs_env_cfg.py
        ├── agents/rsl_rl_ppo_cfg.py
        └── mdp/rewards.py
```

Training loads `resources/BPX/bpx/urdf/bpx.urdf`. Keep the asset directory structure intact so its relative mesh paths remain valid. `MIRRORME_BPX_ASSET_ROOT` can override the asset root for local experiments.

## 5. Training interface

### Actor observations: 45 dimensions

```text
0:3    base angular velocity × 0.25
3:6    projected gravity
6:9    [vx, vy, wz] command × [2.0, 2.0, 0.25]
9:21   joint position - default joint position
21:33  joint velocity × 0.05
33:45  previous raw action
```

Observation noise is enabled during training for angular velocity, gravity, joint position, and joint velocity.

### Critic observations: 48 dimensions

The Critic receives the 45-dimensional policy group plus the simulator-only 3D base linear velocity. Actor normalization is disabled; Critic observation normalization is enabled.

### Actions: 12 dimensions

```text
q_target = q_default + 0.25 × clipped_action
```

Policy joint order:

```text
FL/FR/HL/HR hip roll,
FL/FR/HL/HR hip pitch,
FL/FR/HL/HR knee
```

The default joint pose is hip roll `0.0`, hip pitch `0.8`, and knee `-1.5` radians for each leg. The simulated actuator uses effort limit `30 Nm`, stiffness `30`, damping `1`, and armature `0.003`.

## 6. Environment and PPO configuration

| Setting | Value |
|---|---:|
| Simulation timestep | `0.005 s` |
| Control decimation | `4` |
| Policy frequency | `50 Hz` |
| Episode length | `20 s` |
| Default environments | `4096` |
| Command resampling | `5–9 s` |
| Standing environments | `20%` |
| Forward velocity | `[-1.5, 1.5] m/s` |
| Lateral velocity | `[-1.0, 1.0] m/s` |
| Yaw rate | `[-2.0, 2.0] rad/s` |

The Actor is `45 → 256 → 256 → 128 → 12`; the Critic is `48 → 512 → 256 → 128 → 1`. Both use ELU activations.

Default PPO settings:

| Setting | Value |
|---|---:|
| Rollout length | `24` steps per environment |
| Learning epochs | `5` |
| Mini-batches | `4` |
| Clip parameter | `0.2` |
| Discount / GAE lambda | `0.99 / 0.95` |
| Learning rate | `1e-3`, adaptive KL schedule |
| Entropy coefficient | `0.01` |
| Save interval | `200` iterations |
| Default maximum iterations | `10000` |

Velocity tracking uses weights `+2.0` for XY velocity and `+1.0` for yaw rate. Additional terms cover body stability, base height, foot air time and clearance, sliding, contact force, non-foot contacts, torque, mechanical power, acceleration, action rate, joint limits, leg posture, standing posture, and termination.

Domain randomization covers friction, restitution, base mass and center of mass, initial pose and velocity, joint state, observation noise, and horizontal pushes.

### Actuator calibration and standing posture plan

The next calibration priorities are a measured joint-output torque–speed envelope and a standing-height comparison using sustained joint torque/current and temperature rise. The current fixed `30 Nm` limit and `0.48 m` flat-ground height target are simulation settings, not measured optima. Suspended tests characterize response and delay; a full capability envelope requires controlled loading or manufacturer data with stated operating conditions. Keep conventional domain randomization categories and choose ranges around measured behavior.

See the [BPX actuator calibration and standing-posture plan (Chinese)](docs/bpx-actuator-calibration.zh-CN.md) for measurement, fitting, simulator integration and posture selection. This documentation update does not change training parameters or published models.

## 7. Train

Run a small smoke test first:

```bash
./bpx.sh train --task bpx_flat --num_envs 64 --max_iterations 20 --headless
```

Then start a larger run:

```bash
./bpx.sh train \
  --task bpx_flat \
  --num_envs 1024 \
  --max_iterations 3000 \
  --run_name baseline \
  --headless
```

Training outputs are written to:

```text
logs/bpx_flat/YYYY-MM-DD_HH-MM-SS[_run_name]/
├── model_*.pt
├── params/env.yaml
├── params/agent.yaml
└── events.out.tfevents.*
```

Monitor a run with TensorBoard:

```bash
tensorboard --logdir logs/bpx_flat
```

Resume from a named run:

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

An absolute checkpoint path is also accepted with `--resume --checkpoint /absolute/path/model_2000.pt`.

### Mixed-terrain training

```bash
./bpx.sh train --task bpx_rough --num_envs 64 --max_iterations 20 --headless
```

The map combines flat ground (20%), uniform height noise (30%, -2 to 2 cm), mild slopes (30%) and stairs (20%, 2 to 8 cm). Terrain promotion requires a completed walking episode with sufficient displacement and good velocity tracking; failed or poorly tracked walking episodes are demoted. Standing and pure-yaw episodes hold their levels. Noise amplitude remains fixed, and initial levels are 0 to 1. Command ranges are ±0.7 m/s forward, ±0.45 m/s lateral and ±0.8 rad/s yaw. Each 20-second episode samples longitudinal (30%), lateral (25%), yaw (15%), combined (15%) or standing (15%) commands, with balanced random signs. World-Z base-height and swing-clearance rewards are disabled, and the horizontal-body penalty is reduced.

Logs default to `logs/bpx_rough/`. The unchanged observation and network dimensions allow fine-tuning flat checkpoints with `--resume --checkpoint /absolute/path/flat_model.pt`. `play`, `export` and `contract` accept `--task bpx_rough`; exported filenames are `bpx_rough_45.pt` and `bpx_rough_45.onnx`. Playback freezes curriculum progression at the initial levels; 20 environments cover all terrain columns. The existing `--eval` command suite includes speeds above the rough-task training range. The MuJoCo scene remains flat.

See [仿真指令.md](仿真指令.md#6-多地形训练与评估) for full commands.

## 8. Evaluate and export

Open an interactive evaluation window:

```bash
./bpx.sh play --checkpoint /absolute/path/model_2999.pt
```

Optional evaluation flags include `--num_envs`, `--video`, `--video_length`, `--real_time`, and normal Isaac Lab launcher flags such as `--headless` and `--device`.

### Quantitative evaluation with disturbances

Normal playback disables observation noise, friction/mass/COM randomization and pushes. Use `--robust` to keep these training disturbances, and `--eval` to run a finite command suite with quantitative reports:

```bash
./bpx.sh play \
  --checkpoint pretrained/bpx_flat/model_2999.pt \
  --eval --robust --num_envs 16 --eval_cycles 1 --real_time
```

Each cycle tests standing, forward, stop, backward, left, right, yaw left, yaw right, combined motion and a final stop. Edit `EVALUATION_PHASES` in `scripts/play_evaluation.py` to set the commands. Defaults are ten seconds per phase and two cycles (200 simulation seconds); the single-cycle example takes 100 seconds. Phase transitions preserve robot state to test starts, stops and turns. Physical terminations and 20-second episode timeouts automatically reset the robot and continue the current command.

For more frequent disturbances, add `--push_interval_s 3 5 --push_velocity 0.6`. Pushes change base XY velocity directly, with each component in `[-0.6, 0.6] m/s`; they are not forces in newtons. Friction, mass and COM are randomized at startup, so use multiple seeds and enough environments to sample varied conditions.

The console prints tracking RMSE, physical termination/timeout counts, peak tilt and estimated torque saturation per phase. `evaluations/<timestamp>_robust/summary.json` and `phases.csv` beside the checkpoint also report contact-foot slide speed, actual speed during standing, action changes and torque RMS. Measurements are captured before automatic resets. Tracking means exclude two seconds after phase changes and resets by default; failures, peaks and saturation include the entire phase. ImplicitActuator torque is estimated and sampled at the 50 Hz control boundary.

Compare against a nominal run by removing `--robust` and using the same checkpoint, seed, environment count and phase parameters. Options include `--eval_phase_s`, `--eval_cycles`, `--eval_warmup_s` and `--eval_output_dir`; use `--headless` for batch evaluation. With `--eval --video`, `--video_length` only limits recording, and the full evaluation still runs. Ctrl+C saves completed and partial phases with `completed_schedule: false`.

If startup cannot find `libtorch_cuda_linalg.so`, set the library path after activating the training environment:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

Export without entering the interactive play loop:

```bash
./bpx.sh export --checkpoint /absolute/path/model_2999.pt
```

By default, the exporter creates `exported/bpx_flat_45.pt` and `exported/bpx_flat_45.onnx` beside the checkpoint. Use `--output_dir PATH` to select another directory.

### MuJoCo sim-to-sim test (optional)

Install the optional simulator into the existing BPX Python environment:

```bash
python -m pip install -r requirements-mujoco.txt
./bpx.sh mujoco \
  --policy pretrained/bpx_flat/exported/bpx_flat_45.pt \
  --command 0.7 0 0 --duration 60
```

Use the exported **TorchScript actor**, not an RSL-RL `model_*.pt` checkpoint. MuJoCo physics and actor inference run on CPU without launching Isaac. A desktop display is required for the viewer. `--command VX VY WZ` sets body-frame forward/lateral velocity (m/s) and yaw rate (rad/s). W/S, A/D, Q/E adjust these commands by 0.1; Space stops, R resets, P applies a velocity kick. Keyboard command changes are disabled under `--eval`; reset/push events remain recorded.

Run the shared ten-phase schedule with world XY velocity kicks:

```bash
./bpx.sh mujoco \
  --policy pretrained/bpx_flat/exported/bpx_flat_45.pt \
  --eval --eval_cycles 1 --push_interval_s 3 5 --push_velocity 0.6
```

`--eval` uses `EVALUATION_PHASES` rather than `--command`. Add `--headless` for fast quantitative runs, or `--no-real-time` for unrestricted GUI playback. Remove `--push_interval_s` for a nominal comparison. `--friction` changes the fixed sliding friction coefficient. Reports are saved under the actor's directory in `mujoco_evaluations/<timestamp>/summary.json` and `phases.csv`; `--output_dir` overrides this location. Tracking RMSE excludes `--warmup_s` (default 2 seconds) after phase changes and episode resets; failures, pushes and peaks include all steps. A 20-second timeout is counted separately from physical failures. Closing the window or Ctrl+C saves partial results with `completed_schedule: false`.

The scene is built from the BPX URDF with its masses, inertias, collisions and OBJ visuals. Named joint mapping, the 45-D observation, action scale 0.25, 50 Hz actor rate, 200 Hz physics, PD gains 30/1 and 30 Nm torque limit match the training interface. Body velocities are transformed from world axes into the torso link frame, avoiding MuJoCo's principal-inertia frame. PD torque is explicitly recomputed at each physics step; contact solving and integration differ from Isaac's implicit drives. This test includes velocity kicks only, with no observation noise or mass/COM/motor randomization, and does not establish readiness for physical deployment.

Run CPU interface/measurement tests with `python -m unittest discover -s tests -v` after installing MuJoCo.

## 9. Troubleshooting

| Symptom | Check |
|---|---|
| BPX meshes or URDF are missing | Check that `resources/BPX/bpx/` is complete and that `MIRRORME_BPX_ASSET_ROOT` points to the intended assets |
| `bpx_flat` is not listed | Activate the intended environment and run `./bpx.sh install` |
| `ModuleNotFoundError` for Isaac Lab or Isaac Sim | Confirm the `bpx_train` environment is active and repeat the version check |
| CUDA is unavailable | Check `nvidia-smi`, the NVIDIA driver, GPU visibility, and the CUDA 12.8 PyTorch wheel |
| `flatdict` fails to build | Use the documented `setuptools<81` and `--no-build-isolation` commands |
| Isaac Sim pauses on first launch | Complete the license prompt and allow extension caches to download |
| A long run fails immediately | Run `./bpx.sh contract --headless` and the 64-environment smoke test first |

## References and license

- [Isaac Lab 2.3.0 installation](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html)
- [Isaac Lab pip-package installation](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/isaaclab_pip_installation.html)
- [BPX model assets](https://github.com/mirrormerobotics/BPX)

The project is released under the [BSD 3-Clause License](LICENSE). NVIDIA Isaac Sim and its pip packages are governed by NVIDIA's own license terms.
