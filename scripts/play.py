#!/usr/bin/env python3
"""Play or quantitatively evaluate a BPX checkpoint, optionally with disturbances."""

import argparse
import math
import os
import sys
import time
from datetime import datetime

from project_paths import add_project_source

add_project_source()

from isaaclab.app import AppLauncher

import cli_args


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="bpx_flat", help="Registered Isaac Lab task ID.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--real_time", action="store_true", default=False)
parser.add_argument("--robust", action="store_true", help="Keep training observation noise, domain randomization and pushes.")
parser.add_argument("--eval", action="store_true", help="Run a finite fixed-command suite and save quantitative reports.")
parser.add_argument("--eval_phase_s", type=float, default=10.0, help="Simulation seconds per command phase.")
parser.add_argument("--eval_cycles", type=int, default=2, help="Repeat the ten-phase command suite this many times.")
parser.add_argument("--eval_warmup_s", type=float, default=2.0, help="Exclude settling after phase changes and resets from tracking means.")
parser.add_argument("--eval_output_dir", type=str, default=None, help="Report directory; default is a new checkpoint-local evaluations directory.")
parser.add_argument("--push_interval_s", type=float, nargs=2, default=None, metavar=("MIN", "MAX"), help="Override push interval in robust mode.")
parser.add_argument("--push_velocity", type=float, default=None, help="Override maximum absolute XY velocity kick (m/s) in robust mode.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.num_envs is not None and args_cli.num_envs < 1:
    parser.error("--num_envs must be positive")
if args_cli.eval and (
    not math.isfinite(args_cli.eval_phase_s) or args_cli.eval_phase_s <= 0
    or args_cli.eval_cycles < 1 or not math.isfinite(args_cli.eval_warmup_s)
    or not 0 <= args_cli.eval_warmup_s < args_cli.eval_phase_s
):
    parser.error("evaluation requires positive phase duration/cycles and 0 <= warmup < phase duration")
if args_cli.push_interval_s is not None and (
    not all(math.isfinite(x) for x in args_cli.push_interval_s)
    or not 0 < args_cli.push_interval_s[0] <= args_cli.push_interval_s[1]
):
    parser.error("--push_interval_s requires 0 < MIN <= MAX")
if args_cli.push_velocity is not None and (not math.isfinite(args_cli.push_velocity) or args_cli.push_velocity < 0):
    parser.error("--push_velocity must be finite and nonnegative")
if not args_cli.robust and (args_cli.push_interval_s is not None or args_cli.push_velocity is not None):
    parser.error("push overrides require --robust")
if args_cli.video:
    args_cli.enable_cameras = True
default_hydra_args = ["hydra.run.dir=logs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}"]
sys.argv = [sys.argv[0]] + default_hydra_args + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import mirrorme_rl_train.tasks  # noqa: F401
from mirrorme_rl_train.tasks.locomotion.bpx_flat_env_cfg import FOOT_BODY_NAMES, JOINT_NAMES
from play_evaluation import EVALUATION_PHASES, PlayEvaluator, record_evaluation_metrics
from runner_utils import export_runner_policy, load_runner_checkpoint, make_runner


class EvaluationVelocityCommand(UniformVelocityCommand):
    """Keep the scheduled command, including across automatic episode resets."""

    def __init__(self, cfg, env):
        self.evaluation_command = (0.0, 0.0, 0.0)
        super().__init__(cfg, env)

    def _resample_command(self, env_ids):
        self.vel_command_b[env_ids] = self.vel_command_b.new_tensor(self.evaluation_command)
        self.is_standing_env[env_ids] = not any(self.evaluation_command)
        self.is_heading_env[env_ids] = False

    def set_command(self, command):
        self.evaluation_command = command
        self._resample_command(slice(None))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs or 16
    # Keep reset locations at their initial difficulty during playback/evaluation.
    # Leave generator.curriculum enabled so the map retains graded rows.
    if getattr(env_cfg.curriculum, "terrain_levels", None) is not None:
        env_cfg.curriculum.terrain_levels = None
    if not args_cli.robust:
        env_cfg.observations.policy.enable_corruption = False
        if hasattr(env_cfg.observations, "proprio_history"):
            env_cfg.observations.proprio_history.enable_corruption = False
        env_cfg.events.physics_material = None
        env_cfg.events.add_base_mass = None
        env_cfg.events.base_com = None
        env_cfg.events.push_robot = None
    else:
        if args_cli.push_interval_s is not None:
            env_cfg.events.push_robot.interval_range_s = tuple(args_cli.push_interval_s)
        if args_cli.push_velocity is not None:
            speed = args_cli.push_velocity
            env_cfg.events.push_robot.params["velocity_range"] = {"x": (-speed, speed), "y": (-speed, speed)}
    if args_cli.eval:
        env_cfg.commands.base_velocity.class_type = EvaluationVelocityCommand
        env_cfg.commands.base_velocity.heading_command = False
        # Append after the physical termination terms to capture their signals before reset.
        env_cfg.terminations.evaluation_metrics = TerminationTermCfg(func=record_evaluation_metrics)
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        checkpoint = retrieve_file_path(args_cli.checkpoint)
    else:
        checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(checkpoint)
    env_cfg.log_dir = log_dir
    output_dir = None
    if args_cli.eval:
        output_dir = args_cli.eval_output_dir or os.path.join(
            log_dir, "evaluations", datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            + ("_robust" if args_cli.robust else "_nominal")
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(output_dir, "videos") if output_dir else os.path.join(log_dir, "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = make_runner(env, agent_cfg)
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    load_runner_checkpoint(runner, checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    export_dir = os.path.join(log_dir, "exported")
    export_runner_policy(runner, args_cli.task, export_dir)
    print(f"[INFO] Exported to: {export_dir}")

    dt = env.unwrapped.step_dt
    evaluator = None
    if args_cli.eval:
        phase_steps = math.ceil(args_cli.eval_phase_s / dt)
        if phase_steps <= math.ceil(args_cli.eval_warmup_s / dt):
            raise ValueError("phase must contain at least one control step after warmup")
        evaluator = PlayEvaluator(env.unwrapped, list(JOINT_NAMES), list(FOOT_BODY_NAMES), args_cli.eval_warmup_s)
        env.unwrapped.bpx_evaluator = evaluator
        schedule = [
            (cycle, name, command)
            for cycle in range(1, args_cli.eval_cycles + 1) for name, command in EVALUATION_PHASES
        ]
        metadata = {
            "task": args_cli.task,
            "terrain_type": env_cfg.scene.terrain.terrain_type,
            "max_init_terrain_level": env_cfg.scene.terrain.max_init_terrain_level,
            "checkpoint": os.path.abspath(checkpoint), "seed": agent_cfg.seed,
            "num_envs": env.unwrapped.num_envs, "robust": args_cli.robust,
            "control_dt_s": dt, "phase_duration_s": phase_steps * dt,
            "cycles": args_cli.eval_cycles, "warmup_s": args_cli.eval_warmup_s,
            "observation_noise": env_cfg.observations.policy.enable_corruption,
            "events": {
                name: {
                    "mode": term.mode, "interval_range_s": term.interval_range_s,
                    "params": {key: value for key, value in term.params.items() if key != "asset_cfg"},
                } if term is not None else None
                for name in ("physics_material", "add_base_mass", "base_com", "push_robot")
                for term in (getattr(env_cfg.events, name),)
            },
        }
        print(f"[EVAL] Mode={'robust' if args_cli.robust else 'nominal'}, seed={agent_cfg.seed}, "
              f"duration={len(schedule) * phase_steps * dt:.1f} simulation seconds")
        if args_cli.video:
            print("[EVAL] Video length limits recording only; evaluation runs the full schedule.")
    obs = env.get_observations()
    steps = 0
    completed = False
    try:
        while simulation_app.is_running():
            begin = time.time()
            if evaluator is not None and steps % phase_steps == 0:
                evaluator.finish_phase()
                cycle, name, command = schedule[steps // phase_steps]
                env.unwrapped.command_manager.get_term("base_velocity").set_command(command)
                # Refresh observations so the actor sees the new command on its first step.
                obs = env.get_observations()
                evaluator.begin_phase(cycle, name, command)
                print(f"[EVAL] Begin cycle={cycle} {name}: command={command}", flush=True)
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if hasattr(policy, "reset"):
                    policy.reset(dones)
            steps += 1
            if evaluator is not None and steps >= len(schedule) * phase_steps:
                completed = True
                break
            if evaluator is None and args_cli.video and steps >= args_cli.video_length:
                break
            delay = dt - (time.time() - begin)
            if args_cli.real_time and delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("[INFO] Interrupted; saving completed and partial evaluation phases.")
    finally:
        try:
            if evaluator is not None:
                evaluator.finish_phase()
                evaluator.save(output_dir, metadata, completed)
        finally:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
