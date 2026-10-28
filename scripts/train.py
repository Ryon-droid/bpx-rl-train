#!/usr/bin/env python3
"""Train BPX with PPO or distill a privileged teacher into a blind student."""

import argparse
import os
import sys
import time
from datetime import datetime

from project_paths import add_project_source

add_project_source()

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="bpx_flat", help="Registered Isaac Lab task ID.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=10000)
parser.add_argument("--teacher_checkpoint", type=str, default=None,
                    help="PPO teacher checkpoint for a new *_blind distillation run.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
default_hydra_args = ["hydra.run.dir=logs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}"]
sys.argv = [sys.argv[0]] + default_hydra_args + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import mirrorme_rl_train.tasks  # noqa: F401
from runner_utils import load_runner_checkpoint, make_runner


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    distillation = agent_cfg.class_name == "DistillationRunner"
    if args_cli.teacher_checkpoint and (not distillation or agent_cfg.resume):
        raise ValueError("--teacher_checkpoint requires a new *_blind run, without --resume")
    if distillation and not agent_cfg.resume and not args_cli.teacher_checkpoint:
        raise ValueError("A new *_blind run requires --teacher_checkpoint PATH; resume students with --resume --checkpoint PATH")

    log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, run_name)
    print(f"[INFO] Log directory: {log_dir}")

    resume_path = None
    if args_cli.teacher_checkpoint:
        resume_path = os.path.abspath(os.path.expanduser(args_cli.teacher_checkpoint))
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(resume_path)
    elif agent_cfg.resume:
        # Fine-tuning may intentionally use a checkpoint from a different
        # experiment (for example full-speed -> low-speed).  Preserve the
        # existing run-relative lookup while accepting an explicit file path.
        if agent_cfg.load_checkpoint and os.path.isfile(agent_cfg.load_checkpoint):
            resume_path = os.path.abspath(agent_cfg.load_checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = make_runner(env, agent_cfg, log_dir=log_dir)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        print(f"[INFO] Loading checkpoint: {resume_path}")
        load_runner_checkpoint(runner, resume_path, teacher_initialization=bool(args_cli.teacher_checkpoint))

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    start = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[INFO] Training time: {time.time() - start:.1f} s")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
