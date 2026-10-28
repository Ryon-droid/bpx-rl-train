#!/usr/bin/env python3
"""Export a BPX RSL-RL checkpoint without opening the interactive play loop."""

import argparse
import os
import sys

from project_paths import add_project_source

add_project_source()

from isaaclab.app import AppLauncher

import cli_args


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="bpx_flat", help="Registered Isaac Lab task ID.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--output_dir", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
default_hydra_args = ["hydra.run.dir=logs/hydra/${now:%Y-%m-%d}/${now:%H-%M-%S}"]
sys.argv = [sys.argv[0]] + default_hydra_args + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import mirrorme_rl_train.tasks  # noqa: F401
from policy_contract import actor_input_dim
from runner_utils import export_runner_policy, load_runner_checkpoint, make_runner


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        checkpoint = retrieve_file_path(args_cli.checkpoint)
    else:
        checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    output_dir = args_cli.output_dir or os.path.join(os.path.dirname(checkpoint), "exported")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = make_runner(env, agent_cfg)
    load_runner_checkpoint(runner, checkpoint, load_optimizer=False)
    basename = export_runner_policy(runner, args_cli.task, output_dir)
    print(f"[PASS] checkpoint: {checkpoint}")
    print(f"[PASS] TorchScript: {os.path.join(output_dir, basename + '.pt')}")
    print(f"[PASS] ONNX:       {os.path.join(output_dir, basename + '.onnx')}")
    print(f"[PASS] interface: obs [1,{actor_input_dim(args_cli.task)}] -> actions [1,12]")
    if args_cli.task.endswith("_teacher"):
        print("[WARNING] Teacher export requires simulation-only terrain/velocity; do not deploy on a blind robot.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
