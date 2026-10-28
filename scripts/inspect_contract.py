#!/usr/bin/env python3
"""Start one Isaac Lab environment and assert the public 45-D/12-D contract."""

import argparse
import sys

from policy_contract import actor_input_dim, FRAME_DIM, HISTORY_LENGTH, TERRAIN_DIM

from project_paths import add_project_source

add_project_source()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="bpx_flat", help="Registered Isaac Lab task ID.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import mirrorme_rl_train.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

ACTION_DIM = 12
OBSERVATION_DIM = 45
JOINT_NAMES = (
    "fl_hip_roll_joint",
    "fr_hip_roll_joint",
    "hl_hip_roll_joint",
    "hr_hip_roll_joint",
    "fl_hip_pitch_joint",
    "fr_hip_pitch_joint",
    "hl_hip_pitch_joint",
    "hr_hip_pitch_joint",
    "fl_knee_joint",
    "fr_knee_joint",
    "hl_knee_joint",
    "hr_knee_joint",
)


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=1)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env = gym.make(args_cli.task, cfg=cfg)
    obs, _ = env.reset()
    policy_obs = obs["policy"]
    privileged_obs = obs["privileged"]
    action_dim = env.unwrapped.action_manager.total_action_dim
    actual_joint_names = tuple(env.unwrapped.scene["robot"].joint_names)
    missing = [name for name in JOINT_NAMES if name not in actual_joint_names]

    print(f"policy observation shape: {tuple(policy_obs.shape)}")
    print(f"privileged observation shape: {tuple(privileged_obs.shape)}")
    actor_dim = sum(obs[name].shape[-1] for name in agent_cfg.obs_groups["policy"])
    for name, tensor in obs.items():
        print(f"observation group {name}: {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"non-finite observation group: {name}")
    print(f"actor input dimension: {actor_dim}")
    print(f"action dimension: {action_dim}")
    print("policy joint order:")
    for index, name in enumerate(JOINT_NAMES):
        print(
            f"  {index:02d}: {name} "
            f"(robot index={actual_joint_names.index(name) if name in actual_joint_names else 'MISSING'})"
        )

    if policy_obs.shape[-1] != OBSERVATION_DIM:
        raise RuntimeError(f"expected {OBSERVATION_DIM} observations, got {policy_obs.shape[-1]}")
    if privileged_obs.shape[-1] != 3:
        raise RuntimeError(f"expected 3 privileged observations, got {privileged_obs.shape[-1]}")
    if actor_dim != actor_input_dim(args_cli.task):
        raise RuntimeError(f"actor expected {actor_input_dim(args_cli.task)} inputs, got {actor_dim}")
    if args_cli.task.endswith(("_teacher", "_blind")) and obs["terrain"].shape[-1] != TERRAIN_DIM:
        raise RuntimeError(f"expected {TERRAIN_DIM} terrain samples")
    if args_cli.task.endswith("_blind"):
        if agent_cfg.obs_groups["policy"] != ["proprio_history"]:
            raise RuntimeError("blind student must receive proprio_history only")
        if obs["proprio_history"].shape[-1] != FRAME_DIM * HISTORY_LENGTH:
            raise RuntimeError("wrong proprioceptive history dimension")
    if action_dim != ACTION_DIM:
        raise RuntimeError(f"expected {ACTION_DIM} actions, got {action_dim}")
    if missing:
        raise RuntimeError(f"robot is missing policy joints: {missing}")
    if "critic" in agent_cfg.obs_groups:
        critic_dim = sum(obs[name].shape[-1] for name in agent_cfg.obs_groups["critic"])
        print(f"[PASS] Actor {actor_dim}-D, Critic {critic_dim}-D, Action 12-D runtime contract.")
    else:
        print(f"[PASS] Blind student {actor_dim}-D history -> 12-D actions; no terrain/velocity input.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
