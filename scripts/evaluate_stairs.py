#!/usr/bin/env python3
"""Evaluate one complete stair-flight attempt per robot, without counting resets as successes."""

import argparse
import json
from pathlib import Path

from project_paths import add_project_source

add_project_source()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="bpx_stairs20_teacher")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--step_height", type=float, default=0.20)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--speed", type=float, default=0.30)
parser.add_argument("--backward", action="store_true")
parser.add_argument("--robust", action="store_true")
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not (0.02 <= args.step_height <= 0.20 and 0.15 <= args.speed <= 0.40 and args.num_envs >= 2):
    parser.error("height must be 0.02..0.20 m, speed 0.15..0.40 m/s, and num_envs >= 2")
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import mirrorme_rl_train.tasks  # noqa: F401
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import TerminationTermCfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from runner_utils import load_runner_checkpoint, make_runner


class TraversalCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids):
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 0] = -args.speed if args.backward else args.speed
        self.is_standing_env[env_ids] = False
        self.is_heading_env[env_ids] = False


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.curriculum.terrain_levels = None
    cfg.scene.terrain.max_init_terrain_level = 0
    gen = cfg.scene.terrain.terrain_generator
    gen.num_rows, gen.num_cols = 1, 2
    for term in gen.sub_terrains.values():
        term.step_height_range = (args.step_height, args.step_height)
    cfg.commands.base_velocity.class_type = TraversalCommand
    cfg.commands.base_velocity.debug_vis = False
    cfg.events.reset_base.params["pose_range"].update(x=(-0.1, 0.1), y=(-0.1, 0.1), yaw=(-0.10, 0.10))
    # Seeded small start variations avoid evaluating identical copies of a robot.
    cfg.events.reset_base.params["velocity_range"] = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.events.reset_robot_joints.params.update(position_range=(0.98, 1.02), velocity_range=(0.0, 0.0))
    if not args.robust:
        cfg.observations.policy.enable_corruption = False
        if hasattr(cfg.observations, "proprio_history"):
            cfg.observations.proprio_history.enable_corruption = False
        for name in ("physics_material", "add_base_mass", "base_com", "push_robot"):
            setattr(cfg.events, name, None)
    finished = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
    outcomes = []
    direction = -1.0 if args.backward else 1.0

    def record_before_reset(env):
        position = env.scene["robot"].data.root_pos_w
        origins = env.scene.env_origins
        delta = position - origins
        progress = direction * delta[:, 0]
        # The two columns are a pit (uphill) and a pyramid (downhill).
        uphill = env.scene.terrain.terrain_types == 0
        height_progress = torch.where(uphill, delta[:, 2] - 0.48, 0.48 - delta[:, 2])
        failed = env.termination_manager.terminated
        ended = failed | env.termination_manager.time_outs
        success = (progress >= 3.3) & (delta[:, 1].abs() < 0.75) & (height_progress >= origins[:, 2].abs() - 0.25) & ~failed
        new = ended & ~finished
        ids = new.nonzero().flatten()
        if len(ids):
            for i in ids.tolist():
                outcomes.append({"env": i, "terrain": "up" if bool(uphill[i]) else "down",
                                 "success": bool(success[i]), "physical_failure": bool(failed[i]),
                                 "base_contact": bool(env.termination_manager.get_term("base_contact")[i]),
                                 "bad_orientation": bool(env.termination_manager.get_term("bad_orientation")[i]),
                                 "progress_m": float(progress[i]), "lateral_offset_m": float(delta[i, 1]),
                                 "vertical_progress_m": float(height_progress[i]),
                                 "elapsed_s": float(env.episode_length_buf[i]) * env.step_dt})
            finished[ids] = True
        return torch.zeros_like(finished)

    cfg.terminations.traversal_record = TerminationTermCfg(func=record_before_reset)
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=100.0)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device
    runner = make_runner(env, agent_cfg)
    load_runner_checkpoint(runner, args.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)
    obs = env.get_observations()
    for _ in range(env.unwrapped.max_episode_length + 2):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            if hasattr(policy, "reset"):
                policy.reset(dones)
        if bool(finished.all()):
            break
    if not bool(finished.all()):
        raise RuntimeError("evaluation ended before all first attempts were recorded")
    report = {"checkpoint": str(Path(args.checkpoint).resolve()), "task": args.task,
              "height_m": args.step_height, "tread_m": 0.30, "seed": args.seed, "speed_m_s": direction * args.speed,
              "robust": args.robust, "num_envs": args.num_envs, "attempts": outcomes,
              "notes": "Only the first attempt per robot counts; landing requires forward progress, vertical traversal, no physical failure, and lateral offset <0.75m. Later resets do not count."}
    for name in ("up", "down"):
        attempts = [x for x in outcomes if x["terrain"] == name]
        report[name] = {"attempts": len(attempts), "successes": sum(x["success"] for x in attempts),
                        "physical_failures": sum(x["physical_failure"] for x in attempts)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("[TRAVERSAL]", {name: report[name] for name in ("up", "down")}, flush=True)
    print("[TRAVERSAL] report:", output, flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
