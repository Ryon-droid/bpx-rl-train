"""Stair traversal completion and a curriculum requiring full-flight progress."""

import torch


def stair_flight_boundary(env, distance: float = 3.3):
    # End on the outer landing, before wandering into an adjacent terrain tile.
    delta = env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return delta.abs().amax(dim=1) >= distance


def stair_levels_traversal(env, env_ids, distance: float = 3.3):
    terrain = env.scene.terrain
    command = env.command_manager.get_term("base_velocity")
    speed = command.command[env_ids, :2].norm(dim=1)
    moving = speed > 0.15
    delta = env.scene["robot"].data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    reached_landing = delta.abs().amax(dim=1) >= distance
    failed = env.termination_manager.terminated[env_ids]
    timed_out = env.termination_manager.time_outs[env_ids]
    # UniformVelocityCommand integrates error over its configured 20 s period.
    # Boundary-truncated flights are shorter; recover their actual mean error.
    elapsed = (env.episode_length_buf[env_ids] * env.step_dt).clamp(min=env.step_dt)
    tracking = command.metrics["error_vel_xy"][env_ids] * env.max_episode_length_s / elapsed
    acceptable = tracking <= torch.maximum(0.4 * speed, torch.full_like(speed, 0.12))
    # Initial resets and standing/yaw episodes do not count as traversal attempts.
    completed = env.episode_length_buf[env_ids] > 0
    up = moving & completed & timed_out & ~failed & reached_landing & acceptable
    down = moving & completed & (failed | (timed_out & ~reached_landing)) & ~up
    terrain.update_env_origins(env_ids, up, down)
    levels = terrain.terrain_levels.float()
    return {
        "mean_level": levels.mean(),
        "mean_step_height_cm": 4.0 + 2.0 * levels.mean(),
        "at_20cm_pct": 100.0 * (levels == 8).float().mean(),
        "promoted_pct": 100.0 * up.sum() / moving.sum().clamp(min=1),
    }
