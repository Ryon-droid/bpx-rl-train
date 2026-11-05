"""Stair traversal completion and a curriculum requiring full-flight progress."""

import torch


def stair_flight_boundary(env, distance: float = 3.3):
    # End on the outer landing, before wandering into an adjacent terrain tile.
    delta = env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return delta.abs().amax(dim=1) >= distance


def stair_levels_traversal(env, env_ids, distance: float = 3.3, replay_fraction: float = 0.0,
                           require_straight_progress: bool = False):
    terrain = env.scene.terrain
    command = env.command_manager.get_term("base_velocity")
    speed = command.command[env_ids, :2].norm(dim=1)
    moving = speed > 0.15
    delta = env.scene["robot"].data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    reached_landing = delta.abs().amax(dim=1) >= distance
    if require_straight_progress:
        velocity = command.command[env_ids]
        longitudinal = (velocity[:, 0].abs() > 0.15) & (velocity[:, 1:].abs() < 1e-5).all(dim=1)
        straight_success = (velocity[:, 0].sign() * delta[:, 0] >= distance) & (delta[:, 1].abs() < 0.75)
        reached_landing = torch.where(longitudinal, straight_success, reached_landing)
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
    # A pretrained teacher can revisit hard flights even after a failed reset.
    # The other resets retain the ordinary success/failure curriculum.
    if replay_fraction:
        replay_ids = env_ids[torch.rand(len(env_ids), device=env_ids.device) < replay_fraction]
        top = terrain.cfg.terrain_generator.num_rows
        terrain.terrain_levels[replay_ids] = torch.randint(top - 3, top, (len(replay_ids),), device=env_ids.device)
        terrain.env_origins[replay_ids] = terrain.terrain_origins[
            terrain.terrain_levels[replay_ids], terrain.terrain_types[replay_ids]]
    levels = terrain.terrain_levels.float()
    generator = terrain.cfg.terrain_generator
    term = next(iter(generator.sub_terrains.values()))
    low, high = term.step_height_range
    top_level = generator.num_rows - 1
    heights = (torch.tensor(term.step_heights, device=levels.device)
               if getattr(term, "step_heights", None) is not None
               else torch.linspace(low, high, generator.num_rows, device=levels.device))
    return {
        "mean_level": levels.mean(),
        "mean_step_height_cm": 100.0 * heights[terrain.terrain_levels].mean(),
        f"at_{round(high * 100)}cm_pct": 100.0 * (levels == top_level).float().mean(),
        "promoted_pct": 100.0 * up.sum() / moving.sum().clamp(min=1),
    }


def straight_motion_error(env, command_name: str = "base_velocity", lateral_scale: float = 2.0,
                          yaw_scale: float = 1.0):
    """Extra velocity tracking cost only for deliberate forward/backward flights.

    No position or hidden heading reference enters the policy observation.
    Lateral, turning, mixed and standing commands retain their original rewards.
    """
    command = env.command_manager.get_command(command_name)
    active = (command[:, 0].abs() > 0.15) & (command[:, 1:].abs() < 1e-5).all(dim=1)
    robot = env.scene["robot"].data
    return active * (lateral_scale * robot.root_lin_vel_b[:, 1].square()
                     + yaw_scale * robot.root_ang_vel_b[:, 2].square())
