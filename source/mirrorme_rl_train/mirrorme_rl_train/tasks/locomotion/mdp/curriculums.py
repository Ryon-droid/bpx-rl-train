"""Curriculum terms for mixed-terrain locomotion."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter


def terrain_levels_tracking(
    env,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Update terrain levels from completion, tracking quality and displacement.

    Standing and pure-yaw episodes do not affect terrain difficulty. A walking
    episode is promoted only when it times out, tracks its command and covers a
    useful distance. Clear failures are demoted; borderline episodes hold.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command_term = env.command_manager.get_term("base_velocity")
    command = command_term.command[env_ids]
    speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    moving = speed > 0.15

    distance = torch.linalg.vector_norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    target_distance = torch.clamp(speed * env.max_episode_length_s * 0.4, min=1.5, max=3.0)
    tracking_error = command_term.metrics["error_vel_xy"][env_ids]
    promote_error = torch.maximum(0.35 * speed, torch.full_like(speed, 0.12))
    demote_error = torch.maximum(0.65 * speed, torch.full_like(speed, 0.20))
    timed_out = env.termination_manager.time_outs[env_ids]
    failed = env.termination_manager.terminated[env_ids]

    move_up = moving & timed_out & ~failed & (distance >= target_distance) & (tracking_error <= promote_error)
    move_down = moving & (
        failed | (~timed_out) | (distance < 0.5 * target_distance) | (tracking_error > demote_error)
    )
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    moving_count = moving.sum().clamp(min=1)
    return {
        "mean_level": terrain.terrain_levels.float().mean(),
        "promoted_pct": 100.0 * move_up.sum() / moving_count,
        "demoted_pct": 100.0 * move_down.sum() / moving_count,
    }
