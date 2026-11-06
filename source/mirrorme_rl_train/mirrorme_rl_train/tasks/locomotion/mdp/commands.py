"""Command generators specialized for mixed-terrain training."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand


class DirectionalVelocityCommand(UniformVelocityCommand):
    """Sample one deliberate motion mode and keep it for the configured interval.

    Pure longitudinal and lateral commands make terrain progress measurable and
    ensure that left/right strafing is represented equally during training.
    """

    mode_fractions = (0.35, 0.30, 0.10, 0.15, 0.10)

    def _sample_signed_magnitude(self, count: int, low: float, high: float) -> torch.Tensor:
        magnitude = torch.empty(count, device=self.device).uniform_(low, high)
        sign = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            -torch.ones(count, device=self.device),
            torch.ones(count, device=self.device),
        )
        return magnitude * sign

    def _resample_command(self, env_ids: Sequence[int]):
        self._resample_modes(env_ids, self.mode_fractions)

    def _resample_modes(self, env_ids: Sequence[int], fractions):
        count = len(env_ids)
        if count == 0:
            return

        self.vel_command_b[env_ids] = 0.0
        mode = torch.rand(count, device=self.device)
        longitudinal_fraction, lateral_fraction, yaw_fraction, combined_fraction, _ = fractions
        longitudinal_end = longitudinal_fraction
        lateral_end = longitudinal_end + lateral_fraction
        yaw_end = lateral_end + yaw_fraction
        combined_end = yaw_end + combined_fraction
        longitudinal = mode < longitudinal_end
        lateral = (mode >= longitudinal_end) & (mode < lateral_end)
        yaw = (mode >= lateral_end) & (mode < yaw_end)
        combined = (mode >= yaw_end) & (mode < combined_end)
        standing = mode >= combined_end

        x_max = max(abs(value) for value in self.cfg.ranges.lin_vel_x)
        y_max = max(abs(value) for value in self.cfg.ranges.lin_vel_y)
        yaw_max = max(abs(value) for value in self.cfg.ranges.ang_vel_z)

        self.vel_command_b[env_ids[longitudinal], 0] = self._sample_signed_magnitude(
            int(longitudinal.sum()), min(0.25, x_max), x_max
        )
        self.vel_command_b[env_ids[lateral], 1] = self._sample_signed_magnitude(
            int(lateral.sum()), min(0.10, y_max), y_max
        )
        self.vel_command_b[env_ids[yaw], 2] = self._sample_signed_magnitude(
            int(yaw.sum()), min(0.25, yaw_max), yaw_max
        )

        combined_ids = env_ids[combined]
        combined_count = int(combined.sum())
        if combined_count:
            self.vel_command_b[combined_ids, 0] = self._sample_signed_magnitude(
                combined_count, min(0.20, x_max), x_max
            )
            self.vel_command_b[combined_ids, 1] = self._sample_signed_magnitude(
                combined_count, min(0.15, y_max), y_max
            )
            self.vel_command_b[combined_ids, 2] = self._sample_signed_magnitude(
                combined_count, min(0.20, yaw_max), yaw_max
            )

        self.is_standing_env[env_ids] = standing
        self.is_heading_env[env_ids] = False


class StairVelocityCommand(DirectionalVelocityCommand):
    """Bias commands toward forward and backward stair traversal."""

    mode_fractions = (0.70, 0.10, 0.05, 0.10, 0.05)


class RepairStairVelocityCommand(StairVelocityCommand):
    """Keep stair practice and emphasize lateral/yaw commands on flat tiles."""

    flat_column_start = 8  # Repair generator: four up, four down, four flat columns.
    flat_mode_fractions = (0.10, 0.35, 0.35, 0.10, 0.10)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.terrain = env.scene.terrain

    def _resample_command(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        flat = self.terrain.terrain_types[env_ids] >= self.flat_column_start
        self._resample_modes(env_ids[~flat], self.mode_fractions)
        self._resample_modes(env_ids[flat], self.flat_mode_fractions)


class FocusedStairVelocityCommand(StairVelocityCommand):
    """Retain motion modes, with more backward descents on pyramid tiles."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.terrain = env.scene.terrain

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        velocity = self.vel_command_b[env_ids]
        longitudinal = (velocity[:, 0] != 0) & (velocity[:, 1:] == 0).all(dim=1)
        ids = env_ids[longitudinal]
        # This command belongs to the 50% up / 50% down stair generator.
        down = self.terrain.terrain_types[ids] >= self.terrain.cfg.terrain_generator.num_cols // 2
        probability = torch.where(down, 0.80, 0.50)
        negative = torch.rand(len(ids), device=self.device) < probability
        self.vel_command_b[ids, 0] = self.vel_command_b[ids, 0].abs() * torch.where(negative, -1.0, 1.0)
