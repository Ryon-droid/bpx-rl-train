"""Mild mixed-terrain locomotion with the existing 45-D BPX actor interface."""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from .bpx_flat_env_cfg import BPXFlatEnvCfg
from . import mdp as bpx_mdp


@configclass
class BPXRoughCurriculumCfg:
    """Promote or demote terrain levels from completed directional episodes."""

    terrain_levels = CurrTerm(func=bpx_mdp.terrain_levels_tracking)


@configclass
class BPXRoughEnvCfg(BPXFlatEnvCfg):
    """Train blind locomotion over flat ground, small bumps, slopes and stairs."""

    curriculum: BPXRoughCurriculumCfg = BPXRoughCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = 1
        self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            curriculum=True,
            use_cache=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
                # Uniform noise has a fixed range, independent of difficulty.
                "rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.3,
                    noise_range=(-0.02, 0.02),
                    noise_step=0.005,
                    border_width=0.25,
                ),
                "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.3,
                    slope_range=(0.0, 0.15),
                    platform_width=2.0,
                    border_width=0.25,
                ),
                "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.2,
                    step_height_range=(0.02, 0.08),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
            },
        )

        self.commands.base_velocity.class_type = bpx_mdp.DirectionalVelocityCommand
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.ranges.lin_vel_x = (-0.7, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.45, 0.45)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # These flat-task rewards measure world Z, not height above local terrain.
        self.rewards.base_height_l2 = None
        self.rewards.swing_foot_clearance = None
        self.rewards.flat_orientation_l2.weight = -0.2
