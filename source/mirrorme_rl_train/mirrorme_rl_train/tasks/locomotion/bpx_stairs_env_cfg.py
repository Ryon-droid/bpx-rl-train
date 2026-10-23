"""Stair-focused locomotion using the existing BPX actor interface."""

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from . import mdp as bpx_mdp
from .bpx_rough_env_cfg import BPXRoughEnvCfg


@configclass
class BPXStairsEnvCfg(BPXRoughEnvCfg):
    """Train on ascending and descending pyramid stairs with graded difficulty."""

    def __post_init__(self):
        super().__post_init__()

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
                "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=1.0,
                    step_height_range=(0.02, 0.10),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
            },
        )
        self.commands.base_velocity.class_type = bpx_mdp.StairVelocityCommand
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
