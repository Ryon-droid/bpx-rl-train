"""Bounded 17 cm teacher repair without changing existing task definitions."""

from isaaclab.terrains import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

from .bpx_privileged_env_cfg import BPXStairs17TeacherEnvCfg
from .mdp.commands import RepairStairVelocityCommand
from .mdp.stairs import repair_stair_levels_traversal


@configclass
class BPXStairs17RepairTeacherEnvCfg(BPXStairs17TeacherEnvCfg):
    """One third flat motion practice, two thirds original up/down stairs."""

    def __post_init__(self):
        super().__post_init__()
        generator = self.scene.terrain.terrain_generator
        generator.num_cols = 12
        for term in generator.sub_terrains.values():
            term.proportion = 1 / 3
        generator.sub_terrains['flat'] = MeshPlaneTerrainCfg(proportion=1 / 3)
        self.commands.base_velocity.class_type = RepairStairVelocityCommand
        self.curriculum.terrain_levels.func = repair_stair_levels_traversal
        self.curriculum.terrain_levels.params['flat_column_start'] = 8
        # All sampled lateral commands >= .10, yaw commands >= .25*.5=.125.
        # Both must activate movement rewards and disable the standing penalty.
        self.rewards.stand_still.params['command_threshold'] = .08
        self.rewards.feet_air_time.params['command_threshold'] = .08
