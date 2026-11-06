"""Simulation-only terrain teachers and camera-free history students.

The original flat/rough/stairs tasks keep their 45-D actor contract. These
separate tasks deliberately have different observation/checkpoint contracts.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg, TerminationTermCfg, RewardTermCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from .bpx_flat_env_cfg import BASE_BODY_NAME, BPXSceneCfg, ObservationsCfg
from .bpx_rough_env_cfg import BPXRoughEnvCfg
from .bpx_stairs_env_cfg import BPXStairsEnvCfg
from .stair_terrain import stairs15_generator, stairs16_generator, stairs17_generator, stairs20_generator
from .mdp.commands import FocusedStairVelocityCommand, StairVelocityCommand
from .mdp.stairs import stair_flight_boundary, stair_levels_traversal, straight_motion_error

HISTORY_LENGTH = 20  # 50 Hz control: 0.4 s, 45 values per observation.
HEIGHT_SCAN_DIM = 187  # Inclusive 17 x 11 grid, 0.1 m resolution.


@configclass
class BPXPerceptionSceneCfg(BPXSceneCfg):
    """Ground-truth terrain sensing available only inside the simulator."""

    height_scanner = RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{BASE_BODY_NAME}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class BPXPerceptionObservationsCfg(ObservationsCfg):
    @configclass
    class TerrainCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    terrain: TerrainCfg = TerrainCfg()


@configclass
class BPXBlindObservationsCfg(BPXPerceptionObservationsCfg):
    """Student history contains only the six deployable policy terms."""

    proprio_history: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()

    def __post_init__(self):
        self.proprio_history.history_length = HISTORY_LENGTH
        self.proprio_history.flatten_history_dim = True


@configclass
class BPXRoughTeacherEnvCfg(BPXRoughEnvCfg):
    scene: BPXPerceptionSceneCfg = BPXPerceptionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: BPXPerceptionObservationsCfg = BPXPerceptionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class BPXStairsTeacherEnvCfg(BPXStairsEnvCfg):
    scene: BPXPerceptionSceneCfg = BPXPerceptionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: BPXPerceptionObservationsCfg = BPXPerceptionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class BPXRoughBlindEnvCfg(BPXRoughTeacherEnvCfg):
    observations: BPXBlindObservationsCfg = BPXBlindObservationsCfg()


@configclass
class BPXStairsBlindEnvCfg(BPXStairsTeacherEnvCfg):
    observations: BPXBlindObservationsCfg = BPXBlindObservationsCfg()


@configclass
class BPXStairs20TeacherEnvCfg(BPXStairsTeacherEnvCfg):
    """Progress from 4 cm to exact 20 cm risers on uphill/downhill flights."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = stairs20_generator()
        self.scene.terrain.max_init_terrain_level = 2  # 4, 6 and 8 cm starts.
        self.curriculum.terrain_levels = CurriculumTermCfg(func=stair_levels_traversal)
        self.terminations.stair_landing = TerminationTermCfg(func=stair_flight_boundary, time_out=True)
        self.events.reset_base.params["pose_range"].update(
            x=(-0.1, 0.1), y=(-0.1, 0.1), yaw=(-0.15, 0.15)
        )
        self.commands.base_velocity.ranges.lin_vel_x = (-0.40, 0.40)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.30, 0.30)
        # Allow body rise and leg folding needed for high stairs; keep motor
        # torque, joint-position limits and physical failure criteria intact.
        self.rewards.lin_vel_z_l2.weight = -0.25
        self.rewards.leg_posture_deviation_l1.weight = -0.04
        self.rewards.action_rate_l2.weight = -0.05
        # Per-joint 24 V motor limits are set centrally in resources/mirrorme.py.


@configclass
class BPXStairs20BlindEnvCfg(BPXStairs20TeacherEnvCfg):
    observations: BPXBlindObservationsCfg = BPXBlindObservationsCfg()


@configclass
class BPXStairs16TeacherEnvCfg(BPXStairs20TeacherEnvCfg):
    """Progress from 4 cm to exact 16 cm risers on uphill/downhill flights."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = stairs16_generator()


@configclass
class BPXStairs16BlindEnvCfg(BPXStairs16TeacherEnvCfg):
    observations: BPXBlindObservationsCfg = BPXBlindObservationsCfg()


@configclass
class BPXStairs15TeacherEnvCfg(BPXStairs20TeacherEnvCfg):
    """Exact 4/6/8/10/12/14/15 cm tiers, with the usual traversal curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = stairs15_generator()


@configclass
class BPXStairs15FocusTeacherEnvCfg(BPXStairs15TeacherEnvCfg):
    """Fine-tune a compatible pretrained teacher on its descent weakness."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 6
        self.curriculum.terrain_levels.params["replay_fraction"] = 0.5
        self.commands.base_velocity.class_type = FocusedStairVelocityCommand


@configclass
class BPXStairs15BlindEnvCfg(BPXStairs15TeacherEnvCfg):
    observations: BPXBlindObservationsCfg = BPXBlindObservationsCfg()


@configclass
class BPXStairs17TeacherEnvCfg(BPXStairs20TeacherEnvCfg):
    """Balanced forward/backward practice up to exact 17 cm risers."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = stairs17_generator()
        self.scene.terrain.max_init_terrain_level = 7
        self.commands.base_velocity.class_type = StairVelocityCommand
        self.curriculum.terrain_levels.params.update(replay_fraction=0.35, require_straight_progress=True)
        self.rewards.straight_motion_error = RewardTermCfg(func=straight_motion_error, weight=-1.0)


@configclass
class BPXStairs17BlindEnvCfg(BPXStairs17TeacherEnvCfg):
    observations: BPXBlindObservationsCfg = BPXBlindObservationsCfg()
