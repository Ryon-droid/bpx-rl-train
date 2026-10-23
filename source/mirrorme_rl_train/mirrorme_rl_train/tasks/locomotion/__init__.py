"""Gym registration for BPX flat-ground and mixed-terrain locomotion."""

import gymnasium as gym

from . import agents


gym.register(
    id="bpx_flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bpx_flat_env_cfg:BPXFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BPXFlatPPORunnerCfg",
    },
)

gym.register(
    id="bpx_rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bpx_rough_env_cfg:BPXRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BPXRoughPPORunnerCfg",
    },
)

gym.register(
    id="bpx_stairs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bpx_stairs_env_cfg:BPXStairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BPXStairsPPORunnerCfg",
    },
)
