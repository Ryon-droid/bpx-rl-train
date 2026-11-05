"""Exact stair-height tiers with both uphill and downhill starting platforms."""

import math

import isaaclab.terrains as terrains
from isaaclab.terrains.trimesh.mesh_terrains import inverted_pyramid_stairs_terrain, pyramid_stairs_terrain
from isaaclab.utils import configclass


def tiered_stairs(difficulty, cfg):
    # Isaac Lab jitters difficulty within each row. Quantize it so the last
    # row really contains the configured maximum riser height.
    level = min(cfg.num_levels - 1, max(0, math.floor(difficulty * cfg.num_levels)))
    low, high = cfg.step_height_range
    height = low + (high - low) * level / (cfg.num_levels - 1)
    fixed = cfg.replace(step_height_range=(height, height))
    generate = inverted_pyramid_stairs_terrain if cfg.inverted else pyramid_stairs_terrain
    return generate(0.0, fixed)


@configclass
class TieredStairsTerrainCfg(terrains.MeshPyramidStairsTerrainCfg):
    function = tiered_stairs
    num_levels: int = 9
    inverted: bool = False


def stairs20_generator():
    return stair_generator(max_height=0.20, num_levels=9)


def stairs16_generator():
    return stair_generator(max_height=0.16, num_levels=7)


def stair_generator(*, max_height, num_levels):
    common = dict(step_height_range=(0.04, max_height), num_levels=num_levels, step_width=0.30,
                  platform_width=3.0, border_width=1.0, holes=False)
    return terrains.TerrainGeneratorCfg(
        size=(8.0, 8.0), border_width=10.0, num_rows=num_levels, num_cols=10,
        curriculum=True, use_cache=False, color_scheme="height",
        sub_terrains={
            "up": TieredStairsTerrainCfg(proportion=0.5, inverted=True, **common),
            "down": TieredStairsTerrainCfg(proportion=0.5, inverted=False, **common),
        },
    )
