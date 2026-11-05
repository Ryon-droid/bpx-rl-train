"""Test directional sampling and terrain curriculum without launching Isaac Sim."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch


def load_mdp_module(name):
    stubs = {key: ModuleType(key) for key in (
        "isaaclab", "isaaclab.assets", "isaaclab.managers", "isaaclab.terrains",
        "isaaclab.envs", "isaaclab.envs.mdp", "isaaclab.envs.mdp.commands",
    )}
    stubs["isaaclab.assets"].Articulation = object
    stubs["isaaclab.terrains"].TerrainImporter = object
    stubs["isaaclab.managers"].SceneEntityCfg = lambda name: SimpleNamespace(name=name)
    stubs["isaaclab.envs.mdp.commands"].UniformVelocityCommand = object
    path = Path(__file__).resolve().parents[1] / "source/mirrorme_rl_train/mirrorme_rl_train/tasks/locomotion/mdp"
    spec = importlib.util.spec_from_file_location(f"rough_test_{name}", path / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


commands_module = load_mdp_module("commands")
DirectionalVelocityCommand = commands_module.DirectionalVelocityCommand
StairVelocityCommand = commands_module.StairVelocityCommand
FocusedStairVelocityCommand = commands_module.FocusedStairVelocityCommand
terrain_levels_tracking = load_mdp_module("curriculums").terrain_levels_tracking
stair_levels_traversal = load_mdp_module("stairs").stair_levels_traversal


class TerrainCurriculumTests(unittest.TestCase):
    def make_env(self, commands, distances, errors, timeouts, failures):
        count = len(commands)

        class Scene(dict):
            pass

        scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor([[distance, 0.0, 0.0] for distance in distances])
        )))
        scene.env_origins = torch.zeros(count, 3)
        scene.terrain = SimpleNamespace(
            terrain_levels=torch.ones(count, dtype=torch.long), update_env_origins=Mock(),
            cfg=SimpleNamespace(terrain_generator=SimpleNamespace(
                num_rows=9, sub_terrains={"up": SimpleNamespace(step_height_range=(0.04, 0.20))}
            )),
        )
        command_term = SimpleNamespace(
            command=torch.tensor(commands, dtype=torch.float32),
            metrics={"error_vel_xy": torch.tensor(errors)},
        )
        return SimpleNamespace(
            scene=scene, max_episode_length_s=20,
            command_manager=SimpleNamespace(get_term=lambda name: command_term),
            termination_manager=SimpleNamespace(
                time_outs=torch.tensor(timeouts), terminated=torch.tensor(failures)
            ),
        )

    def assert_updates(self, env, up, down, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(len(env.scene.env_origins))
        result = terrain_levels_tracking(env, env_ids)
        ids, actual_up, actual_down = env.scene.terrain.update_env_origins.call_args.args
        self.assertTrue(torch.equal(ids, env_ids))
        self.assertEqual(actual_up.tolist(), up)
        self.assertEqual(actual_down.tolist(), down)
        self.assertEqual(result["mean_level"].item(), 1.0)
        self.assertFalse((actual_up & actual_down).any())
        return result

    def test_success_and_timeout_failure_overlap(self):
        env = self.make_env([[0.5, 0, 0]] * 3, [3.2] * 3, [0.1] * 3,
                            [True, True, False], [False, True, True])
        result = self.assert_updates(env, [True, False, False], [False, True, True])
        self.assertAlmostEqual(result["promoted_pct"].item(), 100 / 3, places=4)

    def test_standing_pure_yaw_and_initial_reset_hold(self):
        env = self.make_env([[0, 0, 0], [0, 0, 0.8], [0, 0, 0]],
                            [0.0] * 3, [0.0] * 3, [False] * 3, [True, True, False])
        result = self.assert_updates(env, [False] * 3, [False] * 3)
        self.assertEqual(result["demoted_pct"].item(), 0)

    def test_distance_tracking_and_borderline_hold(self):
        env = self.make_env([[0, -0.4, 0]] * 4, [1.0, 3.1, 2.0, 3.1],
                            [0.1, 0.3, 0.18, 0.1], [True] * 4, [False] * 4)
        self.assert_updates(env, [False, False, False, True], [True, True, False, False])

    def test_subset_uses_only_reset_environments(self):
        env = self.make_env([[0.5, 0, 0]] * 3, [3.2, 3.2, 0.0], [0.1] * 3,
                            [True] * 3, [False] * 3)
        self.assert_updates(env, [False, True], [True, False], torch.tensor([2, 0]))


class StairTraversalCurriculumTests(unittest.TestCase):
    make_env = TerrainCurriculumTests.make_env

    def test_nonuniform_15cm_tiers_and_replay_origins(self):
        env = self.make_env([[0.3, 0, 0]] * 20, [3.4] * 20, [0.02] * 20,
                            [True] * 20, [False] * 20)
        env.episode_length_buf = torch.full((20,), 600)
        env.step_dt = 0.02
        terrain = env.scene.terrain
        generator = terrain.cfg.terrain_generator
        generator.num_rows = 7
        generator.sub_terrains["up"].step_height_range = (0.04, 0.15)
        generator.sub_terrains["up"].step_heights = (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.15)
        terrain.terrain_types = torch.arange(20) % 2
        terrain.terrain_origins = torch.arange(42).reshape(7, 2, 3).float()
        terrain.env_origins = torch.zeros(20, 3)
        with torch.random.fork_rng():
            torch.manual_seed(2026)
            result = stair_levels_traversal(env, torch.arange(20), replay_fraction=1.0)
        self.assertTrue(((terrain.terrain_levels >= 4) & (terrain.terrain_levels <= 6)).all())
        self.assertTrue(torch.equal(terrain.env_origins, terrain.terrain_origins[
            terrain.terrain_levels, terrain.terrain_types]))
        expected = torch.tensor([12., 14., 15.])[terrain.terrain_levels - 4].mean().item()
        self.assertAlmostEqual(result["mean_step_height_cm"].item(), expected, places=4)
        self.assertIn("at_15cm_pct", result)

    def test_metrics_follow_16cm_curriculum_cap(self):
        env = self.make_env([[0.3, 0, 0]] * 2, [3.4] * 2, [0.02] * 2,
                            [True] * 2, [False] * 2)
        env.episode_length_buf = torch.tensor([600, 600])
        env.step_dt = 0.02
        generator = env.scene.terrain.cfg.terrain_generator
        generator.num_rows = 7
        generator.sub_terrains["up"].step_height_range = (0.04, 0.16)
        env.scene.terrain.terrain_levels = torch.tensor([0, 6])
        result = stair_levels_traversal(env, torch.arange(2))
        self.assertAlmostEqual(result["mean_step_height_cm"].item(), 10.0, places=4)
        self.assertEqual(result["at_16cm_pct"].item(), 50.0)
        self.assertNotIn("at_20cm_pct", result)

    def test_full_flight_required_and_physical_failure_wins(self):
        env = self.make_env([[0.3, 0, 0]] * 4,
                            [2.2, 3.4, 3.4, 0.2], [0.02] * 4,
                            [True, True, True, False], [False, False, True, False])
        env.episode_length_buf = torch.tensor([1000, 600, 600, 0])
        env.step_dt = 0.02
        stair_levels_traversal(env, torch.arange(4))
        _, up, down = env.scene.terrain.update_env_origins.call_args.args
        self.assertEqual(up.tolist(), [False, True, False, False])
        self.assertEqual(down.tolist(), [True, False, True, False])

    def test_standing_and_initial_reset_cannot_advance(self):
        env = self.make_env([[0, 0, 0], [0.3, 0, 0]],
                            [3.4, 3.4], [0.0, 0.0], [True, True], [False, False])
        env.episode_length_buf = torch.tensor([500, 0])
        env.step_dt = 0.02
        stair_levels_traversal(env, torch.arange(2))
        _, up, down = env.scene.terrain.update_env_origins.call_args.args
        self.assertFalse(up.any())
        self.assertFalse(down.any())


class DirectionalCommandTests(unittest.TestCase):
    def make_command(self, count, command_type=DirectionalVelocityCommand):
        command = command_type.__new__(command_type)
        command.device = "cpu"
        command.cfg = SimpleNamespace(ranges=SimpleNamespace(
            lin_vel_x=(-0.7, 0.7), lin_vel_y=(-0.45, 0.45), ang_vel_z=(-0.8, 0.8)
        ))
        command.vel_command_b = torch.zeros(count, 3)
        command.is_standing_env = torch.zeros(count, dtype=torch.bool)
        command.is_heading_env = torch.ones(count, dtype=torch.bool)
        return command

    def test_focus_retains_modes_and_both_directions(self):
        command = self.make_command(20000, FocusedStairVelocityCommand)
        command.terrain = SimpleNamespace(terrain_types=torch.arange(20000) % 10,
            cfg=SimpleNamespace(terrain_generator=SimpleNamespace(num_cols=10)))
        with torch.random.fork_rng():
            torch.manual_seed(2026)
            command._resample_command(torch.arange(20000))
        velocity = command.vel_command_b
        longitudinal = (velocity[:, 0] != 0) & (velocity[:, 1:] == 0).all(dim=1)
        self.assertAlmostEqual(longitudinal.float().mean().item(), 0.70, delta=0.015)
        for down, probability in ((False, 0.50), (True, 0.80)):
            mask = longitudinal & ((command.terrain.terrain_types >= 5) == down)
            self.assertAlmostEqual((velocity[mask, 0] < 0).float().mean().item(), probability, delta=0.03)
        self.assertAlmostEqual(command.is_standing_env.float().mean().item(), 0.05, delta=0.01)
        before = velocity.clone()
        command._resample_command(torch.tensor([], dtype=torch.long))
        self.assertTrue(torch.equal(before, velocity))

    def test_modes_bounds_and_sign_balance(self):
        command = self.make_command(20000)
        with torch.random.fork_rng():
            torch.manual_seed(2026)
            command._resample_command(torch.arange(20000))
        velocity = command.vel_command_b
        active = velocity != 0
        modes = [
            active[:, 0] & ~active[:, 1] & ~active[:, 2],
            ~active[:, 0] & active[:, 1] & ~active[:, 2],
            ~active[:, 0] & ~active[:, 1] & active[:, 2],
            active.all(dim=1), ~active.any(dim=1),
        ]
        for mask, proportion in zip(modes, [0.35, 0.30, 0.10, 0.15, 0.10]):
            self.assertAlmostEqual(mask.float().mean().item(), proportion, delta=0.015)
        self.assertTrue(torch.equal(command.is_standing_env, modes[-1]))
        self.assertFalse(command.is_heading_env.any())
        for axis, maximum in enumerate([0.7, 0.45, 0.8]):
            values = velocity[active[:, axis], axis]
            self.assertLessEqual(values.abs().max().item(), maximum + 1e-6)
            self.assertAlmostEqual((values > 0).float().mean().item(), 0.5, delta=0.03)
        for mask, axis, minimum in zip(modes[:3], [0, 1, 2], [0.25, 0.10, 0.25]):
            self.assertGreaterEqual(velocity[mask, axis].abs().min().item(), minimum)
        for axis, minimum in enumerate([0.20, 0.15, 0.20]):
            self.assertGreaterEqual(velocity[modes[3], axis].abs().min().item(), minimum)

    def test_stair_modes_prioritize_forward_and_backward_traversal(self):
        command = self.make_command(20000, StairVelocityCommand)
        command.cfg.ranges = SimpleNamespace(
            lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.5, 0.5)
        )
        with torch.random.fork_rng():
            torch.manual_seed(2026)
            command._resample_command(torch.arange(20000))
        velocity = command.vel_command_b
        active = velocity != 0
        longitudinal = active[:, 0] & ~active[:, 1] & ~active[:, 2]
        lateral = ~active[:, 0] & active[:, 1] & ~active[:, 2]
        yaw = ~active[:, 0] & ~active[:, 1] & active[:, 2]
        combined = active.all(dim=1)
        standing = ~active.any(dim=1)
        for mask, proportion in zip(
            [longitudinal, lateral, yaw, combined, standing],
            StairVelocityCommand.mode_fractions,
        ):
            self.assertAlmostEqual(mask.float().mean().item(), proportion, delta=0.015)
        self.assertAlmostEqual((velocity[longitudinal, 0] > 0).float().mean().item(), 0.5, delta=0.03)
        self.assertLessEqual(velocity[:, 0].abs().max().item(), 0.5 + 1e-6)
        self.assertLessEqual(velocity[:, 1].abs().max().item(), 0.2 + 1e-6)
        self.assertLessEqual(velocity[:, 2].abs().max().item(), 0.5 + 1e-6)

    def test_subset_and_empty_resampling_preserve_other_environments(self):
        command = self.make_command(5)
        command.vel_command_b[:] = 9.0
        command._resample_command(torch.tensor([1, 3]))
        self.assertTrue((command.vel_command_b[[0, 2, 4]] == 9.0).all())
        before = command.vel_command_b.clone()
        command._resample_command(torch.tensor([], dtype=torch.long))
        self.assertTrue(torch.equal(before, command.vel_command_b))


if __name__ == "__main__":
    unittest.main()
