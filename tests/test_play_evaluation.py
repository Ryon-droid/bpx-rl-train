"""Check evaluation math and reset/settling accounting without launching Isaac Sim."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.play_evaluation import PlayEvaluator, record_evaluation_metrics


class EvaluationTests(unittest.TestCase):
    def make_env(self):
        data = SimpleNamespace(
            root_lin_vel_b=torch.tensor([[0.4, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            root_ang_vel_b=torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 0.0]]),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
            applied_torque=torch.tensor([[30.0, 0.0], [0.0, 0.0]]),
            joint_effort_limits=torch.full((2, 2), 30.0),
            body_lin_vel_w=torch.tensor([[[0.2, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]),
        )
        robot = SimpleNamespace(data=data, find_joints=lambda *a, **k: ([0, 1], ["j0", "j1"]),
                                find_bodies=lambda *a, **k: ([0], ["foot"]))
        sensor = SimpleNamespace(data=SimpleNamespace(
            net_forces_w_history=torch.tensor([[[[0.0, 0.0, 2.0]]], [[[0.0, 0.0, 2.0]]]])
        ), find_bodies=lambda *a, **k: ([0], ["foot"]))

        class Scene(dict):
            sensors = {"contact_forces": sensor}

        reasons = {"base_contact": torch.tensor([True, False]), "bad_orientation": torch.tensor([False, False])}
        return SimpleNamespace(
            device="cpu", num_envs=2, step_dt=0.02, scene=Scene(robot=robot),
            episode_length_buf=torch.tensor([1, 1]),
            command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([[0.3, 0.0, 0.0]] * 2)),
            action_manager=SimpleNamespace(action=torch.ones(2, 2), prev_action=torch.zeros(2, 2)),
            termination_manager=SimpleNamespace(terminated=torch.tensor([True, False]),
                time_outs=torch.tensor([False, True]), get_term=lambda name: reasons[name]),
        )

    def test_pre_reset_errors_and_failure_accounting(self):
        env = self.make_env()
        evaluator = PlayEvaluator(env, ["j0", "j1"], ["foot"], warmup_s=0)
        env.bpx_evaluator = evaluator
        evaluator.begin_phase(1, "forward", (0.3, 0.0, 0.0))
        result = record_evaluation_metrics(env)
        self.assertFalse(result.any())
        # Simulate Isaac Lab resetting state after the measurement callback.
        env.scene["robot"].data.root_lin_vel_b.zero_()
        env.episode_length_buf.zero_()
        env.termination_manager.terminated.zero_()
        row = evaluator.finish_phase()
        self.assertAlmostEqual(row["rmse_vx_m_s"], 0.1, places=6)
        self.assertAlmostEqual(row["rmse_wz_rad_s"], math.sqrt(0.02), places=6)
        self.assertAlmostEqual(row["mean_contact_slide_m_s"], 0.1, places=6)
        self.assertAlmostEqual(row["rms_estimated_torque_nm"], 15.0)
        self.assertAlmostEqual(row["estimated_torque_saturation_pct"], 25.0)
        self.assertEqual(row["failures"], 1)
        self.assertEqual(row["timeouts"], 1)
        self.assertAlmostEqual(row["mean_completed_episode_s"], 0.02)
        self.assertEqual(row["peak_estimated_torque_nm"], 30.0)
        with tempfile.TemporaryDirectory() as directory:
            evaluator.save(directory, {"seed": 2026}, completed=False)
            report = json.loads((Path(directory) / "summary.json").read_text())
            self.assertFalse(report["completed_schedule"])
            self.assertEqual(report["summary"]["failure_pct_of_completed_episodes"], 50.0)
            self.assertEqual(len((Path(directory) / "phases.csv").read_text().splitlines()), 2)

    def test_warmup_excludes_tracking_but_keeps_failures(self):
        env = self.make_env()
        evaluator = PlayEvaluator(env, ["j0", "j1"], ["foot"], warmup_s=0.02)
        evaluator.begin_phase(1, "stand", (0.0, 0.0, 0.0))
        evaluator.sample(env)
        row = evaluator.finish_phase()
        self.assertIsNone(row["rmse_vx_m_s"])
        self.assertEqual(row["measured_robot_seconds"], 0)
        self.assertEqual(row["failures"], 1)
        self.assertEqual(row["peak_estimated_torque_nm"], 30.0)

    def test_post_reset_warmup_and_timeout_failure_overlap(self):
        env = self.make_env()
        env.termination_manager.time_outs[:] = True
        evaluator = PlayEvaluator(env, ["j0", "j1"], ["foot"], warmup_s=0.02)
        evaluator.begin_phase(1, "forward", (0.3, 0.0, 0.0))
        evaluator.sample(env)
        # Only the second robot has finished settling; first has just reset.
        env.episode_length_buf[:] = torch.tensor([1, 2])
        env.termination_manager.terminated[:] = False
        env.termination_manager.time_outs[:] = False
        evaluator.sample(env)
        row = evaluator.finish_phase()
        self.assertAlmostEqual(row["measured_robot_seconds"], 0.02)
        self.assertEqual(row["failures"], 1)
        self.assertEqual(row["timeouts"], 1)


if __name__ == "__main__":
    unittest.main()
