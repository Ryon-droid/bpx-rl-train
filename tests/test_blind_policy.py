"""History deployment and native RSL-RL distillation checks without Isaac Sim."""

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from policy_contract import POLICY_TERM_DIMS, ProprioHistory
from runner_utils import load_runner_checkpoint


class HistoryDeploymentTests(unittest.TestCase):
    def test_term_major_temporal_order_and_restart(self):
        history = ProprioHistory(length=3)
        first = np.arange(45, dtype=np.float32)
        frames = [first, first + 100, first + 200, first + 300]
        for frame in frames:
            result = history.append(frame)
        # Build the expected layout by term and frame explicitly. A naive
        # frames.reshape(-1) gives a different input and breaks deployment.
        expected = []
        offset = 0
        for dim in POLICY_TERM_DIMS:
            for frame in frames[-3:]:
                expected.extend(frame[offset:offset + dim])
            offset += dim
        np.testing.assert_array_equal(result, expected)
        history.reset()
        result = history.append(first)
        np.testing.assert_array_equal(result[:9], np.tile(first[:3], 3))
        self.assertFalse(np.any(result >= 100))

    def test_invalid_sensor_frames_do_not_enter_history(self):
        history = ProprioHistory()
        for frame in [np.zeros(48), np.full(45, np.nan), np.zeros((1, 45))]:
            with self.assertRaises(ValueError):
                history.append(frame)


@unittest.skipUnless(importlib.util.find_spec("rsl_rl"), "requires rsl-rl-lib==3.0.1")
class NativeDistillationTests(unittest.TestCase):
    def test_teacher_load_student_update_no_terrain_leak_and_checkpoint_reload(self):
        from rsl_rl.modules import ActorCritic, StudentTeacher
        from rsl_rl.algorithms import Distillation
        from tensordict import TensorDict

        torch.manual_seed(2026)
        obs = TensorDict({
            "policy": torch.randn(8, 45),
            "privileged": torch.randn(8, 3),
            "terrain": torch.randn(8, 187),
            "proprio_history": torch.randn(8, 900),
        }, batch_size=[8])
        teacher_groups = {"policy": ["policy", "privileged", "terrain"],
                          "critic": ["policy", "privileged", "terrain"]}
        teacher = ActorCritic(obs, teacher_groups, 12, actor_hidden_dims=[32, 16],
                              critic_hidden_dims=[32, 16], actor_obs_normalization=False,
                              critic_obs_normalization=False)
        student_groups = {"policy": ["proprio_history"],
                          "teacher": ["policy", "privileged", "terrain"]}
        kwargs = dict(student_hidden_dims=[32, 16], teacher_hidden_dims=[32, 16],
                      student_obs_normalization=False, teacher_obs_normalization=False)
        student = StudentTeacher(obs, student_groups, 12, **kwargs)
        self.assertFalse(student.load_state_dict(teacher.state_dict()))
        with torch.no_grad():
            torch.testing.assert_close(student.evaluate(obs), teacher.act_inference(obs))
            baseline = student.act_inference(obs)
            changed = obs.clone()
            changed["terrain"] += 100
            changed["privileged"] -= 100
            changed["policy"] *= -1
            torch.testing.assert_close(baseline, student.act_inference(changed))
        teacher_before = copy.deepcopy(student.teacher.state_dict())
        student_before = copy.deepcopy(student.student.state_dict())
        algorithm = Distillation(student, gradient_length=1, num_learning_epochs=1,
                                 learning_rate=1e-3, max_grad_norm=1.0)
        algorithm.init_storage("distillation", 8, 2, obs, [12])
        with torch.inference_mode():
            for _ in range(2):
                algorithm.act(obs)
                algorithm.process_env_step(obs, torch.ones(8), torch.zeros(8, dtype=torch.bool), {})
        loss = algorithm.update()["behavior"]
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(loss, 0)
        self.assertTrue(any(not torch.equal(student_before[k], v)
                            for k, v in student.student.state_dict().items()))
        for key, value in student.teacher.state_dict().items():
            torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)
        restored = StudentTeacher(obs, student_groups, 12, **kwargs)
        self.assertTrue(restored.load_state_dict(student.state_dict()))
        with torch.no_grad():
            torch.testing.assert_close(restored.act_inference(obs), student.act_inference(obs))

    def test_play_rejects_teacher_as_blind_checkpoint(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        runner = SimpleNamespace(alg=SimpleNamespace(policy=SimpleNamespace(student=object())), load=Mock())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "teacher.pt"
            torch.save({"model_state_dict": {"actor.0.weight": torch.ones(2, 3)}}, checkpoint)
            with self.assertRaisesRegex(ValueError, "trained student checkpoint"):
                load_runner_checkpoint(runner, str(checkpoint))
            runner.load.assert_not_called()
            load_runner_checkpoint(runner, str(checkpoint), teacher_initialization=True)
            runner.load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
