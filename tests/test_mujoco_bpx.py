"""Cross-simulator interface and pre-reset metric checks; no Isaac/GPU required."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mujoco_bpx import BPXSimulation, DEFAULT_JOINT_POS, JOINT_NAMES, PROJECT_ROOT, rpy_quaternion
from play_mujoco import PhaseMetrics


class MuJoCoBPXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = BPXSimulation()

    def setUp(self):
        self.sim.reset()

    def test_named_motor_mapping_and_urdf_inertias(self):
        sim = self.sim
        robot = ET.parse(PROJECT_ROOT / "resources/BPX/bpx/urdf/bpx.urdf").getroot()
        self.assertEqual((sim.model.nq, sim.model.nv, sim.model.nu), (19, 18, 12))
        for i, name in enumerate(JOINT_NAMES):
            joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.assertEqual(sim.model.actuator_trnid[i, 0], joint_id)
            self.assertEqual(sim.qpos_ids[i], sim.model.jnt_qposadr[joint_id])
        # The URDF traverses legs, whereas the actor groups joint types.
        self.assertFalse(np.array_equal(sim.qpos_ids, np.arange(7, 19)))
        for link in robot.findall("link"):
            body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, link.get("name"))
            inertial = link.find("inertial")
            self.assertAlmostEqual(sim.model.body_mass[body_id], float(inertial.find("mass").get("value")))
            raw = inertial.find("inertia")
            xx, yy, zz, xy, xz, yz = [float(raw.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
            expected = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            origin = inertial.find("origin")
            orientation = rpy_quaternion(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, orientation)
            expected = rotation.reshape(3, 3) @ expected @ rotation.reshape(3, 3).T
            mujoco.mju_quat2Mat(rotation, sim.model.body_iquat[body_id])
            actual = rotation.reshape(3, 3) @ np.diag(sim.model.body_inertia[body_id]) @ rotation.reshape(3, 3).T
            np.testing.assert_allclose(actual, expected, atol=5e-9)

    def test_body_frame_observation_and_com_velocity(self):
        sim = self.sim
        # 90-degree roll: world gravity becomes body -Y; body omega is freejoint omega.
        sim.data.qpos[3:7] = rpy_quaternion([np.pi / 2, 0, 0])
        sim.data.qvel[:3] = [1, 2, 3]
        sim.data.qvel[3:6] = [0.4, -0.2, 0.8]
        sim.data.qpos[sim.qpos_ids] += np.arange(12) * 0.01
        sim.data.qvel[sim.qvel_ids] = np.arange(12)
        mujoco.mj_forward(sim.model, sim.data)
        obs = sim.observation([0.7, -0.2, 0.8])
        self.assertEqual(obs.shape, (45,))
        np.testing.assert_allclose(obs[:3], [0.1, -0.05, 0.2], atol=1e-6)
        np.testing.assert_allclose(obs[3:6], [0, -1, 0], atol=1e-6)
        np.testing.assert_allclose(obs[6:9], [1.4, -0.4, 0.2], atol=1e-6)
        np.testing.assert_allclose(obs[9:21], np.arange(12) * 0.01, atol=1e-6)
        np.testing.assert_allclose(obs[21:33], np.arange(12) * 0.05, atol=1e-6)
        # Compare against independent rigid-body COM kinematics.
        rotation = sim.data.xmat[sim.base_id].reshape(3, 3)
        omega_world = rotation @ sim.data.qvel[3:6]
        offset_world = sim.data.xipos[sim.base_id] - sim.data.xpos[sim.base_id]
        expected_velocity = rotation.T @ (sim.data.qvel[:3] + np.cross(omega_world, offset_world))
        np.testing.assert_allclose(sim.body_velocity()[3:], expected_velocity, atol=1e-9)

    def test_action_clip_torque_limit_and_finite_integration(self):
        sim = self.sim
        torque = sim.step(np.full(12, 200.0))
        self.assertLessEqual(torque, 30)
        np.testing.assert_allclose(sim.target, DEFAULT_JOINT_POS + 25)
        np.testing.assert_allclose(sim.observation([0, 0, 0])[33:], 100)
        sim.reset()
        for _ in range(250):
            self.assertLessEqual(sim.step(np.zeros(12)), 30)
        self.assertAlmostEqual(sim.data.time, 5.0)
        self.assertTrue(np.isfinite(sim.data.qpos).all())
        self.assertTrue(np.isfinite(sim.data.qvel).all())
        before = sim.data.qvel[:2].copy()
        sim.push([0.3, -0.2])
        np.testing.assert_allclose(sim.data.qvel[:2], before + [0.3, -0.2])

    def test_terminal_metrics_survive_reset_and_warmup(self):
        sim = self.sim
        metrics = PhaseMetrics(1, "forward", [0.7, 0, 0], sim.control_dt, 0)
        sim.data.qvel[0] = 0.5
        sim.data.time = sim.control_dt
        mujoco.mj_forward(sim.model, sim.data)
        metrics.sample(sim, 30, "bad_orientation", True)
        sim.reset()
        row = metrics.row()
        self.assertAlmostEqual(row["rmse_vx_m_s"], 0.2)
        self.assertEqual(row["failures"], 1)
        self.assertEqual(row["timeouts"], 0)
        self.assertEqual(row["peak_pd_torque_nm"], 30)
        warmup = PhaseMetrics(1, "stand", [0, 0, 0], sim.control_dt, 2)
        warmup.sample(sim, 30, "base_contact", False)
        self.assertIsNone(warmup.row()["rmse_vx_m_s"])
        self.assertEqual(warmup.row()["failures"], 1)


if __name__ == "__main__":
    unittest.main()
