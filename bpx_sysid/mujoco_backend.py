"""Fixed-base CPU replay; PACE-style parameters, MuJoCo physics (not PhysX)."""
from __future__ import annotations
from collections import deque
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scripts.mujoco_bpx import scene_xml, rpy_quaternion, numbers
from .core import JOINT_NAMES, URDF, check_contract


class MuJoCoReplay:
    engine = 'mujoco'
    engine_version = mujoco.__version__

    def __init__(self, contract, population=1):
        check_contract(contract)
        self.contract = contract
        self.population = population
        xml = ET.fromstring(scene_xml(URDF))
        torso = xml.find("worldbody/body[@name='torso']")
        torso.remove(torso.find('freejoint'))
        torso.set('pos', '0 0 1.2')
        torso.set('quat', numbers(rpy_quaternion(contract['base_rpy_rad'])))
        # No ground or self contacts: this is an in-air identification experiment.
        for geom in xml.iter('geom'):
            geom.set('contype', '0')
            geom.set('conaffinity', '0')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
        self.model.opt.timestep = contract['physics_dt_s']
        self.data = mujoco.MjData(self.model)
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        self.qids = self.model.jnt_qposadr[ids]
        self.vids = self.model.jnt_dofadr[ids]
        self.ranges = self.model.jnt_range[ids]
        if (self.model.nq, self.model.nv, self.model.nu) != (12, 12, 12):
            raise ValueError('Expected a fixed base and exactly 12 motors')
        self.kp, self.kd = np.array(contract['kp_nm_rad']), np.array(contract['kd_nm_s_rad'])
        self.effort = np.array(contract['peak_torque_nm'])
        self.speed = np.array(contract['operating_speed_cap_rad_s'])

    def replay(self, p, data):
        p = np.asarray(p, dtype=float)
        bias = p[36:48]
        lag = int(p[48])
        if np.any(data['dof_pos'][0]+bias < self.ranges[:, 0]) or np.any(data['dof_pos'][0]+bias > self.ranges[:, 1]):
            raise ValueError('Initial biased coordinates exceed physical joint limits')
        self.model.dof_armature[self.vids] = p[:12]
        self.model.dof_damping[self.vids] = p[12:24]
        self.model.dof_frictionloss[self.vids] = p[24:36]
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qids] = data['dof_pos'][0]+bias
        self.data.qvel[self.vids] = data['dof_vel'][0]
        mujoco.mj_forward(self.model, self.data)
        history = deque(maxlen=lag+1)
        out = np.empty_like(data['dof_pos'], dtype=float)
        for i, target in enumerate(data['des_dof_pos']):
            q, dq = self.data.qpos[self.qids].copy(), self.data.qvel[self.vids].copy()
            out[i] = q-bias
            if self.contract['delay_mode'] == 'command':
                history.append(target.copy())
                delayed = history[0]
                tau = self.kp*(delayed-(q-bias))-self.kd*dq
            else:
                history.append(self.kp*(target-(q-bias))-self.kd*dq)
                tau = history[0]
            tau = np.clip(tau, -self.effort, self.effort)
            tau = np.where((np.abs(dq) >= self.speed) & (tau*dq > 0), 0., tau)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
            bad = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                   mujoco.mjtWarning.mjWARN_BADQACC)
            if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                    or any(self.data.warning[w].number for w in bad)):
                raise ValueError('Diverged candidate')
        return out

    def evaluate(self, params, data, skip=0, trajectories=False):
        scores, all_q = [], []
        for p in params:
            try:
                q = self.replay(p, data)
                score = float(np.mean(np.square(q[skip:]-data['dof_pos'][skip:])))
                if not np.isfinite(score):
                    raise ValueError('Non-finite loss')
            except ValueError:
                q, score = np.full_like(data['dof_pos'], np.nan), 1e12
            scores.append(score)
            if trajectories:
                all_q.append(q)
        return np.array(scores), np.array(all_q) if trajectories else None

    def close(self):
        pass
