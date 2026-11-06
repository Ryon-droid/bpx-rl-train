"""Fixed-base PhysX replay. Import only AFTER AppLauncher has started Isaac Sim.

Uses absolute target PD with command/effort delay and the 49 PACE-style effective
parameters. A measured DC motor envelope is intentionally not invented.
"""
from __future__ import annotations
import importlib.metadata
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation
from .core import JOINT_NAMES, URDF, check_contract
from resources.mirrorme import BPX_CFG


class IsaacReplay:
    engine = 'physx'
    engine_version = dict(isaaclab=importlib.metadata.version('isaaclab'),
                          isaacsim=importlib.metadata.version('isaacsim'))

    def __init__(self, contract, population=1, device='cuda:0'):
        check_contract(contract)
        self.contract, self.population = contract, population
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=contract['physics_dt_s'], device=device))
        self.device = self.sim.device
        cfg = BPX_CFG.copy()
        cfg.prim_path = '/World/BPX_.*'
        cfg.spawn.asset_path = str(URDF)
        cfg.spawn.fix_base = True
        cfg.spawn.self_collision = False
        cfg.spawn.articulation_props.enabled_self_collisions = False
        cfg.spawn.joint_drive.gains.stiffness = 0.
        cfg.spawn.joint_drive.gains.damping = 0.
        cfg.spawn.activate_contact_sensors = False
        from scripts.mujoco_bpx import rpy_quaternion
        cfg.init_state.rot = tuple(rpy_quaternion(contract['base_rpy_rad']).tolist())
        cfg.init_state.pos = (0., 0., 1.2)
        # All PD is computed explicitly below. The actuator only clips feedforward;
        # no duplicate position drive remains in PhysX.
        from bpx_hardware import actuator_limits
        cfg.actuators = {'legs': IdealPDActuatorCfg(
            joint_names_expr=list(JOINT_NAMES), stiffness=0., damping=0., armature=.003,
            effort_limit=actuator_limits('peak_torque_nm'),
            effort_limit_sim=actuator_limits('peak_torque_nm'),
            velocity_limit_sim=actuator_limits('rated_speed_rad_s'))}
        from isaacsim.core.utils.prims import create_prim
        for i in range(population):
            create_prim(f'/World/BPX_{i}', 'Xform', translation=(i*2.5, 0., 0.))
        # Anchor articulation to child prims, avoiding robot roots named like envs.
        cfg.prim_path = '/World/BPX_.*/Robot'
        self.robot = Articulation(cfg)
        self.sim.reset()
        self.robot.update(contract['physics_dt_s'])
        self.ids = [self.robot.joint_names.index(n) for n in JOINT_NAMES]
        if not self.robot.is_fixed_base or self.robot.num_joints != 12 or self.robot.num_instances != population:
            raise ValueError('Expected fixed-base 12-joint BPX articulations')
        for field in ('joint_stiffness', 'joint_damping'):
            actual = getattr(self.robot.data, field)[:, self.ids]
            torch.testing.assert_close(actual, torch.zeros_like(actual))
        self.kp = torch.tensor(contract['kp_nm_rad'], device=self.device)
        self.kd = torch.tensor(contract['kd_nm_s_rad'], device=self.device)
        self.effort = torch.tensor(contract['peak_torque_nm'], device=self.device)
        self.speed = torch.tensor(contract['operating_speed_cap_rad_s'], device=self.device)
        self.env_ids = torch.arange(population, device=self.device)

    def evaluate(self, params, data, skip=0, trajectories=False):
        count = len(params)
        if count not in (1, self.population):
            raise ValueError('PhysX candidate count must be 1 or equal to population')
        p = torch.as_tensor(np.asarray(params), dtype=torch.float32, device=self.device)
        if count == 1:
            p = p.repeat(self.population, 1)
        robot, ids = self.robot, self.ids
        e = self.env_ids
        robot.reset()
        robot.write_joint_armature_to_sim(p[:, :12], joint_ids=ids, env_ids=e)
        robot.write_joint_viscous_friction_coefficient_to_sim(p[:, 12:24], joint_ids=ids, env_ids=e)
        # Set dynamic to zero before decreasing static friction.
        robot.write_joint_dynamic_friction_coefficient_to_sim(torch.zeros_like(p[:, 24:36]), joint_ids=ids, env_ids=e)
        robot.write_joint_friction_coefficient_to_sim(p[:, 24:36], joint_ids=ids, env_ids=e)
        robot.write_joint_dynamic_friction_coefficient_to_sim(p[:, 24:36], joint_ids=ids, env_ids=e)
        # Read back the properties actually consumed by the solver.
        for attr, values in [('joint_armature', p[:, :12]),
                             ('joint_viscous_friction_coeff', p[:, 12:24]),
                             ('joint_friction_coeff', p[:, 24:36]),
                             ('joint_dynamic_friction_coeff', p[:, 24:36])]:
            actual = getattr(robot.data, attr)[:, ids]
            torch.testing.assert_close(actual, values)
        # Also read PhysX, rather than checking only Isaac's cached tensors.
        actual_armature = robot.root_physx_view.get_dof_armatures()[:, ids].to(self.device)
        torch.testing.assert_close(actual_armature, p[:, :12])
        friction = robot.root_physx_view.get_dof_friction_properties()[:, ids].to(self.device)
        for component, expected in enumerate((p[:, 24:36], p[:, 24:36], p[:, 12:24])):
            torch.testing.assert_close(friction[:, :, component], expected)
        bias = p[:, 36:48]
        q0 = torch.as_tensor(data['dof_pos'][0], dtype=torch.float32, device=self.device)+bias
        dq0 = torch.as_tensor(data['dof_vel'][0], dtype=torch.float32, device=self.device).repeat(self.population, 1)
        limits = robot.data.joint_pos_limits[:, ids]
        invalid = ((q0 < limits[:, :, 0]) | (q0 > limits[:, :, 1])).any(dim=1)
        q0 = q0.clamp(limits[:, :, 0], limits[:, :, 1])
        robot.write_joint_state_to_sim(q0, dq0, joint_ids=ids, env_ids=e)
        # Invalidate cached state after direct writes, without stepping time.
        robot.update(0.)
        lag = p[:, 48].long()
        capacity = int(lag.max().item())+1
        history = torch.empty((capacity, self.population, 12), device=self.device)
        targets = torch.as_tensor(data['des_dof_pos'], dtype=torch.float32, device=self.device)
        measured = torch.as_tensor(data['dof_pos'], dtype=torch.float32, device=self.device)
        scores = torch.zeros(self.population, dtype=torch.float64, device=self.device)
        out = []
        for i, target in enumerate(targets):
            q, dq = robot.data.joint_pos[:, ids], robot.data.joint_vel[:, ids]
            encoder = q-bias
            if trajectories:
                out.append(encoder[:count].cpu().numpy())
            finite = torch.isfinite(encoder).all(dim=1) & torch.isfinite(dq).all(dim=1)
            invalid |= ~finite
            if i >= skip:
                error = torch.nan_to_num(encoder-measured[i], nan=1e6, posinf=1e6, neginf=-1e6)
                scores += (error.double()**2).mean(dim=1)
            signal = target.expand(self.population, -1) if self.contract['delay_mode'] == 'command' else self.kp*(target-encoder)-self.kd*dq
            history[i % capacity] = signal
            # First signal is the prehistory, as in the CPU replay.
            delayed = history[(torch.clamp(i-lag, min=0) % capacity), e]
            tau = self.kp*(delayed-encoder)-self.kd*dq if self.contract['delay_mode'] == 'command' else delayed
            tau = tau.clamp(-self.effort, self.effort)
            tau = torch.where((dq.abs() >= self.speed) & (tau*dq > 0), 0., tau)
            tau = torch.where(invalid[:, None], 0., tau)
            robot.set_joint_effort_target(tau, joint_ids=ids)
            robot.write_data_to_sim()
            self.sim.step(render=False)
            robot.update(self.contract['physics_dt_s'])
        scores /= len(targets)-skip
        scores[invalid] = 1e12
        predictions = np.stack(out, axis=1) if trajectories else None
        return scores[:count].cpu().numpy(), predictions

    def close(self):
        # AppLauncher owns the process-wide SimulationContext lifecycle.
        # Stopping this direct GPU articulation before Kit teardown can block
        # on Isaac Sim 5.1; the launcher closes the app after output is flushed.
        pass
