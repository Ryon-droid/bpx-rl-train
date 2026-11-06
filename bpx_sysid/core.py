"""Named coordinates, trajectories, and versioned identification parameters."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from bpx_hardware import JOINT_NAMES, PROFILE, joint_values, urdf_position_limits

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / 'resources/BPX/bpx/urdf/bpx.urdf'
SDK_NAMES = tuple(f'{leg}_{kind}_joint' for leg in ('fl', 'fr', 'hl', 'hr')
                  for kind in ('hip_roll', 'hip_pitch', 'knee'))
DEFAULT_POSITION = np.array([0.] * 4 + [.8] * 4 + [-1.5] * 4)
PARAMETER_FIELDS = ('armature_kg_m2', 'viscous_nm_s_rad', 'coulomb_nm', 'encoder_bias_rad')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reorder(values, source_names, target_names=JOINT_NAMES):
    if len(source_names) != 12 or set(source_names) != set(JOINT_NAMES):
        raise ValueError('Expected each of the 12 BPX joint names exactly once')
    a = np.asarray(values, dtype=float)
    if a.shape[-1] != 12 or not np.isfinite(a).all():
        raise ValueError('Joint data must end with 12 finite values')
    return a[..., [source_names.index(name) for name in target_names]]


def vector(value, name, minimum=None):
    a = np.broadcast_to(np.asarray(value, dtype=float), (12,)).copy()
    if not np.isfinite(a).all() or (minimum is not None and np.any(a < minimum)):
        raise ValueError(f'Invalid {name}')
    return a


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def model_contract(dt=.005, kp=30., kd=1., delay_mode='command'):
    if not np.isfinite(dt) or not .0001 <= dt <= .02:
        raise ValueError('physics_dt_s must be between 0.0001 and 0.02 seconds')
    if delay_mode not in ('command', 'effort'):
        raise ValueError('Unknown delay mode')
    return dict(schema_version=1, joint_names=list(JOINT_NAMES), hardware_profile=PROFILE,
                urdf_sha256=digest(URDF), physics_dt_s=float(dt),
                kp_nm_rad=vector(kp, 'kp', 0).tolist(), kd_nm_s_rad=vector(kd, 'kd', 0).tolist(),
                delay_mode=delay_mode, base_rpy_rad=[0., 0., 0.],
                bias_definition='q_encoder = q_physics - encoder_bias_rad',
                sampling_phase='state_before_command',
                peak_torque_nm=list(joint_values('peak_torque_nm')),
                operating_speed_cap_rad_s=list(joint_values('rated_speed_rad_s')),
                torque_speed_curve_identified=False)


def check_contract(c):
    if c['schema_version'] != 1 or c['joint_names'] != list(JOINT_NAMES):
        raise ValueError('Unsupported schema or joint order')
    if c['hardware_profile'] != PROFILE or c['urdf_sha256'] != digest(URDF):
        raise ValueError('Hardware/URDF mismatch; explicitly regenerate or migrate the experiment')
    model_contract(c['physics_dt_s'], c['kp_nm_rad'], c['kd_nm_s_rad'], c['delay_mode'])
    if c.get('bias_definition') != 'q_encoder = q_physics - encoder_bias_rad':
        raise ValueError('Unknown encoder bias sign convention')
    if c.get('sampling_phase') != 'state_before_command':
        raise ValueError('Unsupported sampling phase')
    for field in ('peak_torque_nm', 'operating_speed_cap_rad_s'):
        expected = joint_values('peak_torque_nm' if field == 'peak_torque_nm' else 'rated_speed_rad_s')
        if not np.array_equal(c[field], expected):
            raise ValueError(f'{field} differs from current hardware profile')
    rpy = np.asarray(c.get('base_rpy_rad'), dtype=float)
    if rpy.shape != (3,) or not np.isfinite(rpy).all():
        raise ValueError('Expected finite base_rpy_rad')


def parameter_vector(p):
    check_contract(p)
    parts = [vector(p[k], k, 0 if k != 'encoder_bias_rad' else None) for k in PARAMETER_FIELDS]
    delay = p['delay_steps']
    if not isinstance(delay, int) or not 0 <= delay <= 1000:
        raise ValueError('delay_steps must be an integer in [0,1000]')
    if not np.isclose(p['delay_s'], delay * p['physics_dt_s'], rtol=0, atol=1e-12):
        raise ValueError('Delay seconds and physics steps disagree')
    return np.concatenate([*parts, [delay]])


def parameters(values, contract, **provenance):
    a = np.asarray(values, dtype=float)
    if a.shape != (49,) or not np.isfinite(a).all():
        raise ValueError('Expected 49 finite parameters')
    if np.any(a[:36] < 0) or not 0 <= a[-1] <= 1000:
        raise ValueError('Invalid nonnegative physical parameters / delay')
    out = dict(contract)
    out.update({k: a[i*12:(i+1)*12].tolist() for i, k in enumerate(PARAMETER_FIELDS)})
    out.update(delay_steps=int(a[-1]), delay_s=int(a[-1])*contract['physics_dt_s'],
               real_robot_ready=False, **provenance)
    parameter_vector(out)
    return out


def trajectory(dt=.005, duration=20., amplitude=.05, f0=.1, f1=1., joint='fl_hip_pitch_joint',
               ramp=2., hold=2., max_speed=.5, max_acceleration=5.):
    numbers = np.array([dt, duration, amplitude, f0, f1, ramp, hold, max_speed, max_acceleration])
    if (not np.isfinite(numbers).all() or not .0001 <= dt <= .02 or duration < 4*dt
            or amplitude <= 0 or f0 <= 0 or f1 < f0 or ramp <= 0 or hold < dt
            or max_speed <= 0 or max_acceleration <= 0 or 2*ramp > duration or f1*dt > .1):
        raise ValueError('Invalid chirp timing, amplitude or limits')
    if joint not in JOINT_NAMES and joint != 'all':
        raise ValueError('Unknown joint name')
    count = int(round(duration / dt))
    if not np.isclose(count*dt, duration):
        raise ValueError('duration must be a multiple of dt')
    t_chirp = np.arange(count+1)*dt
    # Quintic envelope: continuous position, velocity and acceleration at endpoints.
    s = np.minimum(np.minimum(t_chirp, duration-t_chirp)/ramp, 1.).clip(0, 1)
    env = s**3*(10-15*s+6*s*s)
    phase = 2*np.pi*(f0*t_chirp+(f1-f0)*t_chirp**2/(2*duration))
    q = np.tile(DEFAULT_POSITION, (count+1, 1))
    active = range(12) if joint == 'all' else [JOINT_NAMES.index(joint)]
    for j in active:
        q[:, j] += amplitude*env*np.sin(phase + (j*np.pi/6 if joint == 'all' else 0.))
    nh = int(np.ceil(hold/dt))
    q = np.concatenate([np.tile(DEFAULT_POSITION, (nh, 1)), q, np.tile(DEFAULT_POSITION, (nh, 1))])
    time = np.arange(len(q))*dt
    dq = np.gradient(q, dt, axis=0, edge_order=2)
    ddq = np.gradient(dq, dt, axis=0, edge_order=2)
    limits = np.array(list(urdf_position_limits(URDF).values()))
    if np.any(q < limits[:, 0]+.1) or np.any(q > limits[:, 1]-.1):
        raise ValueError('Trajectory violates URDF angle limits with 0.1 rad margin')
    if np.abs(dq).max() > max_speed or np.abs(ddq).max() > max_acceleration:
        raise ValueError('Chirp exceeds requested velocity/acceleration limit; reduce amplitude/frequency')
    return time, q, dict(peak_speed_rad_s=float(np.abs(dq).max()),
                        peak_acceleration_rad_s2=float(np.abs(ddq).max()),
                        collision_checked=False, hardware_motion_approved=False,
                        joint=joint, amplitude_rad=amplitude, f0_hz=f0, f1_hz=f1)


def software_provenance():
    paths = sorted((ROOT / "bpx_sysid").glob("*.py"))
    paths += [ROOT / "scripts/sysid.py", ROOT / "scripts/sysid_isaac.py"]
    return dict(python_version=sys.version.split()[0],
                source_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths})
