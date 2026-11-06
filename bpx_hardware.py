"""BPX output-joint specifications; independent of Isaac, Torch and MuJoCo.

Angle limits come from the selected URDF. Motor values come from the supplied
24 V BPX motor specification, pp. 2-3. Rated speed is used as a conservative
simulation operating cap, not claimed as measured no-load/maximum speed.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

PROFILE = 'bpx_24v_spec_20261106_v1'
JOINT_NAMES = tuple(f'{leg}_{kind}_joint' for kind in ('hip_roll', 'hip_pitch', 'knee')
                    for leg in ('fl', 'fr', 'hl', 'hr'))
MOTOR_SPECS = {
    'hip_roll': dict(series='HIPX', rated_torque_nm=10.8, peak_torque_nm=20.,
                     rated_speed_rad_s=12., peak_power_w_approx=200., peak_current_a=20., rated_voltage_v=24.),
    'hip_pitch': dict(series='HIPY', rated_torque_nm=10.8, peak_torque_nm=20.,
                      rated_speed_rad_s=12., peak_power_w_approx=200., peak_current_a=20., rated_voltage_v=24.),
    'knee': dict(series='KNEE', rated_torque_nm=15., peak_torque_nm=40.,
                 rated_speed_rad_s=10., peak_power_w_approx=230., peak_current_a=25., rated_voltage_v=24.),
}


def motor_spec(joint_name):
    if joint_name not in JOINT_NAMES:
        raise ValueError(f'Unknown BPX joint: {joint_name}')
    return MOTOR_SPECS[joint_name[3:-6]]


def joint_values(field, names=JOINT_NAMES):
    return tuple(motor_spec(name)[field] for name in names)


def actuator_limits(field):
    return {name: motor_spec(name)[field] for name in JOINT_NAMES}


def urdf_position_limits(urdf_path):
    joints = {j.get('name'): j for j in ET.parse(Path(urdf_path)).getroot().findall('joint')}
    limits = {}
    for name in JOINT_NAMES:
        limit = joints[name].find('limit')
        lo, hi = float(limit.get('lower')), float(limit.get('upper'))
        if not lo < hi:
            raise ValueError(f'Invalid URDF angle limits: {name}')
        limits[name] = (lo, hi)
    return limits


def hardware_contract(urdf_path):
    return dict(profile=PROFILE, joint_names=JOINT_NAMES,
                joint_position_limits_rad=urdf_position_limits(urdf_path),
                peak_torque_nm=joint_values('peak_torque_nm'),
                rated_torque_nm=joint_values('rated_torque_nm'),
                operating_speed_cap_rad_s=joint_values('rated_speed_rad_s'),
                speed_cap_basis='conservative use of rated speed; not a measured maximum',
                motor_specs=MOTOR_SPECS, measured_torque_speed_curve=False,
                thermal_or_peak_duration_model=False,
                power_current_model=False,
                note='Peak power/current recorded as specifications; electrical/mechanical power definition, '
                     'torque constant, efficiency and allowable peak duration are not supplied.')
