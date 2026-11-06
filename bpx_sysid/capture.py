"""State-only SDK logger. Never constructs JointLevelControl or sends motor commands."""
from __future__ import annotations
import json
from pathlib import Path
import time
import math
import uuid
from .core import SDK_NAMES, digest, URDF


def snapshot(state):
    """Bracket cached joint reads; diagnostics have independent freshness."""
    before = state.getJointStateTimestamp()
    q = state.getJointPosition()
    dq = state.getJointVelocity()
    tau = state.getJointTorque()
    after = state.getJointStateTimestamp()
    if before is None or before != after or any(x is None for x in (q, dq, tau)):
        return None
    if any(len(x) != 12 or not all(math.isfinite(v) for v in x) for x in (q, dq, tau)):
        return None
    return dict(kind='feedback', seq=int(after), device_timestamp_raw=int(after),
                host_monotonic_ns=time.monotonic_ns(), q=q, dq=dq, tau=tau)


def collect(ip, output, seconds=20., rate=100., sdk=None):
    if not math.isfinite(seconds) or not 0 < seconds <= 600 or not math.isfinite(rate) or not 1 <= rate <= 200:
        raise ValueError('Use duration (0,600] seconds and poll rate [1,200] Hz')
    if sdk is None:
        import bpx_sdk as sdk
    # RequestRobotState uses the state subscription; JointLevelControl can switch
    # host-server mode even without setJointCommand, so is deliberately not used.
    state = sdk.RequestRobotState()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = repeats = incoherent = 0
    last = None
    start = time.monotonic()
    with path.open('x', encoding='utf-8', buffering=1) as f:
        def emit(value):
            f.write(json.dumps(value, allow_nan=False)+'\n')
        emit(dict(kind='metadata', source='read_only', schema_version=1, session_id=uuid.uuid4().hex,
                  joint_names=list(SDK_NAMES), time_basis='host_monotonic_ns',
                  sequence_kind='device_timestamp_ms', capture_semantics='sequence_bracketed_cache',
                  sdk_version=getattr(sdk, '__version__', 'unknown'), ip=ip,
                  urdf_sha256=digest(URDF), requested_poll_hz=rate,
                  note='State-only diagnostics; no target commands, not suitable for fitting. '
                       'Cache bracketing is best effort, not an atomic packet capture.'))
        try:
            state.setRobotIp(ip)
            state.setRobotStateUploadRate(int(rate))
            if not state.connect():
                raise ConnectionError('SDK state connection failed')
            start = time.monotonic()
            deadline = start
            next_diagnostics = start + .1
            while time.monotonic()-start < seconds:
                row = snapshot(state)
                if row is None:
                    incoherent += 1
                elif row['seq'] == last:
                    repeats += 1
                else:
                    emit(row)
                    last = row['seq']
                    frames += 1
                if time.monotonic() >= next_diagnostics:
                    def optional(name):
                        v = getattr(state, name)()
                        if isinstance(v, (int, float)):
                            return v if math.isfinite(v) else None
                        if isinstance(v, (list, tuple)) and all(isinstance(x,(int,float)) and math.isfinite(x) for x in v):
                            return list(v)
                        return None
                    emit(dict(kind='diagnostics', host_monotonic_ns=time.monotonic_ns(),
                         imu_timestamp_ms=optional('getImuTimestamp'),
                         imu_rpy_rad=optional('getImuRpy'), imu_omega_rad_s=optional('getImuOmega'),
                         motor_temperature=optional('getMotorTemperature'),
                         driver_temperature=optional('getDriverTemperature'),
                         battery_timestamp_ms=optional('getBatteryTimestamp'),
                         battery_current_raw=optional('getBatteryCurrent'),
                         firmware_version=optional('getRobotVersion'),
                         freshness_note='Separate cache groups; not atomic with joint feedback'))
                    next_diagnostics = time.monotonic()+.1
                deadline += 1/rate
                time.sleep(max(0., deadline-time.monotonic()))
                if time.monotonic()-deadline > 1/rate:
                    deadline = time.monotonic()
        finally:
            state.disconnect()
            elapsed = max(time.monotonic()-start, 1e-9)
            summary = dict(kind='summary', elapsed_s=elapsed, unique_frames=frames,
                           duplicate_polls=repeats, incoherent_polls=incoherent,
                           observed_unique_hz=frames/elapsed, commands_sent=0)
            emit(summary)
    return summary
