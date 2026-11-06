"""Auditable raw events -> uniform pre-command samples. No implicit lag alignment."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .core import JOINT_NAMES, check_contract, digest, reorder, write_json


def save_dataset(directory, time, q, target, contract, *, velocity=None, **metadata):
    check_contract(contract)
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    t, q, target = np.asarray(time), np.asarray(q), np.asarray(target)
    dq = np.gradient(q, contract['physics_dt_s'], axis=0) if velocity is None else np.asarray(velocity)
    payload = dict(time=t, dof_pos=q, des_dof_pos=target, dof_vel=dq)
    validate_arrays(payload, contract)
    directory.mkdir(parents=True)
    np.savez_compressed(directory/'data.npz', **payload)
    write_json(directory/'metadata.json', dict(contract, **metadata, data_sha256=digest(directory/'data.npz')))


def validate_arrays(data, meta):
    check_contract(meta)
    t = data['time']
    if t.ndim != 1 or len(t) < 4 or not np.isfinite(t).all():
        raise ValueError('Need at least four finite, uniformly spaced samples')
    if not np.allclose(np.diff(t), meta['physics_dt_s'], rtol=1e-6, atol=1e-9):
        raise ValueError('Dataset dt does not match physical replay step')
    for name in ('dof_pos', 'des_dof_pos', 'dof_vel'):
        if data[name].shape != (len(t), 12) or not np.isfinite(data[name]).all():
            raise ValueError(f'Invalid {name}; expected finite [T,12]')


def load_dataset(directory):
    directory = Path(directory)
    meta = json.loads((directory/'metadata.json').read_text())
    if digest(directory/'data.npz') != meta['data_sha256']:
        raise ValueError('Dataset hash mismatch')
    with np.load(directory/'data.npz', allow_pickle=False) as source:
        data = {name: source[name].copy() for name in source.files}
    validate_arrays(data, meta)
    return data, meta


def prepare(raw_path, output, contract, max_gap_s=.03):
    """Raw q/dq and command times must use the same host monotonic clock."""
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError('max_gap_s must be positive')
    rows = [json.loads(line) for line in Path(raw_path).read_text().splitlines() if line.strip()]
    headers = [x for x in rows if x['kind'] == 'metadata']
    if len(headers) != 1:
        raise ValueError('Expected exactly one metadata record')
    header = headers[0]
    if header.get('source') == 'read_only':
        raise ValueError('Read-only feedback contains no known excitation; use it for data-quality inspection only')
    if not isinstance(header.get('session_id'), str) or not header['session_id']:
        raise ValueError('Raw metadata needs a stable session_id; preserve it across splits')
    if header.get('time_basis') != 'host_monotonic_ns':
        raise ValueError('Unknown clock basis; convert clocks explicitly before preparing')
    if header.get('capture_semantics') != 'sequence_bracketed_cache':
        raise ValueError('Unreviewed capture semantics')
    # SDK state timestamp is a cache identity, not a calibrated host time.
    # Host retrieval time preserves the observed lag, including polling uncertainty.
    frames = []
    duplicates = gaps = 0
    last = None
    for row in rows:
        if row['kind'] != 'feedback':
            continue
        seq = int(row['seq'])
        if last is not None:
            delta = (seq-last) % (2**32)
            if delta == 0:
                duplicates += 1
                continue
            if delta > 2**31:
                raise ValueError('Feedback sequence reset/out-of-order; split the session')
            # Ordinary SDK timestamp is milliseconds, not a packet counter.
            if header.get('sequence_kind') == 'packet_counter':
                gaps += delta-1
        last = seq
        frames.append(row)
    commands = [x for x in rows if x['kind'] == 'command']
    if len(frames) < 4 or not commands:
        raise ValueError('Need feedback and actual successful command-send events')
    if any(x.get('sent_ok') is not True for x in commands):
        raise ValueError('Failed command sends; split/review the session')
    ns = int(frames[0]['host_monotonic_ns'])
    ft = np.array([(int(x['host_monotonic_ns'])-ns)*1e-9 for x in frames])
    ct = np.array([(int(x['host_monotonic_ns'])-ns)*1e-9 for x in commands])
    if not np.isfinite(ft).all() or not np.isfinite(ct).all() or np.any(np.diff(ft) <= 0) or np.any(np.diff(ct) <= 0):
        raise ValueError('Raw event times must strictly increase in each stream')
    if np.diff(ft).max() > max_gap_s or (len(ct)>1 and np.diff(ct).max()>max_gap_s):
        raise ValueError('Feedback/command gap too long; do not interpolate across dropouts')
    for x in commands:
        if not np.array_equal(reorder(x['kp'], header['joint_names']), contract['kp_nm_rad']):
            raise ValueError('Kp changed or does not match replay')
        if not np.array_equal(reorder(x['kd'], header['joint_names']), contract['kd_nm_s_rad']):
            raise ValueError('Kd changed or does not match replay')
        if np.any(reorder(x['dq_des'], header['joint_names'])) or np.any(reorder(x['tau_ff'], header['joint_names'])):
            raise ValueError('This first version requires zero target velocity and feedforward')
    q = reorder([x['q'] for x in frames], header['joint_names'])
    dq = reorder([x['dq'] for x in frames], header['joint_names'])
    cmd = reorder([x['q_des'] for x in commands], header['joint_names'])
    dt = contract['physics_dt_s']
    start, end = max(ft[0], ct[0]), min(ft[-1], ct[-1])
    grid = start + np.arange(int(np.floor((end-start)/dt))+1)*dt
    if len(grid) < 4:
        raise ValueError('Not enough overlapping command and feedback time')
    qi = np.column_stack([np.interp(grid, ft, q[:, j]) for j in range(12)])
    dqi = np.column_stack([np.interp(grid, ft, dq[:, j]) for j in range(12)])
    # Commands are zero-order held by the drive, never linearly interpolated.
    targets = cmd[np.searchsorted(ct, grid, side='right')-1]
    save_dataset(output, grid-grid[0], qi, targets, contract, velocity=dqi,
                 source='hardware', source_session_id=header['session_id'], raw_sha256=digest(raw_path), raw_metadata=header,
                 time_basis='host_monotonic_ns', interpolation='q/dq linear; commands zero-order hold',
                 duplicate_frames=duplicates, missing_packets=gaps,
                 max_feedback_gap_s=float(np.diff(ft).max()),
                 delay_scope='effective delay includes cache polling and timestamp uncertainty',
                 calibrated=False, real_robot_ready=False)
