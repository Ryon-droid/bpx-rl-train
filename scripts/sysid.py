#!/usr/bin/env python3
"""BPX identification preparation; offline by default, explicit state-only SDK capture."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import uuid
import numpy as np
from project_paths import add_project_source
add_project_source()
from bpx_sysid.core import (ROOT, JOINT_NAMES, PARAMETER_FIELDS, model_contract,
                            parameters, parameter_vector, trajectory, write_json, software_provenance)
from bpx_sysid.data import load_dataset, prepare, save_dataset


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='Generate an OFFLINE absolute-angle chirp; sends no commands')
    plan.add_argument('--output', required=True)
    plan.add_argument('--joint', default='fl_hip_pitch_joint', choices=[*JOINT_NAMES, 'all'])
    plan.add_argument('--dt', type=float, default=.005)
    plan.add_argument('--duration', type=float, default=20.)
    plan.add_argument('--amplitude', type=float, default=.05)
    plan.add_argument('--f0', type=float, default=.1)
    plan.add_argument('--f1', type=float, default=1.)
    plan.add_argument('--max-speed', type=float, default=.5)
    plan.add_argument('--max-acceleration', type=float, default=5.)
    synthetic = sub.add_parser('synthetic', help='Generate known-parameter, fixed-base MuJoCo data')
    synthetic.add_argument('--output', required=True)
    synthetic.add_argument('--duration', type=float, default=4.)
    synthetic.add_argument('--dt', type=float, default=.005)
    synthetic.add_argument('--delay-mode', choices=['command', 'effort'], default='command')
    collect = sub.add_parser('collect-readonly', help='Use RequestRobotState only; sends no joint commands')
    collect.add_argument('--ip', required=True)
    collect.add_argument('--output', required=True)
    collect.add_argument('--seconds', type=float, default=20.)
    collect.add_argument('--rate', type=float, default=100.)
    prep = sub.add_parser('prepare', help='Convert externally collected feedback+actual command JSONL')
    prep.add_argument('--raw', required=True)
    prep.add_argument('--contract', required=True)
    prep.add_argument('--output', required=True)
    prep.add_argument('--max-gap', type=float, default=.03)
    inspect = sub.add_parser('inspect', help='Inspect a prepared dataset without simulation')
    inspect.add_argument('--data', required=True)
    fit = sub.add_parser('fit', help='CPU MuJoCo CMA-ES (not the paper PhysX engine)')
    fit.add_argument('--data', required=True)
    fit.add_argument('--bounds', required=True)
    fit.add_argument('--output', required=True)
    fit.add_argument('--population', type=int, default=32)
    fit.add_argument('--generations', type=int, default=10)
    fit.add_argument('--seed', type=int, default=2026)
    fit.add_argument('--warmup', type=float, default=.5)
    replay = sub.add_parser('replay', help='Independent MuJoCo replay of an exported parameter file')
    replay.add_argument('--data', required=True)
    replay.add_argument('--parameters', required=True)
    replay.add_argument('--output', required=True)
    replay.add_argument('--warmup', type=float, default=.5)
    return p


def replay_report(backend, data, meta, path, output, warmup):
    from bpx_sysid.optimize import metrics
    p = json.loads(Path(path).read_text())
    values = parameter_vector(p)
    # Reject changing controller/timing/model while claiming a matched replay.
    for key in model_contract():
        if p[key] != meta[key]:
            raise ValueError(f'Replay contract differs: {key}')
    skip = int(round(warmup/meta['physics_dt_s']))
    if not np.isfinite(warmup) or warmup < 0 or skip >= len(data['time'])-1:
        raise ValueError('Invalid warmup interval')
    _, q = backend.evaluate([values], data, skip, trajectories=True)
    source = p.get('training_data_sha256')
    train_session, eval_session = p.get('training_session_id'), meta.get('source_session_id')
    result = dict(metrics(q[0], data, skip), engine=backend.engine,
                  same_as_fitting_data=source == meta['data_sha256'] if source else None,
                  different_data_file=bool(source and source != meta['data_sha256']),
                  independent_session=bool(train_session and eval_session and train_session != eval_session
                                           and source and source != meta['data_sha256']),
                  engine_version=backend.engine_version, software=software_provenance(),
                  training_engine=p.get('fitting_engine'), evaluation_engine=backend.engine,
                  # A different data hash is provenance evidence, not a pass criterion.
                  calibrated=False, real_robot_ready=False)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    write_json(target/'report.json', result)
    np.save(target/'predicted_encoder_pos.npy', q[0])
    return result


def main():
    args = parser().parse_args()
    if args.command == 'collect-readonly':
        from bpx_sysid.capture import collect
        summary = collect(args.ip, args.output, args.seconds, args.rate)
        print(json.dumps(summary, indent=2))
        if summary['unique_frames'] == 0:
            raise SystemExit('No joint feedback arrived; inspect connection and firmware before continuing')
        return
    if args.command == 'plan':
        t, q, checks = trajectory(dt=args.dt, duration=args.duration, amplitude=args.amplitude,
                f0=args.f0, f1=args.f1, joint=args.joint, max_speed=args.max_speed,
                max_acceleration=args.max_acceleration)
        dest = Path(args.output)
        dest.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(dest/'trajectory.npz', time=t, des_dof_pos=q)
        np.savetxt(dest/'trajectory.csv', np.column_stack([t, q]), delimiter=',',
                   header=','.join(['time_s', *JOINT_NAMES]), comments='')
        write_json(dest/'contract.json', model_contract(args.dt))
        write_json(dest/'checks.json', checks)
        print(json.dumps(checks, indent=2))
        return
    if args.command == 'prepare':
        prepare(args.raw, args.output, json.loads(Path(args.contract).read_text()), args.max_gap)
        return
    if args.command == 'synthetic':
        from bpx_sysid.mujoco_backend import MuJoCoReplay
        c = model_contract(args.dt, delay_mode=args.delay_mode)
        t, target, _ = trajectory(dt=args.dt, duration=args.duration, joint='all', ramp=min(1.,args.duration/2))
        truth = np.r_[np.linspace(.003, .009, 12), np.linspace(.02, .08, 12),
                       np.linspace(.01, .04, 12), np.linspace(-.01, .01, 12), 3]
        model = MuJoCoReplay(c)
        provisional = dict(dof_pos=target.copy(), dof_vel=np.zeros_like(target), des_dof_pos=target)
        measured = model.replay(truth, provisional)
        velocity = np.gradient(measured, args.dt, axis=0)
        velocity[0] = 0.  # Actual known initial velocity, not a finite-difference estimate.
        save_dataset(args.output, t, measured, target, c, velocity=velocity,
                     source='synthetic', source_session_id=uuid.uuid4().hex, engine='mujoco', calibrated=False, real_robot_ready=False)
        write_json(Path(args.output)/'truth.json', parameters(truth, c, source='synthetic_truth'))
        print(f'Wrote {len(t)} known-parameter samples; no hardware used.')
        return
    data, meta = load_dataset(args.data)
    if args.command == 'inspect':
        print(json.dumps(dict(source=meta['source'], samples=len(data['time']),
            duration_s=float(data['time'][-1]-data['time'][0]),
            physics_dt_s=meta['physics_dt_s'], joint_names=meta['joint_names'],
            position_min_rad=data['dof_pos'].min(axis=0).tolist(),
            position_max_rad=data['dof_pos'].max(axis=0).tolist()), indent=2))
        return
    from bpx_sysid.mujoco_backend import MuJoCoReplay
    backend = MuJoCoReplay(meta, getattr(args, 'population', 1))
    try:
        if args.command == 'fit':
            from bpx_sysid.optimize import fit
            fit(backend, data, meta, args.bounds, args.output, args.population,
                args.generations, args.seed, args.warmup)
        elif args.command == 'replay':
            print(json.dumps(replay_report(backend, data, meta, args.parameters, args.output, args.warmup), indent=2))
    finally:
        backend.close()


if __name__ == '__main__':
    main()
