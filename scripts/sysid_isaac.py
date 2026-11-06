#!/usr/bin/env python3
"""Fixed-base PhysX identification/replay, isolated from locomotion training."""
import argparse
from project_paths import add_project_source
add_project_source()
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode', choices=['check', 'synthetic', 'fit', 'replay'], default='check')
p.add_argument('--data', required=True)
p.add_argument('--parameters', required=True, help='Parameter file for check/replay or model provenance')
p.add_argument('--bounds')
p.add_argument('--output', required=True)
p.add_argument('--population', type=int, default=32)
p.add_argument('--generations', type=int, default=10)
p.add_argument('--seed', type=int, default=2026)
p.add_argument('--warmup', type=float, default=.5)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.mode == 'fit' and not a.bounds:
    p.error('--bounds is required for fit')
app = AppLauncher(a).app
try:
    import json
    from pathlib import Path
    from bpx_sysid.core import parameter_vector, write_json, model_contract
    from bpx_sysid.data import load_dataset
    from bpx_sysid.isaac_backend import IsaacReplay
    from sysid import replay_report
    d, m = load_dataset(a.data)
    params = json.loads(Path(a.parameters).read_text())
    parameter_vector(params)
    for key in model_contract():
        if params[key] != m[key]:
            raise ValueError(f'Parameter/data contract mismatch: {key}')
    backend = IsaacReplay(m, a.population if a.mode == 'fit' else 1, a.device or 'cuda:0')
    try:
        if a.mode == 'synthetic':
            import uuid
            import numpy as np
            from bpx_sysid.data import save_dataset
            from bpx_sysid.core import parameters
            values = parameter_vector(params)
            _, predictions = backend.evaluate([values], d, 0, trajectories=True)
            velocity = np.gradient(predictions[0], m['physics_dt_s'], axis=0)
            velocity[0] = d['dof_vel'][0]
            contract = {key: m[key] for key in model_contract()}
            save_dataset(a.output, d['time'], predictions[0], d['des_dof_pos'], contract,
                         velocity=velocity, source='synthetic', engine='physx',
                         source_session_id=uuid.uuid4().hex, calibrated=False, real_robot_ready=False)
            write_json(Path(a.output)/'truth.json', parameters(values, contract, source='synthetic_truth'))
            print('PhysX synthetic trajectory and truth saved.', flush=True)
        elif a.mode == 'fit':
            from bpx_sysid.optimize import fit
            fit(backend, d, m, a.bounds, a.output, a.population, a.generations, a.seed, a.warmup)
        else:
            report = replay_report(backend, d, m, a.parameters, a.output, a.warmup)
            print(json.dumps(report, indent=2))
            if a.mode == 'check':
                print('Fixed-base PhysX replay and parameter-write checks completed; not calibration qualification.')
    finally:
        backend.close()
except BaseException as exc:
    # Kit's immediate shutdown may terminate with status zero before Python can
    # re-raise an exception. Preserve CLI failure status and the full traceback.
    import os
    import sys
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(130 if isinstance(exc, KeyboardInterrupt) else 1)
finally:
    print('Identification outputs flushed; closing Isaac Sim.', flush=True)
    app.close(wait_for_replicator=False, skip_cleanup=True)
