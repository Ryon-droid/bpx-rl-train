"""PACE-style normalized CMA-ES; evaluate exported candidates independently."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from .core import PARAMETER_FIELDS, parameters, parameter_vector, digest, write_json, software_provenance


def bounds_matrix(path, synthetic):
    spec = json.loads(Path(path).read_text())
    if spec.get('synthetic_only', True) and not synthetic:
        raise ValueError('Synthetic example bounds are not approved for hardware data')
    if spec.get('joint_order') != 'bpx_type_major' or not spec.get('basis'):
        raise ValueError('Bounds need coordinate convention and a documented basis')
    rows = []
    for name in PARAMETER_FIELDS:
        a = np.asarray(spec[name], dtype=float)
        rows.extend(np.broadcast_to(a, (12, 2)).tolist())
    rows.append(spec['delay_steps'])
    b = np.array(rows, dtype=float)
    if b.shape != (49, 2) or not np.isfinite(b).all() or np.any(b[:, 0] >= b[:, 1]):
        raise ValueError('Need 49 finite, nonempty bounds')
    if np.any(b[:36, 0] < 0) or b[-1, 0] < 0 or b[-1, 1] > 1000:
        raise ValueError('Invalid physical bounds')
    if not np.equal(b[-1], np.floor(b[-1])).all():
        raise ValueError('Delay endpoints must be integer steps')
    return b, spec


def effective_parameters(a):
    a = np.array(a, copy=True)
    a[..., -1] = np.floor(a[..., -1])
    return a


def metrics(prediction, data, skip):
    error = prediction[skip:]-data['dof_pos'][skip:]
    if not np.isfinite(error).all():
        raise ValueError('Non-finite replay')
    return dict(mse_rad2=float(np.mean(error**2)),
                per_joint_rmse_rad=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                max_abs_error_rad=float(np.abs(error).max()), samples=len(error))


def fit(backend, data, meta, bounds_path, output, population=32, generations=10,
        seed=2026, warmup_s=.5):
    from cmaes import CMA
    if population < 4 or generations < 1:
        raise ValueError('Need population >=4 and generations >=1')
    skip = int(round(warmup_s/meta['physics_dt_s']))
    if not np.isfinite(warmup_s) or warmup_s < 0 or skip >= len(data['time'])-1:
        raise ValueError('Invalid warmup interval')
    bounds, spec = bounds_matrix(bounds_path, meta.get('source') == 'synthetic')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cma = CMA(mean=np.zeros(49), sigma=.3, bounds=np.tile([-1., 1.], (49, 1)),
              seed=seed, population_size=population)
    best_loss, best = float('inf'), None
    history = []
    for generation in range(generations):
        start = time.monotonic()
        xs = np.array([cma.ask() for _ in range(population)])
        ps = effective_parameters(bounds[:, 0]+(xs+1)/2*(bounds[:, 1]-bounds[:, 0]))
        scores, _ = backend.evaluate(ps, data, skip)
        cma.tell([(x, float(y)) for x, y in zip(xs, scores)])
        idx = int(np.argmin(scores))
        if scores[idx] < best_loss:
            best_loss, best = float(scores[idx]), ps[idx].copy()
        elapsed = time.monotonic()-start
        row = dict(generation=generation, best_mse_rad2=best_loss, generation_s=elapsed,
                   remaining_estimate_s=elapsed*(generations-generation-1))
        history.append(row)
        # Save the best actual candidate each generation, never substitute CMA's mean.
        checkpoint = parameters(best, meta, fitting_engine=backend.engine,
                                training_data_sha256=meta['data_sha256'], training_session_id=meta.get('source_session_id'),
                                software=software_provenance(), engine_version=backend.engine_version, seed=seed,
                                bounds_sha256=digest(bounds_path), calibrated=False)
        write_json(output/f'candidate_{generation:04d}.json', checkpoint)
        print(json.dumps(row), flush=True)
    if best is None or best_loss >= 1e12:
        raise ValueError('All candidates diverged; no valid calibration exported')
    # Both the best candidate and the distribution mean are independently replayed.
    mean = effective_parameters(bounds[:, 0]+(cma.mean+1)/2*(bounds[:, 1]-bounds[:, 0]))
    report = dict(algorithm='49D normalized CMA-ES, PACE-style trajectory loss',
                  engine=backend.engine, engine_version=backend.engine_version, software=software_provenance(),
                  history=history, bounds_basis=spec['basis'],
                  warmup_s=warmup_s, training_data_sha256=meta['data_sha256'],
                  holdout_validated=False, calibrated=False, real_robot_ready=False)
    for name, p in [('best', best), ('mean', mean)]:
        _, trajectories = backend.evaluate([p], data, skip, trajectories=True)
        try:
            report[name] = metrics(trajectories[0], data, skip)
        except ValueError:
            report[name] = dict(valid=False, reason='Diverged independent replay')
            if name == 'best':
                raise
        write_json(output/f'{name}.json', parameters(p, meta, fitting_engine=backend.engine,
                    training_data_sha256=meta['data_sha256'], training_session_id=meta.get('source_session_id'),
                                software=software_provenance(), engine_version=backend.engine_version, seed=seed,
                    bounds_sha256=digest(bounds_path), calibrated=False))
        np.save(output/f'{name}_replay.npy', trajectories[0])
    write_json(output/'report.json', report)
    return report
