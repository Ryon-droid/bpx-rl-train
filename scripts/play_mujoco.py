"""CPU MuJoCo playback of BPX single-frame or blind-history TorchScript actors."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from queue import SimpleQueue
import time

import mujoco
import numpy as np
import torch

from mujoco_bpx import BPXSimulation, KP, KD, TORQUE_LIMIT
from play_evaluation import EVALUATION_PHASES
from policy_contract import ProprioHistory


class PhaseMetrics:
    """Sample before reset; retain failures/peaks during tracking warmup."""

    def __init__(self, cycle, name, command, dt, warmup_s):
        self.cycle, self.name = cycle, name
        self.command = np.asarray(command, dtype=float).copy()
        self.dt, self.warmup_s = dt, warmup_s
        self.steps = self.samples = self.failures = self.timeouts = self.pushes = 0
        self.error_sq = np.zeros(3)
        self.velocity_sum = np.zeros(3)
        self.peak_tilt = self.peak_torque = 0.0
        self.reasons = {"bad_orientation": 0, "base_contact": 0}

    def sample(self, sim, torque, reason, timeout):
        self.steps += 1
        velocity = sim.body_velocity()
        measured = np.array([velocity[3], velocity[4], velocity[2]])
        self.peak_tilt = max(self.peak_tilt, math.degrees(math.acos(float(np.clip(-sim.gravity()[2], -1, 1)))))
        self.peak_torque = max(self.peak_torque, torque)
        if self.steps * self.dt > self.warmup_s + 1e-9 and sim.data.time > self.warmup_s + 1e-9:
            self.samples += 1
            self.error_sq += (measured - self.command) ** 2
            self.velocity_sum += measured
        if reason:
            self.failures += 1
            self.reasons[reason] += 1
        elif timeout:
            self.timeouts += 1

    def row(self):
        rmse = np.sqrt(self.error_sq / self.samples) if self.samples else [None] * 3
        mean = self.velocity_sum / self.samples if self.samples else [None] * 3
        return dict(cycle=self.cycle, phase=self.name, command_vx=float(self.command[0]),
                    command_vy=float(self.command[1]), command_wz=float(self.command[2]),
                    simulated_s=self.steps * self.dt, measured_s=self.samples * self.dt,
                    rmse_vx_m_s=rmse[0], rmse_vy_m_s=rmse[1], rmse_wz_rad_s=rmse[2],
                    mean_vx_m_s=mean[0], mean_vy_m_s=mean[1], mean_wz_rad_s=mean[2],
                    failures=self.failures, timeouts=self.timeouts, pushes=self.pushes,
                    **self.reasons, peak_tilt_deg=self.peak_tilt, peak_pd_torque_nm=self.peak_torque)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="Exported *_45.pt or *_blind_900.pt, not model_<iteration>.pt")
    parser.add_argument("--history_length", type=int, default=1, choices=(1, 20),
                        help="Use 20 for exported *_blind_900.pt, 1 for the original 45-D actor.")
    parser.add_argument("--command", type=float, nargs=3, default=(0.7, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation seconds for constant-command playback")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--eval", action="store_true", help="Use EVALUATION_PHASES from play_evaluation.py")
    parser.add_argument("--eval_phase_s", type=float, default=10.0)
    parser.add_argument("--eval_cycles", type=int, default=1)
    parser.add_argument("--warmup_s", type=float, default=2.0)
    parser.add_argument("--push_interval_s", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--push_velocity", type=float, default=0.6, help="XY velocity kick: each world-axis uniform +/- this m/s")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    for name in ("duration", "eval_phase_s", "friction"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive and finite")
    for name in ("warmup_s", "push_velocity"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name} must be nonnegative and finite")
    if args.eval_cycles < 1 or not np.isfinite(args.command).all():
        parser.error("Require --eval_cycles >= 1 and finite --command values")
    if args.push_interval_s and (not np.isfinite(args.push_interval_s).all()
                               or not 0 < args.push_interval_s[0] <= args.push_interval_s[1]):
        parser.error("--push_interval_s requires 0 < MIN <= MAX, both finite")
    if not args.policy.is_file():
        parser.error(f"Policy file not found: {args.policy}")
    if args.real_time is None:
        args.real_time = not args.headless
    return args


def run(args):
    torch.set_num_threads(1)
    history_length = args.history_length
    history = ProprioHistory(history_length)
    input_dim = 45 * history_length
    try:
        policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
        with torch.inference_mode():
            probe = policy(torch.zeros(1, input_dim))
        if probe.shape != (1, 12) or not torch.isfinite(probe).all():
            raise ValueError(f"Expected finite [1, 12] output for [1, {input_dim}] input")
    except Exception as exc:
        raise RuntimeError("Load an exported TorchScript actor; *_blind_900.pt requires --history_length 20") from exc
    if hasattr(policy, "reset"):
        policy.reset()
    sim = BPXSimulation(args.urdf, args.friction)
    dt = sim.control_dt
    schedule = [(cycle, name, command, args.eval_phase_s)
                for cycle in range(1, args.eval_cycles + 1) for name, command in EVALUATION_PHASES] if args.eval else [
                    (1, "constant", tuple(args.command), args.duration)]
    output = args.output_dir or args.policy.resolve().parent / "mujoco_evaluations" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    keys = SimpleQueue()
    viewer = None
    rows, push_events, command_events = [], [], []
    active = None
    completed = False
    error = None
    global_step = 0
    next_push = rng.uniform(*args.push_interval_s) if args.push_interval_s else math.inf

    def reset():
        sim.reset()
        history.reset()
        if hasattr(policy, "reset"):
            policy.reset()

    def finish_phase():
        nonlocal active
        if active is not None and active.steps:
            row = active.row()
            rows.append(row)
            rmse = [row[field] for field in ("rmse_vx_m_s", "rmse_vy_m_s", "rmse_wz_rad_s")]
            print(f"[PHASE] {active.name}: RMSE(vx,vy,wz)={rmse}, failures={active.failures}, "
                  f"pushes={active.pushes}, peak tilt={active.peak_tilt:.1f} deg", flush=True)
        active = None

    def push():
        kick = rng.uniform(-args.push_velocity, args.push_velocity, size=2)
        sim.push(kick)
        active.pushes += 1
        push_events.append(dict(time_s=global_step * dt, phase=active.name, delta_v_world_xy=kick.tolist()))
        print(f"[PUSH] t={global_step * dt:.2f}s, world XY delta-v={kick.round(3)} m/s", flush=True)

    try:
        if not args.headless:
            from mujoco import viewer as mj_viewer

            viewer = mj_viewer.launch_passive(sim.model, sim.data, key_callback=keys.put)
            with viewer.lock():
                viewer.opt.geomgroup[3] = 0  # show OBJ visuals, hide collision primitives
                viewer.cam.distance = 2.5
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -20
            print("Keys: W/S vx, A/D vy, Q/E yaw (+/-0.1); Space stop; R reset; P push. Close window to quit.")
        print(f"[MuJoCo {mujoco.__version__}] CPU physics + CPU actor, {1 / dt:.0f} Hz control; reports: {output}", flush=True)
        wall_start = time.monotonic()
        stopped = False
        for cycle, name, command, duration in schedule:
            active = PhaseMetrics(cycle, name, command, dt, args.warmup_s)
            print(f"[COMMAND] cycle={cycle}, {name}: {active.command.tolist()}", flush=True)
            for _ in range(math.ceil(duration / dt)):
                if viewer is not None and not viewer.is_running():
                    stopped = True
                    break
                while not keys.empty():
                    key = keys.get()
                    if key == ord("R"):
                        reset()
                        command_events.append(dict(time_s=global_step * dt, event="manual_reset"))
                    elif key == ord("P"):
                        push()
                    elif not args.eval:
                        changed = False
                        directions = {ord("W"): (0, 0.1), ord("S"): (0, -0.1),
                                      ord("A"): (1, 0.1), ord("D"): (1, -0.1),
                                      ord("Q"): (2, 0.1), ord("E"): (2, -0.1)}
                        if key == 32:
                            new_command = np.zeros(3)
                            changed = True
                        elif key in directions:
                            axis, delta = directions[key]
                            new_command = active.command.copy()
                            new_command[axis] += delta
                            changed = True
                        if changed:
                            finish_phase()
                            active = PhaseMetrics(cycle, "interactive", new_command, dt, args.warmup_s)
                            command_events.append(dict(time_s=global_step * dt, command=new_command.tolist()))
                            print(f"[COMMAND] {new_command.round(3).tolist()}", flush=True)
                if global_step * dt + 1e-9 >= next_push:
                    push()
                    next_push = global_step * dt + rng.uniform(*args.push_interval_s)
                with torch.inference_mode():
                    observation = history.append(sim.observation(active.command))
                    action = policy(torch.from_numpy(observation).unsqueeze(0))[0].numpy()
                torque = sim.step(action)
                global_step += 1
                reason = sim.failure_reason()
                timeout = sim.data.time >= 20.0 - 1e-9
                active.sample(sim, torque, reason, timeout)
                if reason or timeout:
                    if reason:
                        print(f"[FAIL] t={global_step * dt:.2f}s, {reason}; reset", flush=True)
                    reset()
                if viewer is not None:
                    with viewer.lock():
                        viewer.cam.lookat[:] = sim.data.xpos[sim.base_id]
                    viewer.sync()
                if args.real_time:
                    remaining = wall_start + global_step * dt - time.monotonic()
                    if remaining > 0:
                        time.sleep(min(remaining, dt))
            finish_phase()
            if stopped:
                break
        completed = not stopped
    except KeyboardInterrupt:
        print("Interrupted; saving partial evaluation.", flush=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        finish_phase()
        if viewer is not None:
            viewer.close()
        samples = sum(row["measured_s"] / dt for row in rows)
        aggregate = {"simulated_s": global_step * dt, "measured_s": samples * dt,
                     "failures": sum(row["failures"] for row in rows),
                     "timeouts": sum(row["timeouts"] for row in rows), "pushes": len(push_events),
                     "peak_tilt_deg": max((row["peak_tilt_deg"] for row in rows), default=0.0),
                     "peak_pd_torque_nm": max((row["peak_pd_torque_nm"] for row in rows), default=0.0)}
        for field in ("rmse_vx_m_s", "rmse_vy_m_s", "rmse_wz_rad_s"):
            aggregate[field] = math.sqrt(sum(row[field] ** 2 * row["measured_s"] / dt
                                           for row in rows if row[field] is not None) / samples) if samples else None
        metadata = dict(engine="mujoco", version=mujoco.__version__, device="cpu",
                        policy=str(args.policy.resolve()), urdf=str(args.urdf.resolve()) if args.urdf else "default BPX URDF",
                        friction=args.friction, kp=KP, kd=KD, torque_limit_nm=TORQUE_LIMIT,
                        physics_dt=sim.model.opt.timestep, control_dt=dt, seed=args.seed,
                        warmup_s=args.warmup_s, push_interval_s=args.push_interval_s,
                        push_velocity_m_s=args.push_velocity, eval=args.eval,
                        eval_phase_s=args.eval_phase_s, eval_cycles=args.eval_cycles,
                        perturbation="world XY additive velocity kicks only; no mass/COM/motor randomization",
                        termination="tilt >65 deg or torso-ground force >5 N; timeout 20 s")
        report = dict(metadata=metadata, completed_schedule=completed, error=error, summary=aggregate,
                      phases=rows, push_events=push_events, command_events=command_events)
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        with (output / "phases.csv").open("w", newline="") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        print(f"[RESULT] completed={completed}, failures={aggregate['failures']}, reports: {output}", flush=True)


if __name__ == "__main__":
    run(parse_args())
