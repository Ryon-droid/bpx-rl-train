"""Fixed-command evaluation and measurements taken before Isaac Lab auto-resets."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch


EVALUATION_PHASES = (
    ("stand", (0.0, 0.0, 0.0)),
    ("forward", (0.7, 0.0, 0.0)),
    ("stop", (0.0, 0.0, 0.0)),
    ("backward", (-0.7, 0.0, 0.0)),
    ("left", (0.0, 0.7, 0.0)),
    ("right", (0.0, -0.2, 0.0)),
    ("yaw_left", (0.0, 0.0, 0.8)),
    ("yaw_right", (0.0, 0.0, -0.5)),
    ("combined", (0.3, 0.15, 0.3)),
    ("stop_end", (0.0, 0.0, 0.0)),
)


def record_evaluation_metrics(env):
    """Last termination term: observe the final state and never terminate an env."""
    evaluator = getattr(env, "bpx_evaluator", None)
    if evaluator is not None:
        evaluator.sample(env)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


class PlayEvaluator:
    """Accumulate actual tracking errors rather than command-manager log metrics."""

    SUM_FIELDS = (
        "samples", "vx_error_sq", "vy_error_sq", "wz_error_sq", "xy_error",
        "xy_speed", "tilt_deg", "torque_sq", "action_delta_sq", "slide_speed",
        "contact_samples", "saturated_joint_samples", "all_joint_samples",
        "failures", "timeouts", "base_contact", "bad_orientation",
        "completed_episode_seconds", "completed_episodes",
    )

    def __init__(self, env, joint_names, foot_names, warmup_s):
        self.dt = env.step_dt
        self.num_envs = env.num_envs
        self.warmup_steps = math.ceil(warmup_s / self.dt)
        self.joint_ids, _ = env.scene["robot"].find_joints(joint_names, preserve_order=True)
        self.foot_ids, _ = env.scene["robot"].find_bodies(foot_names, preserve_order=True)
        self.sensor_foot_ids, _ = env.scene.sensors["contact_forces"].find_bodies(
            foot_names, preserve_order=True
        )
        self.device = env.device
        self.rows = []
        self.active = False

    def begin_phase(self, cycle, name, command):
        self.cycle, self.name, self.command = cycle, name, command
        self.steps = 0
        self.sums = torch.zeros(len(self.SUM_FIELDS), dtype=torch.float64, device=self.device)
        self.peaks = torch.zeros(2, dtype=torch.float64, device=self.device)
        self.active = True

    def sample(self, env):
        if not self.active:
            return
        self.steps += 1
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        valid = (env.episode_length_buf > self.warmup_steps) & (self.steps > self.warmup_steps)
        command = env.command_manager.get_command("base_velocity")
        error_xy = robot.data.root_lin_vel_b[:, :2] - command[:, :2]
        error_yaw = robot.data.root_ang_vel_b[:, 2] - command[:, 2]
        speed_xy = robot.data.root_lin_vel_b[:, :2].norm(dim=-1)
        tilt_deg = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)) * (180 / math.pi)
        torque = robot.data.applied_torque[:, self.joint_ids]
        limits = robot.data.joint_effort_limits[:, self.joint_ids]
        saturated = (limits > 0) & (torque.abs() >= 0.95 * limits)
        delta = env.action_manager.action - env.action_manager.prev_action
        # Match the contact criterion used by the training feet_slide reward.
        contact = sensor.data.net_forces_w_history[:, :, self.sensor_foot_ids, :].norm(dim=-1).amax(dim=1) > 1.0
        slide = robot.data.body_lin_vel_w[:, self.foot_ids, :2].norm(dim=-1)
        contacts_valid = contact & valid[:, None]
        manager = env.termination_manager
        failed = manager.terminated
        timeout = manager.time_outs & ~failed
        ended = failed | timeout
        values = (
            valid.sum(), (error_xy[:, 0].square() * valid).sum(),
            (error_xy[:, 1].square() * valid).sum(), (error_yaw.square() * valid).sum(),
            (error_xy.norm(dim=-1) * valid).sum(), (speed_xy * valid).sum(),
            (tilt_deg * valid).sum(), (torque.square() * valid[:, None]).sum(),
            (delta.square() * valid[:, None]).sum(), (slide * contacts_valid).sum(),
            contacts_valid.sum(), saturated.sum(), torque.numel() * torch.ones((), device=self.device),
            failed.sum(), timeout.sum(), (manager.get_term("base_contact") & failed).sum(),
            (manager.get_term("bad_orientation") & failed).sum(),
            (env.episode_length_buf * self.dt * ended).sum(), ended.sum(),
        )
        self.sums += torch.stack(values).to(torch.float64)
        self.peaks = torch.maximum(self.peaks, torch.stack((tilt_deg.max(), torque.abs().max())).to(torch.float64))

    def finish_phase(self):
        if not self.active or self.steps == 0:
            self.active = False
            return None
        self.active = False
        sums = dict(zip(self.SUM_FIELDS, self.sums.cpu().tolist()))
        tilt_peak, torque_peak = self.peaks.cpu().tolist()
        n = sums["samples"]
        joints = len(self.joint_ids)
        row = {
            "cycle": self.cycle, "phase": self.name,
            "command_vx_m_s": self.command[0], "command_vy_m_s": self.command[1],
            "command_wz_rad_s": self.command[2], "sim_seconds": self.steps * self.dt,
            "robot_seconds": self.steps * self.dt * self.num_envs,
            "measured_robot_seconds": n * self.dt,
            "rmse_vx_m_s": math.sqrt(sums["vx_error_sq"] / n) if n else None,
            "rmse_vy_m_s": math.sqrt(sums["vy_error_sq"] / n) if n else None,
            "rmse_wz_rad_s": math.sqrt(sums["wz_error_sq"] / n) if n else None,
            "mean_xy_error_m_s": sums["xy_error"] / n if n else None,
            "mean_xy_speed_m_s": sums["xy_speed"] / n if n else None,
            "mean_tilt_deg": sums["tilt_deg"] / n if n else None,
            "peak_tilt_deg": tilt_peak, "peak_estimated_torque_nm": torque_peak,
            "rms_estimated_torque_nm": math.sqrt(sums["torque_sq"] / (n * joints)) if n else None,
            "rms_action_delta": math.sqrt(sums["action_delta_sq"] / (n * joints)) if n else None,
            "mean_contact_slide_m_s": sums["slide_speed"] / sums["contact_samples"] if sums["contact_samples"] else None,
            "estimated_torque_saturation_pct": 100 * sums["saturated_joint_samples"] / sums["all_joint_samples"],
            "failures": int(sums["failures"]), "timeouts": int(sums["timeouts"]),
            "base_contact_terminations": int(sums["base_contact"]),
            "bad_orientation_terminations": int(sums["bad_orientation"]),
            "mean_completed_episode_s": sums["completed_episode_seconds"] / sums["completed_episodes"] if sums["completed_episodes"] else None,
        }
        row["failures_per_robot_minute"] = row["failures"] / (row["robot_seconds"] / 60)
        self.rows.append(row)
        errors = "/".join(f"{row[key]:.3f}" if row[key] is not None else "n/a" for key in (
            "rmse_vx_m_s", "rmse_vy_m_s", "rmse_wz_rad_s"
        ))
        print(
            f"[EVAL] cycle={self.cycle} {self.name}: RMSE vx/vy/wz={errors} (m/s,m/s,rad/s), "
            f"failures={row['failures']}, timeouts={row['timeouts']}, "
            f"peak tilt={tilt_peak:.1f} deg, estimated torque saturation="
            f"{row['estimated_torque_saturation_pct']:.2f}%",
            flush=True,
        )
        return row

    def save(self, output_dir, metadata, completed):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exposure = sum(row["robot_seconds"] for row in self.rows)
        failures = sum(row["failures"] for row in self.rows)
        timeouts = sum(row["timeouts"] for row in self.rows)
        report = {
            "metadata": metadata, "completed_schedule": completed,
            "measurement_notes": [
                "Tracking, mean tilt, torque RMS, action delta and slide exclude phase and post-reset warmup.",
                "Failures, timeouts, peaks and saturation include warmup and use pre-reset states.",
                "A failure is a physical termination, not a timeout; automatic resets continue the schedule.",
                "ImplicitActuator torque is an estimate, sampled at the 50 Hz control boundary, not measured motor torque.",
                "Termination reason counts follow Isaac Lab attribution when several terms trigger together.",
                "Phases continue across episode timeouts; completed episodes may span multiple phases.",
            ],
            "summary": {
                "robot_seconds": exposure, "failures": failures, "timeouts": timeouts,
                "failures_per_robot_minute": failures / (exposure / 60) if exposure else None,
                "failure_pct_of_completed_episodes": 100 * failures / (failures + timeouts) if failures + timeouts else None,
            },
            "phases": self.rows,
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        if self.rows:
            with (output_dir / "phases.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
        rate = report["summary"]["failures_per_robot_minute"]
        print(f"[EVAL] Total: failures={failures}, timeouts={timeouts}, robot exposure={exposure:.1f} s")
        if rate is not None:
            print(f"[EVAL] Failures per robot minute: {rate:.4f}")
        print(f"[EVAL] Reports: {output_dir.resolve()}")
