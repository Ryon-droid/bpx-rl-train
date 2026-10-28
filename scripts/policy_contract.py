"""Observation contracts shared by training, export and camera-free inference."""

from __future__ import annotations

import json
from pathlib import Path

POLICY_TERM_DIMS = (3, 3, 3, 12, 12, 12)
POLICY_TERMS = (
    "base_ang_vel", "projected_gravity", "velocity_commands",
    "joint_pos_rel", "joint_vel_rel", "last_action",
)
FRAME_DIM = sum(POLICY_TERM_DIMS)
HISTORY_LENGTH = 20
TERRAIN_DIM = 187


def actor_input_dim(task: str) -> int:
    if task.endswith("_teacher"):
        return FRAME_DIM + 3 + TERRAIN_DIM
    if task.endswith("_blind"):
        return FRAME_DIM * HISTORY_LENGTH
    return FRAME_DIM


def export_basename(task: str) -> str:
    return f"{task}_{actor_input_dim(task)}"


def write_export_contract(task: str, output_dir: str) -> Path:
    contract = {
        "task": task,
        "input_dim": actor_input_dim(task),
        "output_dim": 12,
        "control_dt_s": 0.02,
        "deployment_allowed": not task.endswith("_teacher"),
        "history_length": HISTORY_LENGTH if task.endswith("_blind") else 1,
        "frame_dim": FRAME_DIM,
        "term_names": POLICY_TERMS,
        "term_dims": POLICY_TERM_DIMS,
        # Isaac Lab flattens each term's [time, dim], then joins the terms.
        "history_layout": "term_major_oldest_first",
        "history_initialization": "repeat_first_frame",
        "history_reset": "clear_on_episode_reset_or_controller_restart",
        "history_input": "scaled_clipped_45d_frame",
    }
    target = Path(output_dir) / f"{export_basename(task)}.json"
    target.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return target


class ProprioHistory:
    """Numpy inference history matching Isaac Lab 2.3.0 observation buffers.

    Call append exactly once per 20 ms control tick, after building the same
    scaled 45-D observation as the old policy. Never include linear velocity,
    terrain heights, contact forces, RGB or depth. Reset when control restarts.
    """

    def __init__(self, length: int = HISTORY_LENGTH):
        if length < 1:
            raise ValueError("history length must be positive")
        self.length = length
        self._frames = None

    def reset(self):
        self._frames = None

    def append(self, frame):
        import numpy as np

        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (FRAME_DIM,) or not np.isfinite(frame).all():
            raise ValueError(f"expected a finite ({FRAME_DIM},) proprioceptive frame")
        if self._frames is None:
            self._frames = np.repeat(frame[None, :], self.length, axis=0)
        else:
            self._frames[:-1] = self._frames[1:]
            self._frames[-1] = frame
        blocks = []
        start = 0
        for dim in POLICY_TERM_DIMS:
            blocks.append(self._frames[:, start:start + dim].reshape(-1))
            start += dim
        return np.concatenate(blocks)
