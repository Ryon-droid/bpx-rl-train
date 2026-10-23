"""BPX URDF to MuJoCo scene and the Isaac-trained 45-D policy interface.

This module deliberately imports no Isaac Sim/Lab packages.
"""

from __future__ import annotations

import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep this order aligned with bpx_flat_env_cfg.JOINT_NAMES, never MJCF traversal order.
JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for joint in ("hip_roll", "hip_pitch", "knee")
    for leg in ("fl", "fr", "hl", "hr")
)
DEFAULT_JOINT_POS = np.array([0.0] * 4 + [0.8] * 4 + [-1.5] * 4)
CONTROL_DECIMATION = 4
ACTION_SCALE = 0.25
ACTION_CLIP = 100.0
KP, KD, TORQUE_LIMIT = 30.0, 1.0, 30.0


def numbers(values):
    return " ".join(f"{float(value):.12g}" for value in values)


def rpy_quaternion(rpy):
    """URDF fixed-axis XYZ angles -> MuJoCo wxyz quaternion."""
    r, p, y = np.asarray(rpy, dtype=float) / 2
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def origin_attributes(element):
    origin = element.find("origin")
    xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
    rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
    return {"pos": numbers(xyz), "quat": numbers(rpy_quaternion(rpy))}


def scene_xml(urdf_path=None, friction=1.0):
    """Preserve URDF link frames, inertias and collision primitives; add a free base."""
    asset_root = Path(os.environ.get("MIRRORME_BPX_ASSET_ROOT", PROJECT_ROOT / "resources/BPX"))
    urdf_path = Path(urdf_path or asset_root / "bpx/urdf/bpx.urdf").expanduser().resolve()
    robot = ET.parse(urdf_path).getroot()
    scene = ET.parse(PROJECT_ROOT / "resources/mujoco/scene.xml").getroot()
    scene.find("default/geom").set("friction", numbers([friction, 0.005, 0.0001]))
    asset, world = scene.find("asset"), scene.find("worldbody")
    actuators = ET.SubElement(scene, "actuator")
    links = {link.get("name"): link for link in robot.findall("link")}
    children = {}
    child_names = set()
    for joint in robot.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
        child_names.add(joint.find("child").get("link"))
    roots = set(links) - child_names
    if roots != {"torso"}:
        raise ValueError(f"Expected a single torso root, found {roots}")
    meshes = {}

    def add_link(name, parent, incoming=None):
        body = ET.SubElement(parent, "body", name=name, **(
            {"pos": "0 0 0.48"} if incoming is None else origin_attributes(incoming)
        ))
        if incoming is None:
            ET.SubElement(body, "freejoint", name="floating_base")
        elif incoming.get("type") != "fixed":
            if incoming.get("type") != "revolute":
                raise ValueError(f"Unsupported URDF joint type: {incoming.get('type')}")
            limit = incoming.find("limit")
            ET.SubElement(body, "joint", name=incoming.get("name"), type="hinge",
                          axis=incoming.find("axis").get("xyz"),
                          range=f"{limit.get('lower')} {limit.get('upper')}")
        link = links[name]
        inertial = link.find("inertial")
        if inertial is not None:
            inertia = inertial.find("inertia")
            xx, yy, zz, xy, xz, yz = [float(inertia.get(key)) for key in (
                "ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
            matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            attrs = origin_attributes(inertial)
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, np.fromstring(attrs.pop("quat"), sep=" "))
            matrix = rotation.reshape(3, 3) @ matrix @ rotation.reshape(3, 3).T
            ET.SubElement(body, "inertial", mass=inertial.find("mass").get("value"),
                          fullinertia=numbers([matrix[0, 0], matrix[1, 1], matrix[2, 2],
                                               matrix[0, 1], matrix[0, 2], matrix[1, 2]]), **attrs)
        for kind in ("collision", "visual"):
            for index, shape in enumerate(link.findall(kind)):
                geometry = shape.find("geometry")[0]
                attrs = origin_attributes(shape)
                attrs["name"] = f"{name}_{kind}_{index}"
                if geometry.tag == "mesh":
                    mesh_path = (urdf_path.parent / geometry.get("filename")).resolve()
                    scale = geometry.get("scale", "1 1 1")
                    key = (str(mesh_path), scale)
                    if key not in meshes:
                        mesh_name = f"robot_mesh_{len(meshes)}"
                        ET.SubElement(asset, "mesh", name=mesh_name, file=str(mesh_path), scale=scale)
                        meshes[key] = mesh_name
                    attrs.update(type="mesh", mesh=meshes[key])
                elif geometry.tag == "box":
                    attrs.update(type="box", size=numbers(np.fromstring(geometry.get("size"), sep=" ") / 2))
                elif geometry.tag == "cylinder":
                    attrs.update(type="cylinder", size=numbers([float(geometry.get("radius")),
                                                               float(geometry.get("length")) / 2]))
                elif geometry.tag == "sphere":
                    attrs.update(type="sphere", size=geometry.get("radius"))
                else:
                    raise ValueError(f"Unsupported URDF geometry: {geometry.tag}")
                if kind == "visual":
                    attrs.update(contype="0", conaffinity="0", group="2", material="robot_material")
                else:
                    attrs.update(group="3", rgba="0.3 0.8 0.3 0.3")
                ET.SubElement(body, "geom", **attrs)
        for joint in children.get(name, []):
            add_link(joint.find("child").get("link"), body, joint)

    add_link("torso", world)
    for joint_name in JOINT_NAMES:
        ET.SubElement(actuators, "motor", name=f"motor_{joint_name}", joint=joint_name,
                      gear="1", ctrllimited="true", ctrlrange="-30 30")
    ET.indent(scene)
    return ET.tostring(scene, encoding="unicode")


class BPXSimulation:
    def __init__(self, urdf_path=None, friction=1.0):
        self.xml = scene_xml(urdf_path, friction)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        if min(ids) < 0 or self.model.nq != 19 or self.model.nv != 18 or self.model.nu != 12:
            raise ValueError("BPX requires one free base and 12 named hinge joints/motors")
        self.qpos_ids = self.model.jnt_qposadr[ids]
        self.qvel_ids = self.model.jnt_dofadr[ids]
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.action = np.zeros(12)
        self.target = DEFAULT_JOINT_POS.copy()
        self.reset()

    @property
    def control_dt(self):
        return self.model.opt.timestep * CONTROL_DECIMATION

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_ids] = DEFAULT_JOINT_POS
        self.action.fill(0)
        self.target = DEFAULT_JOINT_POS.copy()
        mujoco.mj_forward(self.model, self.data)

    def body_velocity(self):
        velocity = np.empty(6)
        # mj_objectVelocity(BODY, local=1) uses the principal-inertia frame,
        # which need not coincide with the URDF torso / Isaac actor frame.
        # Obtain COM velocity in world axes, then rotate into the link frame.
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.base_id, velocity, 0)
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        velocity[:3] = rotation.T @ velocity[:3]
        velocity[3:] = rotation.T @ velocity[3:]
        return velocity  # [body angular velocity, body linear velocity]

    def gravity(self):
        return self.data.xmat[self.base_id].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

    def observation(self, command):
        # Isaac Lab clips each term before applying its observation scale.
        return np.concatenate([
            np.clip(self.body_velocity()[:3], -100, 100) * 0.25,
            np.clip(self.gravity(), -100, 100),
            np.clip(command, -100, 100) * [2.0, 2.0, 0.25],
            np.clip(self.data.qpos[self.qpos_ids] - DEFAULT_JOINT_POS, -100, 100),
            np.clip(self.data.qvel[self.qvel_ids], -100, 100) * 0.05,
            np.clip(self.action, -100, 100),
        ]).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("Policy output must be 12 finite actions")
        self.action = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
        self.target = DEFAULT_JOINT_POS + ACTION_SCALE * self.action
        peak_torque = 0.0
        # Hold position targets for 20 ms; recompute saturated PD torque every 5 ms.
        for _ in range(CONTROL_DECIMATION):
            torque = KP * (self.target - self.data.qpos[self.qpos_ids]) - KD * self.data.qvel[self.qvel_ids]
            self.data.ctrl[:] = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)
            peak_torque = max(peak_torque, float(np.abs(self.data.ctrl).max()))
            mujoco.mj_step(self.model, self.data)
        # Refresh body poses and velocities at the completed integration time.
        mujoco.mj_forward(self.model, self.data)
        bad_states = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                      mujoco.mjtWarning.mjWARN_BADQACC)
        if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or any(self.data.warning[warning].number for warning in bad_states)):
            raise RuntimeError("MuJoCo reported an unstable/non-finite simulation state")
        return peak_torque

    def push(self, velocity_xy):
        # Free-joint translation velocities are world-frame, matching Isaac's XY kick.
        self.data.qvel[:2] += velocity_xy
        mujoco.mj_forward(self.model, self.data)

    def failure_reason(self):
        if -self.gravity()[2] < np.cos(np.deg2rad(65.0)):
            return "bad_orientation"
        for i, contact in enumerate(self.data.contact):
            other = contact.geom2 if contact.geom1 == self.ground_id else (
                contact.geom1 if contact.geom2 == self.ground_id else -1)
            if other >= 0 and self.model.geom_bodyid[other] == self.base_id:
                force = np.empty(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                if np.linalg.norm(force[:3]) > 5.0:
                    return "base_contact"
        return None
