#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDS-free MuJoCo runner for unitree_rl_lab policies (G1 29dof) — multi-task.

Layout:
    deploy/mujoco_py/
    ├── run_mujoco.py
    ├── g1_model/  (g1_29dof.xml, scene_29dof.xml, scene_box.xml, meshes/)
    └── tasks/
        └── <task>/
            ├── policy.onnx      (exported policy)
            ├── deploy.yaml      (obs / joint map / gains / default pose)
            └── task.yaml        (optional per-task viewer settings)

Run:
    python run_mujoco.py --task squat
    python run_mujoco.py --task velocity --vx 0.5
    python run_mujoco.py --task <your_task>

Add a new task: make tasks/<name>/ and drop policy.onnx + deploy.yaml in it.
(An optional task.yaml can set scene / period / base_height / box / command.)

Requires: pip install mujoco onnxruntime numpy pyyaml
"""

import argparse
import math
import os
import time
from collections import deque

import numpy as np
import yaml
import mujoco
import mujoco.viewer
import onnxruntime as ort


JOINT_SDK_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
NJ = len(JOINT_SDK_NAMES)

# observation terms that require a "box" body in the scene
BOX_TERMS = {"box_rel", "box_dist_heading", "box_in_hands"}
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- quaternion utils
def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def quat_rotate_inverse(q, v):
    return quat_to_mat(q).T @ np.asarray(v, dtype=float)


def yaw_quat(q):
    w, x, y, z = q
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)])


# ---------------------------------------------------------------- runner
class Runner:
    def __init__(self, args):
        self.args = args
        task_dir = os.path.join(args.tasks_dir, args.task)
        if not os.path.isdir(task_dir):
            raise SystemExit(f"[error] task folder not found: {task_dir}")

        policy_path = args.policy or os.path.join(task_dir, "policy.onnx")
        deploy_path = args.deploy or os.path.join(task_dir, "deploy.yaml")
        for p in (policy_path, deploy_path):
            if not os.path.isfile(p):
                raise SystemExit(f"[error] missing file: {p}")

        # per-task settings (all optional)
        self.task_cfg = {}
        tcfg = os.path.join(task_dir, "task.yaml")
        if os.path.isfile(tcfg):
            with open(tcfg) as f:
                self.task_cfg = yaml.safe_load(f) or {}

        def setting(name, default):
            if getattr(args, name, None) is not None:
                return getattr(args, name)
            return self.task_cfg.get(name, default)

        # ---- deploy.yaml ----
        with open(deploy_path) as f:
            self.cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

        self.step_dt = float(self.cfg["step_dt"])
        self.isaac2sdk = np.array(self.cfg["joint_ids_map"], dtype=int)
        self.kp_sdk = np.array(self.cfg["stiffness"], dtype=float)
        self.kd_sdk = np.array(self.cfg["damping"], dtype=float)
        self.default_isaac = np.array(self.cfg["default_joint_pos"], dtype=float)
        self.default_sdk = np.zeros(NJ)
        self.default_sdk[self.isaac2sdk] = self.default_isaac

        act = list(self.cfg["actions"].values())[0]
        sc = act["scale"]
        self.act_scale = np.array(sc, dtype=float) if isinstance(sc, list) else np.full(NJ, float(sc))
        off = act.get("offset", 0.0)
        self.act_offset = np.array(off, dtype=float) if (isinstance(off, list) and len(off) == NJ) \
            else self.default_isaac.copy()

        # ---- observation terms ----
        self.obs_terms = []
        self.needs_box = False
        for name, term in self.cfg["observations"].items():
            s = term.get("scale", None)
            s = np.array(s, dtype=float) if isinstance(s, list) else None
            h = int(term.get("history_length", 1) or 1)
            self.obs_terms.append({"name": name, "scale": s, "params": term.get("params", {}) or {}, "h": h})
            if name in BOX_TERMS:
                self.needs_box = True
        self.hist_order = setting("history_order", "oldest_first")

        # squat period
        self.period = float(setting("period", 6.0))
        for t in self.obs_terms:
            if t["name"] == "squat_phase" and "period" in t["params"]:
                self.period = float(t["params"]["period"])

        # command / init
        self.vx = float(setting("vx", 0.0)); self.vy = float(setting("vy", 0.0)); self.wz = float(setting("wz", 0.0))
        self.base_height = float(setting("base_height", 0.80))
        self.box_x = float(setting("box_x", 0.7)); self.box_y = float(setting("box_y", 0.0))

        # ---- scene selection ----
        scene = setting("scene", None)
        if scene is None:
            scene = "scene_box.xml" if self.needs_box else "scene_29dof.xml"
        self.scene_path = scene if os.path.isabs(scene) else os.path.join(args.model_dir, scene)
        if not os.path.isfile(self.scene_path):
            raise SystemExit(f"[error] scene not found: {self.scene_path}")

        # ---- MuJoCo ----
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)
        self.n_sub = max(1, round(self.step_dt / self.model.opt.timestep))
        self._build_indices()

        # ---- ONNX ----
        self.sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.obs_dim = self.sess.get_inputs()[0].shape[-1]
        self.prev_action = np.zeros(NJ, dtype=np.float32)

        self._hist = {t["name"]: deque(maxlen=t["h"]) for t in self.obs_terms}
        self._reset_pose()
        self.sim_time = 0.0

    # ----------------------------------------------------------- indices
    def _name2id(self, obj, name):
        return mujoco.mj_name2id(self.model, obj, name)

    def _build_indices(self):
        m = self.model
        self.q_adr = np.array([m.jnt_qposadr[self._name2id(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINT_SDK_NAMES])
        self.d_adr = np.array([m.jnt_dofadr[self._name2id(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINT_SDK_NAMES])

        self.act_ids = np.full(NJ, -1, dtype=int)
        for aid in range(m.nu):
            jid = m.actuator_trnid[aid, 0]
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jn in JOINT_SDK_NAMES:
                self.act_ids[JOINT_SDK_NAMES.index(jn)] = aid
        if np.any(self.act_ids < 0):
            raise RuntimeError("some joints have no actuator in the model")
        self.ctrl_lo = m.actuator_ctrlrange[self.act_ids, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[self.act_ids, 1].copy()

        self.pelvis_bid = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.lhand_bid = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        self.rhand_bid = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        self.box_bid = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "box")  # -1 if absent

        base_jid = self._name2id(mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self.base_qadr = m.jnt_qposadr[base_jid]
        self.base_dadr = m.jnt_dofadr[base_jid]
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.gyro_adr = m.sensor_adr[sid] if sid >= 0 else None

    # ----------------------------------------------------------- reset
    def _reset_pose(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.base_qadr:self.base_qadr + 3] = [0.0, 0.0, self.base_height]
        d.qpos[self.base_qadr + 3:self.base_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[self.q_adr] = self.default_sdk
        box_jid = self._name2id(mujoco.mjtObj.mjOBJ_JOINT, "box_free")
        if box_jid >= 0:
            badr = m.jnt_qposadr[box_jid]
            d.qpos[badr:badr + 7] = [self.box_x, self.box_y, 0.1, 1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        for dq in self._hist.values():
            dq.clear()

    # ----------------------------------------------------------- obs
    def _state(self):
        d = self.data
        q_sdk = d.qpos[self.q_adr].copy()
        qd_sdk = d.qvel[self.d_adr].copy()
        base_quat = d.qpos[self.base_qadr + 3:self.base_qadr + 7].copy()
        base_pos = d.xpos[self.pelvis_bid].copy()
        if self.gyro_adr is not None:
            ang_b = d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()
        else:
            ang_b = quat_rotate_inverse(base_quat, d.qvel[self.base_dadr + 3:self.base_dadr + 6])
        return q_sdk, qd_sdk, base_quat, base_pos, ang_b

    def _term_value(self, name, st):
        q_sdk, qd_sdk, base_quat, base_pos, ang_b = st
        d = self.data
        if name == "base_ang_vel":
            return ang_b
        if name == "projected_gravity":
            return quat_rotate_inverse(base_quat, [0.0, 0.0, -1.0])
        if name in ("velocity_commands", "keyboard_velocity_commands"):
            return np.array([self.vx, self.vy, self.wz])
        if name in ("joint_pos", "joint_pos_rel"):
            return q_sdk[self.isaac2sdk] - self.default_isaac
        if name in ("joint_vel", "joint_vel_rel"):
            return qd_sdk[self.isaac2sdk]
        if name in ("actions", "last_action"):
            return self.prev_action.astype(float)
        if name in ("gait_phase", "squat_phase"):
            phi = (self.sim_time % self.period) / self.period
            return np.array([math.sin(2 * math.pi * phi), math.cos(2 * math.pi * phi)])

        yq = yaw_quat(base_quat)
        lhand = d.xpos[self.lhand_bid].copy()
        rhand = d.xpos[self.rhand_bid].copy()
        box_pos = d.xpos[self.box_bid].copy() if self.box_bid >= 0 else np.zeros(3)
        if name == "box_rel":
            return quat_rotate_inverse(yq, box_pos - base_pos)
        if name == "box_dist_heading":
            rel = quat_rotate_inverse(yq, box_pos - base_pos)
            dist = math.sqrt(rel[0] ** 2 + rel[1] ** 2 + 1e-8)
            yaw = math.atan2(rel[1], rel[0])
            return np.array([dist, math.sin(yaw), math.cos(yaw)])
        if name == "hand_pos":
            return np.concatenate([quat_rotate_inverse(yq, lhand - base_pos),
                                   quat_rotate_inverse(yq, rhand - base_pos)])
        if name == "box_in_hands":
            return np.concatenate([box_pos - lhand, box_pos - rhand])
        if name == "hand_touch":
            fl = np.linalg.norm(d.cfrc_ext[self.lhand_bid, 3:6])
            fr = np.linalg.norm(d.cfrc_ext[self.rhand_bid, 3:6])
            return np.array([float(fl > 1.0), float(fr > 1.0)])
        raise RuntimeError(f"unhandled observation term: {name}")

    def build_obs(self):
        st = self._state()
        parts = []
        for t in self.obs_terms:
            v = np.asarray(self._term_value(t["name"], st), dtype=float)
            if t["scale"] is not None:
                v = v * t["scale"]
            dq = self._hist[t["name"]]
            if len(dq) == 0:
                for _ in range(t["h"]):
                    dq.append(v)
            else:
                dq.append(v)
            frames = list(dq)
            if self.hist_order == "newest_first":
                frames = frames[::-1]
            parts.append(np.concatenate(frames) if t["h"] > 1 else v)
        return np.concatenate(parts).astype(np.float32)

    # ----------------------------------------------------------- control
    def policy_step(self):
        obs = self.build_obs()
        if obs.shape[0] != self.obs_dim:
            raise SystemExit(
                f"[error] built observation = {obs.shape[0]} but policy expects {self.obs_dim}.\n"
                f"        deploy.yaml does not match this policy.onnx for task '{self.args.task}'.")
        action = self.sess.run(None, {self.in_name: obs[None, :]})[0][0]
        self.prev_action = action.astype(np.float32)
        target_isaac = self.act_offset + self.act_scale * action
        target_sdk = np.zeros(NJ)
        target_sdk[self.isaac2sdk] = target_isaac
        return target_sdk

    def run(self):
        print(f"[info] task='{self.args.task}' scene='{os.path.basename(self.scene_path)}' "
              f"obs_dim={self.obs_dim} needs_box={self.needs_box} decimation={self.n_sub} "
              f"period={self.period}s cmd=({self.vx},{self.vy},{self.wz})")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                t0 = time.time()
                target_sdk = self.policy_step()
                for _ in range(self.n_sub):
                    q = self.data.qpos[self.q_adr]
                    qd = self.data.qvel[self.d_adr]
                    tau = np.clip(self.kp_sdk * (target_sdk - q) - self.kd_sdk * qd, self.ctrl_lo, self.ctrl_hi)
                    self.data.ctrl[self.act_ids] = tau
                    mujoco.mj_step(self.model, self.data)
                self.sim_time += self.step_dt
                viewer.sync()
                dt = self.step_dt - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)


def main():
    p = argparse.ArgumentParser(description="Multi-task DDS-free MuJoCo runner (G1 29dof).")
    p.add_argument("--task", required=True, help="folder name under --tasks-dir")
    p.add_argument("--tasks-dir", default=os.path.join(HERE, "tasks"))
    p.add_argument("--model-dir", default=os.path.join(HERE, "g1_model"))
    p.add_argument("--policy", default=None, help="override tasks/<task>/policy.onnx")
    p.add_argument("--deploy", default=None, help="override tasks/<task>/deploy.yaml")
    p.add_argument("--scene", default=None, help="override scene xml (else auto: box vs no-box)")
    p.add_argument("--period", type=float, default=None)
    p.add_argument("--base-height", dest="base_height", type=float, default=None)
    p.add_argument("--box-x", dest="box_x", type=float, default=None)
    p.add_argument("--box-y", dest="box_y", type=float, default=None)
    p.add_argument("--vx", type=float, default=None)
    p.add_argument("--vy", type=float, default=None)
    p.add_argument("--wz", type=float, default=None)
    p.add_argument("--history-order", dest="history_order", choices=["oldest_first", "newest_first"], default=None)
    args = p.parse_args()
    Runner(args).run()


if __name__ == "__main__":
    main()
