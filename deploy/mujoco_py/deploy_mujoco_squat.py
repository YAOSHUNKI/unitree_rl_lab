#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDS-free MuJoCo viewer for unitree_rl_lab policies (G1 29dof).

Loads an exported policy (policy.onnx) + its deploy.yaml and runs it directly
in MuJoCo — no unitree_mujoco, no g1_ctrl, no DDS, no joystick.

Designed for the PeriodicSquat policy (118-dim observation incl. box / hand /
squat_phase terms), but the observation is built generically from deploy.yaml
so it also works for the plain velocity policy (set --vx/--vy/--wz).

Usage:
    python deploy_mujoco_squat.py \
        --policy policies/squat/policy.onnx \
        --deploy policies/squat/deploy.yaml \
        --scene  g1_model/scene_squat.xml

Requires: pip install mujoco onnxruntime numpy pyyaml
"""

import argparse
import math
import time

import numpy as np
import yaml
import mujoco
import mujoco.viewer
import onnxruntime as ort


# SDK joint order (matches env.yaml joint_sdk_names and the g1_29dof.xml order)
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
NJ = len(JOINT_SDK_NAMES)  # 29


# ---------------------------------------------------------------- quaternion utils
def quat_to_mat(q):
    """q = (w,x,y,z) -> 3x3 rotation matrix (body->world)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def quat_rotate_inverse(q, v):
    """Rotate vector v from world into the frame described by q (w,x,y,z)."""
    return quat_to_mat(q).T @ np.asarray(v, dtype=float)


def yaw_quat(q):
    """Keep only the yaw component of q (w,x,y,z)."""
    w, x, y, z = q
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)])


# ---------------------------------------------------------------- main runner
class Runner:
    def __init__(self, args):
        self.args = args

        # ---- load deploy.yaml ----
        with open(args.deploy, "r") as f:
            self.cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

        self.step_dt = float(self.cfg["step_dt"])
        self.isaac2sdk = np.array(self.cfg["joint_ids_map"], dtype=int)   # isaac idx -> sdk idx
        self.kp_sdk = np.array(self.cfg["stiffness"], dtype=float)        # sdk order
        self.kd_sdk = np.array(self.cfg["damping"], dtype=float)          # sdk order

        default_isaac = np.array(self.cfg["default_joint_pos"], dtype=float)  # isaac order
        self.default_isaac = default_isaac
        self.default_sdk = np.zeros(NJ)
        self.default_sdk[self.isaac2sdk] = default_isaac

        # ---- action term (scale + offset), isaac order ----
        act = list(self.cfg["actions"].values())[0]
        scale = act["scale"]
        self.act_scale = np.array(scale, dtype=float) if isinstance(scale, list) \
            else np.full(NJ, float(scale))
        off = act.get("offset", 0.0)
        if isinstance(off, list) and len(off) == NJ:
            self.act_offset = np.array(off, dtype=float)          # velocity export case
        else:
            # JointPositionAction(use_default_offset=True): offset == default pose.
            # The squat export writes offset: 0.0 (custom term name), so fall back here.
            self.act_offset = default_isaac.copy()

        # ---- observation terms (name, scale, params) in order ----
        self.obs_terms = []
        for name, term in self.cfg["observations"].items():
            sc = term.get("scale", None)
            sc = np.array(sc, dtype=float) if isinstance(sc, list) else None
            self.obs_terms.append((name, sc, term.get("params", {}) or {}))

        # squat phase period (from the squat_phase obs term, default 6.0)
        self.period = args.period
        for name, _, params in self.obs_terms:
            if name == "squat_phase" and "period" in params:
                self.period = float(params["period"])

        # ---- MuJoCo model ----
        self.model = mujoco.MjModel.from_xml_path(args.scene)
        self.data = mujoco.MjData(self.model)
        self.n_sub = max(1, round(self.step_dt / self.model.opt.timestep))

        self._build_indices()

        # ---- ONNX policy ----
        self.sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.obs_dim = self.sess.get_inputs()[0].shape[-1]
        self.prev_action = np.zeros(NJ, dtype=np.float32)   # isaac order, raw

        # ---- initial pose ----
        self._reset_pose()
        self.sim_time = 0.0

    # ----------------------------------------------------------- indices
    def _jid(self, obj, name):
        i = mujoco.mj_name2id(self.model, obj, name)
        return i

    def _build_indices(self):
        m = self.model
        # 29 joint qpos / qvel addresses in SDK order
        self.q_adr = np.zeros(NJ, dtype=int)
        self.d_adr = np.zeros(NJ, dtype=int)
        for k, jn in enumerate(JOINT_SDK_NAMES):
            jid = self._jid(mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise RuntimeError(f"joint not found in model: {jn}")
            self.q_adr[k] = m.jnt_qposadr[jid]
            self.d_adr[k] = m.jnt_dofadr[jid]

        # actuator id in SDK order (match by transmitted joint)
        self.act_ids = np.full(NJ, -1, dtype=int)
        for aid in range(m.nu):
            jid = m.actuator_trnid[aid, 0]
            jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname in JOINT_SDK_NAMES:
                self.act_ids[JOINT_SDK_NAMES.index(jname)] = aid
        if np.any(self.act_ids < 0):
            missing = [JOINT_SDK_NAMES[i] for i in np.where(self.act_ids < 0)[0]]
            raise RuntimeError(f"actuators missing for joints: {missing}")
        self.ctrl_lo = m.actuator_ctrlrange[self.act_ids, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[self.act_ids, 1].copy()

        # bodies
        self.pelvis_bid = self._jid(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.lhand_bid = self._jid(mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        self.rhand_bid = self._jid(mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        self.box_bid = self._jid(mujoco.mjtObj.mjOBJ_BODY, "box")

        # base free joint qpos/qvel start
        base_jid = self._jid(mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self.base_qadr = m.jnt_qposadr[base_jid]
        self.base_dadr = m.jnt_dofadr[base_jid]

        # gyro sensor (base angular velocity in base frame)
        self.gyro_adr = None
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        if sid >= 0:
            self.gyro_adr = m.sensor_adr[sid]

    # ----------------------------------------------------------- reset
    def _reset_pose(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        # robot base
        d.qpos[self.base_qadr + 0: self.base_qadr + 3] = [0.0, 0.0, self.args.base_height]
        d.qpos[self.base_qadr + 3: self.base_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        # joints -> default (SDK order)
        d.qpos[self.q_adr] = self.default_sdk
        # box
        box_jid = self._jid(mujoco.mjtObj.mjOBJ_JOINT, "box_free")
        badr = m.jnt_qposadr[box_jid]
        d.qpos[badr:badr + 7] = [self.args.box_x, self.args.box_y, 0.1, 1, 0, 0, 0]
        mujoco.mj_forward(m, d)

    # ----------------------------------------------------------- observation
    def _read_state(self):
        d = self.data
        q_sdk = d.qpos[self.q_adr].copy()
        qd_sdk = d.qvel[self.d_adr].copy()
        base_quat = d.qpos[self.base_qadr + 3: self.base_qadr + 7].copy()
        base_pos = d.xpos[self.pelvis_bid].copy()
        if self.gyro_adr is not None:
            ang_vel_b = d.sensordata[self.gyro_adr: self.gyro_adr + 3].copy()
        else:
            ang_vel_b = quat_rotate_inverse(base_quat, d.qvel[self.base_dadr + 3: self.base_dadr + 6])
        return q_sdk, qd_sdk, base_quat, base_pos, ang_vel_b

    def _obs_term(self, name, params, state):
        q_sdk, qd_sdk, base_quat, base_pos, ang_vel_b = state
        d = self.data

        if name == "base_ang_vel":
            return ang_vel_b
        if name == "projected_gravity":
            return quat_rotate_inverse(base_quat, [0.0, 0.0, -1.0])
        if name in ("velocity_commands",):
            return np.array([self.args.vx, self.args.vy, self.args.wz])
        if name in ("joint_pos", "joint_pos_rel"):
            q_isaac = q_sdk[self.isaac2sdk]
            return q_isaac - self.default_isaac
        if name in ("joint_vel", "joint_vel_rel"):
            return qd_sdk[self.isaac2sdk]
        if name in ("actions", "last_action"):
            return self.prev_action.astype(float)

        # ---- box / hand terms (yaw-only base frame) ----
        yq = yaw_quat(base_quat)
        box_pos = d.xpos[self.box_bid].copy()
        lhand = d.xpos[self.lhand_bid].copy()
        rhand = d.xpos[self.rhand_bid].copy()

        if name == "box_rel":
            return quat_rotate_inverse(yq, box_pos - base_pos)
        if name == "box_dist_heading":
            rel = quat_rotate_inverse(yq, box_pos - base_pos)
            dx, dy = rel[0], rel[1]
            dist = math.sqrt(dx * dx + dy * dy + 1e-8)
            yaw = math.atan2(dy, dx)
            return np.array([dist, math.sin(yaw), math.cos(yaw)])
        if name == "hand_pos":
            lp = quat_rotate_inverse(yq, lhand - base_pos)
            rp = quat_rotate_inverse(yq, rhand - base_pos)
            return np.concatenate([lp, rp])
        if name == "box_in_hands":
            return np.concatenate([box_pos - lhand, box_pos - rhand])
        if name == "hand_touch":
            fl = np.linalg.norm(d.cfrc_ext[self.lhand_bid, 3:6])
            fr = np.linalg.norm(d.cfrc_ext[self.rhand_bid, 3:6])
            return np.array([float(fl > 1.0), float(fr > 1.0)])
        if name == "squat_phase":
            phi = (self.sim_time % self.period) / self.period
            return np.array([math.sin(2 * math.pi * phi), math.cos(2 * math.pi * phi)])

        raise RuntimeError(f"unhandled observation term: {name}")

    def build_obs(self):
        state = self._read_state()
        parts = []
        for name, scale, params in self.obs_terms:
            v = np.asarray(self._obs_term(name, params, state), dtype=float)
            if scale is not None:
                v = v * scale
            parts.append(v)
        obs = np.concatenate(parts).astype(np.float32)
        return obs

    # ----------------------------------------------------------- control
    def policy_step(self):
        obs = self.build_obs()
        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(
                f"observation size {obs.shape[0]} != policy input {self.obs_dim}. "
                f"deploy.yaml does not match this policy.onnx.")
        action = self.sess.run(None, {self.in_name: obs[None, :]})[0][0]
        self.prev_action = action.astype(np.float32)
        # processed target (isaac) -> sdk
        target_isaac = self.act_offset + self.act_scale * action
        target_sdk = np.zeros(NJ)
        target_sdk[self.isaac2sdk] = target_isaac
        return target_sdk

    def run(self):
        print(f"[info] obs_dim={self.obs_dim}  step_dt={self.step_dt}  "
              f"sim_dt={self.model.opt.timestep}  decimation={self.n_sub}  "
              f"squat_period={self.period}s")
        print("[info] close the viewer window (or Ctrl-C) to stop.")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                t0 = time.time()
                target_sdk = self.policy_step()
                for _ in range(self.n_sub):
                    q_sdk = self.data.qpos[self.q_adr]
                    qd_sdk = self.data.qvel[self.d_adr]
                    tau = self.kp_sdk * (target_sdk - q_sdk) - self.kd_sdk * qd_sdk
                    tau = np.clip(tau, self.ctrl_lo, self.ctrl_hi)
                    self.data.ctrl[self.act_ids] = tau
                    mujoco.mj_step(self.model, self.data)
                self.sim_time += self.step_dt
                viewer.sync()
                dt = self.step_dt - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="policies/squat/policy.onnx")
    p.add_argument("--deploy", default="policies/squat/deploy.yaml")
    p.add_argument("--scene", default="g1_model/scene_squat.xml")
    p.add_argument("--period", type=float, default=6.0,
                   help="squat period [s] (overridden by deploy.yaml squat_phase param)")
    p.add_argument("--base-height", type=float, default=0.80)
    p.add_argument("--box-x", type=float, default=0.7)
    p.add_argument("--box-y", type=float, default=0.0)
    # velocity command (only used if the policy has a velocity_commands term)
    p.add_argument("--vx", type=float, default=0.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    args = p.parse_args()

    Runner(args).run()


if __name__ == "__main__":
    main()
