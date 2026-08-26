"""Reward terms for the pickup-and-carry task.

Behaviour is broken into staged phases and rewards for later phases only
fire when earlier-phase preconditions are satisfied, so the agent cannot
farm reward out of context (e.g. squatting in the middle of the arena).

Phases:
  1. Approach the box (walk toward it)
  2. Face the box
  3. Squat when near the box
  4. Bring both hands close to / in contact with the box
  5. Grasp the box (both hands touching, close to center)
  6. Lift the box / stand back up while grasped
  7. Carry the box while tracking a base-velocity command
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ===========================================================================
# Helpers
# ===========================================================================


def _box_relative_xy(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    box: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    rel_w = box.data.root_pos_w - robot.data.root_pos_w
    rel_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel_w)
    return rel_b[:, 0], rel_b[:, 1]


def _hand_positions_w(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    hand_body_names: list[str],
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies(list(hand_body_names))
    return robot.data.body_pos_w[:, body_ids, :]


def _squat_phase(env: "ManagerBasedRLEnv", period: float) -> torch.Tensor:
    """[0, 1) の周期位相。0=立ち、0.5=しゃがみ、1=立ち。"""
    return ((env.episode_length_buf * env.step_dt) % period) / period


def _squat_depth(env: "ManagerBasedRLEnv", period: float) -> torch.Tensor:
    """[0, 1] のしゃがみ深さ。0=立ち期、1=しゃがみピーク。"""
    phi = _squat_phase(env, period)
    return 0.5 - 0.5 * torch.cos(2 * math.pi * phi)


def is_grasped(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    hand_body_names: list[str],
    contact_sensor_name: str,
    grasp_dist: float = 0.18,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    box: RigidObject = env.scene[object_cfg.name]
    hands_w = _hand_positions_w(env, robot_cfg, list(hand_body_names))
    d = torch.linalg.norm(hands_w - box.data.root_pos_w.unsqueeze(1), dim=-1)
    both_near = (d < grasp_dist).all(dim=-1)

    cs: ContactSensor = env.scene.sensors[contact_sensor_name]
    forces = torch.linalg.norm(cs.data.net_forces_w[:, :, :], dim=-1)
    both_touch = (forces > force_threshold).sum(dim=-1) >= 2
    return (both_near & both_touch).float()


# ===========================================================================
# Phases 1-7 (pickup & carry) — 変更なし
# ===========================================================================


def approach_box(env, target_distance=0.55, std=0.35,
                 object_cfg=SceneEntityCfg("box"),
                 robot_cfg=SceneEntityCfg("robot")):
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    err = dist - target_distance
    return torch.exp(-(err * err) / (std * std))


def face_box(env, object_cfg=SceneEntityCfg("box"),
             robot_cfg=SceneEntityCfg("robot")):
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    return dx / dist


def squat_when_near_box(env, target_height=0.50, near_threshold=0.75, std=0.10,
                        object_cfg=SceneEntityCfg("box"),
                        robot_cfg=SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    near = (dist < near_threshold).float()
    h = robot.data.root_pos_w[:, 2]
    err = h - target_height
    return near * torch.exp(-(err * err) / (std * std))


def hold_still_when_squatting(env, near_threshold=0.75,
                              object_cfg=SceneEntityCfg("box"),
                              robot_cfg=SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    near = (dist < near_threshold).float()
    lin_speed = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
    ang_speed = torch.abs(robot.data.root_ang_vel_b[:, 2])
    return -near * (lin_speed + 0.5 * ang_speed)


def hands_near_box(env, std=0.15,
                   hand_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
                   near_threshold=0.75,
                   object_cfg=SceneEntityCfg("box"),
                   robot_cfg=SceneEntityCfg("robot")):
    box: RigidObject = env.scene[object_cfg.name]
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist_base = torch.sqrt(dx * dx + dy * dy + 1e-8)
    near = (dist_base < near_threshold).float()
    hands_w = _hand_positions_w(env, robot_cfg, list(hand_body_names))
    d = torch.linalg.norm(hands_w - box.data.root_pos_w.unsqueeze(1), dim=-1)
    score = torch.exp(-(d * d) / (std * std)).mean(dim=-1)
    return near * score


def hands_contact_box(env,
                      sensor_cfg=SceneEntityCfg("hand_contact", body_names=[".*_wrist_yaw_link"]),
                      force_threshold=1.0):
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.linalg.norm(cs.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    hit = (forces > force_threshold).float()
    both = (hit.sum(dim=-1) >= 2).float()
    any_ = (hit.sum(dim=-1) >= 1).float()
    return 0.3 * (any_ - both) + 1.0 * both


def grasp_bonus(env,
                hand_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
                contact_sensor_name="hand_contact", grasp_dist=0.18,
                object_cfg=SceneEntityCfg("box"),
                robot_cfg=SceneEntityCfg("robot")):
    return is_grasped(env, object_cfg, robot_cfg, list(hand_body_names),
                      contact_sensor_name, grasp_dist)


def lift_box(env, initial_z=0.1, target_lift=0.4, std=0.12,
             hand_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
             contact_sensor_name="hand_contact",
             object_cfg=SceneEntityCfg("box"),
             robot_cfg=SceneEntityCfg("robot")):
    box: RigidObject = env.scene[object_cfg.name]
    lifted = box.data.root_pos_w[:, 2] - initial_z
    err = lifted - target_lift
    gauss = torch.exp(-(err * err) / (std * std))
    gated = is_grasped(env, object_cfg, robot_cfg, list(hand_body_names), contact_sensor_name)
    return gated * gauss


def stand_up_when_lifting(env, stand_height=0.75, std=0.10,
                          hand_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
                          contact_sensor_name="hand_contact",
                          object_cfg=SceneEntityCfg("box"),
                          robot_cfg=SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]
    err = robot.data.root_pos_w[:, 2] - stand_height
    gauss = torch.exp(-(err * err) / (std * std))
    gated = is_grasped(env, object_cfg, robot_cfg, list(hand_body_names), contact_sensor_name)
    return gated * gauss


def carry_box_velocity(env, command_name="base_velocity", std=0.4,
                       hand_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
                       contact_sensor_name="hand_contact",
                       object_cfg=SceneEntityCfg("box"),
                       robot_cfg=SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    err = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2] - cmd, dim=-1)
    gauss = torch.exp(-(err * err) / (std * std))
    gated = is_grasped(env, object_cfg, robot_cfg, list(hand_body_names), contact_sensor_name)
    return gated * gauss


def drop_box_penalty(env, min_z=0.05, object_cfg=SceneEntityCfg("box")):
    box: RigidObject = env.scene[object_cfg.name]
    return (box.data.root_pos_w[:, 2] < min_z).float() * -1.0


def box_collision_penalty(env, min_distance=0.30,
                          object_cfg=SceneEntityCfg("box"),
                          robot_cfg=SceneEntityCfg("robot")):
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    intrusion = (min_distance - dist).clamp(min=0.0)
    return -intrusion


# ===========================================================================
# Knee-primary squat rewards (これが今回のメイン)
# ---------------------------------------------------------------------------
# 設計方針:
#   - 「膝が曲がること」を最優先の主報酬にする
#   - 「胴体が低いこと」の報酬は膝が曲がっている時だけ与える
#     → 開脚で下げても報酬ゼロ、膝で下げると報酬2倍(高さ×膝)
#   - 開脚は 3 種類のペナルティで多角的に叩く(hip_roll角、絶対値、足間距離)
#   - 立ち止まりは周期目標で強制的に損させる
# ===========================================================================


def knee_bent_reward(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_knee_joint", "right_knee_joint"]
    ),
    max_angle: float = 1.5,
) -> torch.Tensor:
    """膝が曲がっているほど良い(常時ON、リニア、target無し)。

    しゃがみ学習の最主要報酬。target がないので勾配が消えず、初期方策から
    徐々に膝屈曲を発見させられる。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    return q.clamp(0.0, max_angle).mean(dim=-1)


def hip_pitch_bent_reward(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"]
    ),
    max_abs_angle: float = 1.2,
) -> torch.Tensor:
    """股関節pitchが曲がっているほど良い。膝と協調して初めて胴体が下がる。"""
    robot: Articulation = env.scene[robot_cfg.name]
    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    return q.abs().clamp(0.0, max_abs_angle).mean(dim=-1)


def height_low_gated_by_knee(
    env: "ManagerBasedRLEnv",
    max_height: float = 0.78,
    min_height: float = 0.40,
    knee_gate_min: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knee_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_knee_joint", "right_knee_joint"]
    ),
) -> torch.Tensor:
    """胴体が低いほど良い。ただし膝が曲がっていない場合は報酬ゼロ。

    開脚で下げても膝が伸びたままなら knee_ok≈0 で報酬が消えるので、
    「胴体を下げる = 膝を曲げる」以外の解が事実上塞がれる。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    h = robot.data.root_pos_w[:, 2]
    height_score = ((max_height - h) / (max_height - min_height)).clamp(0.0, 1.0)

    knee = robot.data.joint_pos[:, knee_cfg.joint_ids].mean(dim=-1)
    knee_ok = torch.sigmoid(10.0 * (knee - knee_gate_min))  # 0.5rad未満で0、以上で1

    return height_score * knee_ok


# ---------- 周期スクワット目標 ----------


def periodic_height_target(
    env: "ManagerBasedRLEnv",
    period: float = 3.0,
    stand_height: float = 0.75,
    squat_height: float = 0.50,
    std: float = 0.08,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """胴体高さが sin 波の目標に追従するほど高報酬(ガウシアン)。"""
    robot: Articulation = env.scene[robot_cfg.name]
    phi = _squat_phase(env, period)
    mid = 0.5 * (stand_height + squat_height)
    amp = 0.5 * (stand_height - squat_height)
    target = mid + amp * torch.cos(2 * math.pi * phi)
    h = robot.data.root_pos_w[:, 2]
    err = h - target
    return torch.exp(-(err * err) / (std * std))


def periodic_knee_target(
    env: "ManagerBasedRLEnv",
    period: float = 3.0,
    stand_knee: float = 0.1,
    squat_knee: float = 1.3,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_knee_joint", "right_knee_joint"]
    ),
) -> torch.Tensor:
    """膝角度が周期目標に近いほど良い(リニア、常時ON)。"""
    robot: Articulation = env.scene[robot_cfg.name]
    depth = _squat_depth(env, period)  # 0..1
    target = stand_knee + (squat_knee - stand_knee) * depth
    q = robot.data.joint_pos[:, robot_cfg.joint_ids].mean(dim=-1)
    return -torch.abs(q - target)


def periodic_hip_pitch_target(
    env: "ManagerBasedRLEnv",
    period: float = 3.0,
    stand_hip: float = 0.0,
    squat_hip: float = -0.7,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"]
    ),
) -> torch.Tensor:
    """股関節pitchが周期目標に近いほど良い(膝と協調して胴体を下げる)。"""
    robot: Articulation = env.scene[robot_cfg.name]
    depth = _squat_depth(env, period)
    target = stand_hip + (squat_hip - stand_hip) * depth
    q = robot.data.joint_pos[:, robot_cfg.joint_ids].mean(dim=-1)
    return -torch.abs(q - target)


# ---------- 開脚抑制(多角的に叩く) ----------


def hip_abduction_penalty(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_hip_roll_joint", "right_hip_roll_joint"]
    ),
) -> torch.Tensor:
    """hip_roll の二乗和ペナルティ。大きく開くほど急激に痛い。"""
    robot: Articulation = env.scene[robot_cfg.name]
    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    return -(q ** 2).sum(dim=-1)


def hip_roll_magnitude_penalty(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_hip_roll_joint", "right_hip_roll_joint"]
    ),
) -> torch.Tensor:
    """hip_roll の絶対値和ペナルティ。小さな外転も見逃さない。"""
    robot: Articulation = env.scene[robot_cfg.name]
    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    return -q.abs().sum(dim=-1)


def feet_lateral_distance_penalty(
    env: "ManagerBasedRLEnv",
    max_stance_width: float = 0.25,
    foot_body_names: list[str] = (".*_ankle_roll_link",),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """足の左右間隔が max_stance_width を超えた分だけリニアにペナルティ。"""
    robot: Articulation = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies(list(foot_body_names))
    feet_w = robot.data.body_pos_w[:, body_ids, :]
    rel = feet_w[:, 0, :] - feet_w[:, 1, :]
    rel_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel)
    lateral = torch.abs(rel_b[:, 1])
    excess = (lateral - max_stance_width).clamp(min=0.0)
    return -excess


def leg_symmetry_penalty(
    env: "ManagerBasedRLEnv",
    robot_cfg_roll: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_hip_roll_joint", "right_hip_roll_joint"]
    ),
    robot_cfg_pitch: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"]
    ),
    robot_cfg_knee: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_knee_joint", "right_knee_joint"]
    ),
) -> torch.Tensor:
    """左右対称なほど良い。hip_roll は符号反転で対称になる点に注意。"""
    robot: Articulation = env.scene[robot_cfg_roll.name]
    q_roll = robot.data.joint_pos[:, robot_cfg_roll.joint_ids]
    q_pitch = robot.data.joint_pos[:, robot_cfg_pitch.joint_ids]
    q_knee = robot.data.joint_pos[:, robot_cfg_knee.joint_ids]
    d_roll = q_roll[:, 0] + q_roll[:, 1]
    d_pitch = q_pitch[:, 0] - q_pitch[:, 1]
    d_knee = q_knee[:, 0] - q_knee[:, 1]
    return -(d_roll ** 2 + d_pitch ** 2 + d_knee ** 2)


# ---------- 足浮き抑制(緩め) ----------


def feet_air_time_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "foot_contact", body_names=[".*_ankle_roll_link"]
    ),
    grace_period: float = 0.2,
) -> torch.Tensor:
    """空中滞在時間が grace_period を超えた分だけリニアにペナルティ。"""
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air = cs.data.current_air_time[:, sensor_cfg.body_ids]
    excess = (air - grace_period).clamp(min=0.0).sum(dim=-1)
    return -excess


# ---------- 立ち止まり抑制 ----------


def freeze_penalty(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vz_threshold: float = 0.05,
) -> torch.Tensor:
    """胴体の上下速度がほぼゼロの間ペナルティ(周期スクワット中の停止を防ぐ)。"""
    robot: Articulation = env.scene[robot_cfg.name]
    vz = torch.abs(robot.data.root_lin_vel_w[:, 2])
    return -(vz < vz_threshold).float()

def knee_flexion_when_squatting(
    env: "ManagerBasedRLEnv",
    target_knee_angle: float = 1.2,
    std: float = 0.3,
    near_threshold: float = 0.75,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_knee_joint", "right_knee_joint"]
    ),
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """箱の近くにいるとき膝が target_knee_angle 付近に曲がっているほど高報酬。

    本タスク(pickup_carry_env_cfg)から参照されているので保持。
    squat-only では periodic_knee_target を使うためこの関数は使わない。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    near = (dist < near_threshold).float()

    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    err = q.mean(dim=-1) - target_knee_angle
    return near * torch.exp(-(err * err) / (std * std))

# ---------- 転倒(寝転がり)ペナルティ ----------


def upright_bonus(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """胴体が上向きに立っているほど良い。

    projected_gravity_b[:, 2] は立位で -1、横倒しで 0、逆さまで +1。
    (1 - g_z) / 2 で 立位=1, 横倒=0.5, 逆さま=0 になる。
    しゃがみの前傾程度なら 0.7〜0.9 くらい残る。
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    g_z = robot.data.projected_gravity_b[:, 2]  # 立位で -1
    return (1.0 - g_z) * 0.5


def fallen_penalty(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tilt_threshold: float = -0.3,   # g_z がこれより大きい(=より水平)ならペナルティ
    height_threshold: float = 0.30,  # pelvis がこれより低いとペナルティ
) -> torch.Tensor:
    """明らかに寝転がっているとき大きな負の報酬。

    - tilt_threshold=-0.3: g_z が -0.3 より大きい(=約72°以上傾いている)
    - height_threshold=0.30: pelvis が 30cm 以下(=寝てる/座り込み)
    どちらかを満たせばペナルティ。
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    g_z = robot.data.projected_gravity_b[:, 2]
    h = robot.data.root_pos_w[:, 2]

    too_tilted = (g_z > tilt_threshold).float()
    too_low = (h < height_threshold).float()
    return -(too_tilted + too_low)   # 両方満たせば -2、片方で -1

# ===========================================================================
# v2: 参照姿勢トラッキング型スクワット報酬
# ---------------------------------------------------------------------------
# 設計原則:
#   1. すべての項を [0, 1] の正値に。→ 生存が常に得 = 自殺(寝転がり)しない
#   2. 脚の全関節を「一本の参照姿勢」で同時に追従。
#      → 膝・股・足首・内転が互いに矛盾せず、開脚も自動的に潰れる
#   3. 転倒は「罰」ではなく「終了条件」で扱う
#   4. SceneEntityCfg は必ず RewTerm の params で渡すこと!
#      (デフォルト引数の SceneEntityCfg は resolve されず全関節を指す)
# ===========================================================================


def squat_pose_tracking(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    std: float = 0.50,
    stand_hip_pitch: float = -0.10,
    squat_hip_pitch: float = -0.95,
    stand_knee: float = 0.30,
    squat_knee: float = 1.50,
    stand_ankle: float = -0.20,
    squat_ankle: float = -0.55,
    hip_pitch_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ankle_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lateral_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """脚の参照姿勢への追従度 [0, 1]。これがスクワット学習の主報酬。

    位相 phi に応じて (hip_pitch, knee, ankle_pitch) の目標が
    立ち姿勢 <-> しゃがみ姿勢 の間を sin 補間する。
    lateral (hip_roll / hip_yaw) は常に 0 目標 = 開脚とねじれを禁止。

    立ち姿勢では hip_pitch + knee + ankle_pitch = 0 (胴体が垂直)。
    しゃがみ姿勢も -0.95 + 1.50 - 0.55 = 0 でほぼ垂直を保つ。

    NOTE: 4つの SceneEntityCfg は必ず RewTerm(params=...) で渡すこと。
    """
    robot: Articulation = env.scene[hip_pitch_cfg.name]
    depth = _squat_depth(env, period).unsqueeze(-1)  # (N, 1)

    err_sq = torch.zeros(env.num_envs, device=robot.data.joint_pos.device)

    for cfg, q_stand, q_squat in (
        (hip_pitch_cfg, stand_hip_pitch, squat_hip_pitch),
        (knee_cfg, stand_knee, squat_knee),
        (ankle_cfg, stand_ankle, squat_ankle),
    ):
        target = q_stand + (q_squat - q_stand) * depth      # (N, 1)
        q = robot.data.joint_pos[:, cfg.joint_ids]          # (N, 2)
        err_sq = err_sq + ((q - target) ** 2).mean(dim=-1)

    # hip_roll / hip_yaw は常に 0 (開脚・ねじれの禁止)
    q_lat = robot.data.joint_pos[:, lateral_cfg.joint_ids]
    err_sq = err_sq + (q_lat ** 2).mean(dim=-1)

    return torch.exp(-err_sq / (std * std))


def squat_height_tracking(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    stand_height: float = 0.74,
    squat_height: float = 0.52,
    std: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """胴体高さの参照追従 [0, 1]。関節角だけ合っていて体が浮く/傾く解を潰す。"""
    robot: Articulation = env.scene[robot_cfg.name]
    depth = _squat_depth(env, period)
    target = stand_height + (squat_height - stand_height) * depth
    err = robot.data.root_pos_w[:, 2] - target
    return torch.exp(-(err * err) / (std * std))


def feet_grounded(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("foot_contact"),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """接地している足の割合 [0, 0.5, 1.0]。正報酬なので跳ねる解を潰す。"""
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.linalg.norm(cs.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    return (forces > force_threshold).float().mean(dim=-1)


def feet_stance_width(
    env: "ManagerBasedRLEnv",
    target_width: float = 0.20,
    std: float = 0.12,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """足の左右間隔が target_width に近いほど良い [0, 1]。

    ペナルティではなく正報酬にすることで、報酬の総和が常に正に保たれる。
    NOTE: robot_cfg は body_names を指定して params で渡すこと。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    feet_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]   # (N, 2, 3)
    rel = feet_w[:, 0, :] - feet_w[:, 1, :]
    rel_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel)
    width = torch.abs(rel_b[:, 1])
    err = width - target_width
    return torch.exp(-(err * err) / (std * std))

# ---------------------------------------------------------------------------
# その場に留まる (定位置保持) 報酬 -- すべて [0, 1] の正値
# ---------------------------------------------------------------------------


def stay_in_place(
    env: "ManagerBasedRLEnv",
    std: float = 0.25,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """スポーン位置(env原点)からの水平ドリフトが小さいほど良い [0, 1]。

    std=0.25 なら 25cm ずれて 0.37、50cm ずれて 0.02 まで落ちる。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    offset = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    d_sq = (offset * offset).sum(dim=-1)
    return torch.exp(-d_sq / (std * std))


def low_base_speed(
    env: "ManagerBasedRLEnv",
    std: float = 0.30,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """胴体の水平速度が小さいほど良い [0, 1]。

    上下(z)は見ないのでスクワットの沈み込み/立ち上がりは阻害しない。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    v_sq = (robot.data.root_lin_vel_b[:, :2] ** 2).sum(dim=-1)
    return torch.exp(-v_sq / (std * std))


def heading_hold(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """初期のヨー向き(world +x)を保っているほど良い [0, 1]。

    yaw-only クォータニオン (w,0,0,z) に対し cos(yaw) = 2w^2 - 1。
    正面向きで 1.0、90度回って 0.5、真後ろで 0.0。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    w = yaw_quat(robot.data.root_quat_w)[:, 0]
    cos_yaw = (2.0 * w * w - 1.0).clamp(-1.0, 1.0)
    return (cos_yaw + 1.0) * 0.5


def feet_no_slip(
    env: "ManagerBasedRLEnv",
    std: float = 0.15,
    force_threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("foot_contact"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """接地している足が滑っていないほど良い [0, 1]。

    ドリフトの物理的な原因はほぼこれ。接地中の足だけ水平速度を見るので、
    遊脚(浮いている足)の動きは罰しない。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    forces = torch.linalg.norm(cs.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    in_contact = (forces > force_threshold).float()                      # (N, 2)

    foot_v = torch.linalg.norm(
        robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1
    )                                                                    # (N, 2)

    slip_sq = ((in_contact * foot_v) ** 2).sum(dim=-1)
    return torch.exp(-slip_sq / (std * std))

# ---------------------------------------------------------------------------
# 定位置保持 -- ペナルティ版 (動くとマイナス)
# ---------------------------------------------------------------------------
# 正報酬版 (stay_in_place など) は「棒立ちで満点」なので、何もしないだけで
# 報酬を稼げてしまう。ペナルティ版なら静止で 0、動いた分だけマイナス。
#
# すべて -(1 - exp(-x^2/std^2)) の形で値域 [-1, 0] に有界化してある。
# 有界にするのは重要: 1step あたりの合計が負になると、エージェントは
# 「早く終了した方が得」と判断してわざと転倒するようになる。
# ---------------------------------------------------------------------------


def drift_penalty(
    env: "ManagerBasedRLEnv",
    std: float = 0.25,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """スポーン位置から水平にずれた分だけマイナス [-1, 0]。

    静止で 0、25cm ずれて -0.63、50cm ずれて -0.98。
    """
    return stay_in_place(env, std=std, robot_cfg=robot_cfg) - 1.0


def base_speed_penalty(
    env: "ManagerBasedRLEnv",
    std: float = 0.30,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """胴体の水平速度の分だけマイナス [-1, 0]。上下(z)は見ない。"""
    return low_base_speed(env, std=std, robot_cfg=robot_cfg) - 1.0


def heading_penalty(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """初期のヨー向きからずれた分だけマイナス [-1, 0]。"""
    return heading_hold(env, robot_cfg=robot_cfg) - 1.0


def feet_slip_penalty(
    env: "ManagerBasedRLEnv",
    std: float = 0.15,
    force_threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("foot_contact"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """接地している足が滑った分だけマイナス [-1, 0]。

    ドリフトの物理的な原因はほぼこれ。遊脚は見ないので踏み替え自体は罰しない。
    """
    return feet_no_slip(
        env, std=std, force_threshold=force_threshold,
        sensor_cfg=sensor_cfg, asset_cfg=asset_cfg,
    ) - 1.0


def stance_width_penalty(
    env: "ManagerBasedRLEnv",
    target_width: float = 0.20,
    std: float = 0.12,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """足幅が target_width からずれた分だけマイナス [-1, 0]。"""
    return feet_stance_width(
        env, target_width=target_width, std=std, robot_cfg=robot_cfg
    ) - 1.0

# ---------------------------------------------------------------------------
# v3: 手を前に出す / 左右対称 / 胴を正面に向ける
# ---------------------------------------------------------------------------


def _hands_in_yaw_frame(
    env: "ManagerBasedRLEnv",
    hand_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """骨盤原点・ヨーのみ揃えた座標系での手の位置 (N, 2, 3)。

    yaw_quat を使うので「前(x)」は胴が前傾していても水平前方を指す。
    関節角の符号規約に依存しないので、USD の定義を調べなくても意図を書ける。
    """
    robot: Articulation = env.scene[hand_cfg.name]
    hands_w = robot.data.body_pos_w[:, hand_cfg.body_ids, :]          # (N, 2, 3)
    rel_w = hands_w - robot.data.root_pos_w.unsqueeze(1)              # (N, 2, 3)

    n_hand = rel_w.shape[1]
    q = yaw_quat(robot.data.root_quat_w).unsqueeze(1).expand(-1, n_hand, -1)
    rel_b = quat_apply_inverse(q.reshape(-1, 4), rel_w.reshape(-1, 3))
    return rel_b.reshape(rel_w.shape)


def hands_forward_tracking(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    stand_x: float = 0.05,
    squat_x: float = 0.30,
    stand_z: float = -0.15,
    squat_z: float = -0.10,
    std: float = 0.25,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """しゃがむにつれて両手が前方へ出るほど良い [0, 1]。

    骨盤から見た手の (前後 x, 上下 z) を位相に応じた目標に追従させる。
    左右どちらの手も同じ目標なので、この項自体が対称性も促す。

    NOTE: stand_x / stand_z は G1 の腕の自然位置の推定値。
          PLAY で実測して合わせると精度が上がる。
    """
    depth = _squat_depth(env, period)
    rel_b = _hands_in_yaw_frame(env, hand_cfg)                        # (N, 2, 3)

    tx = (stand_x + (squat_x - stand_x) * depth).unsqueeze(-1)        # (N, 1)
    tz = (stand_z + (squat_z - stand_z) * depth).unsqueeze(-1)

    err = (rel_b[:, :, 0] - tx) ** 2 + (rel_b[:, :, 2] - tz) ** 2     # (N, 2)
    return torch.exp(-err.mean(dim=-1) / (std * std))


def hands_symmetry_penalty(
    env: "ManagerBasedRLEnv",
    std: float = 0.10,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """両手が左右対称でないほどマイナス [-1, 0]。

    対称の定義 (骨盤ヨー座標系):
      前後 x : 左右で一致  -> 差が 0
      左右 y : 符号が反転  -> 和が 0
      上下 z : 左右で一致  -> 差が 0
    すべて絶対値/二乗で見るので body_ids の左右の並び順に依存しない。
    """
    rel_b = _hands_in_yaw_frame(env, hand_cfg)
    d_fwd = rel_b[:, 0, 0] - rel_b[:, 1, 0]
    d_lat = rel_b[:, 0, 1] + rel_b[:, 1, 1]
    d_up = rel_b[:, 0, 2] - rel_b[:, 1, 2]
    err = d_fwd ** 2 + d_lat ** 2 + d_up ** 2
    return torch.exp(-err / (std * std)) - 1.0


def torso_roll_penalty(
    env: "ManagerBasedRLEnv",
    std: float = 0.15,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """胴の左右傾き(ロール)だけを罰する [-1, 0]。

    projected_gravity_b = (gx, gy, gz)。前傾すると gx が動き、
    左右に傾くと gy が動く。gy だけを見るので、
    深いスクワットに必要な前傾(ピッチ)は一切罰しない。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    g_y = robot.data.projected_gravity_b[:, 1]
    return torch.exp(-(g_y * g_y) / (std * std)) - 1.0

# ---------------------------------------------------------------------------
# v4: 開脚の専用抑制 (pose_track の lateral 項では弱すぎたため独立させる)
# ---------------------------------------------------------------------------
# 深いスクワットは narrow stance だと踏ん張れないので、開脚が「安い抜け道」
# になる。pose_track の lateral 項はグループ内平均なので他の関節に薄められ、
# 単独では抑止力が足りない。そこで専用項として切り出す。
#
# ただし完全に 0 を要求するのは非現実的: 大腿が水平近くまで来る深さでは、
# 人間でも脚をやや開かないと胴の入るスペースがない。
# そこで「深さに応じた妥当な開き」を目標にし、それを超えた分を強く罰する。
# ---------------------------------------------------------------------------


def hip_abduction_tracking(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    stand_abduction: float = 0.00,
    squat_abduction: float = 0.18,
    std: float = 0.12,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """hip_roll の開き量が深さ相応かどうか [-1, 0]。

    |hip_roll| の平均を使うので左右の符号規約に依存しない。
    立ち位相では 0、完全しゃがみでは squat_abduction (約10度) までを許容し、
    それを「超えた分」だけマイナス (閉じている分は罰しない)。
    std が狭いので大きな開脚は即 -1 に飽和する。

    NOTE: robot_cfg は hip_roll だけを含む SceneEntityCfg を params で渡すこと。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    depth = _squat_depth(env, period)
    target = stand_abduction + (squat_abduction - stand_abduction) * depth

    abd = robot.data.joint_pos[:, robot_cfg.joint_ids].abs().mean(dim=-1)
    # 片側のみ: 目標より閉じている分は罰しない (clamp(min=0))。
    # 両側にすると「脚を閉じた棒立ち」が減点され、開脚を促してしまう。
    excess = (abd - target).clamp(min=0.0)
    return torch.exp(-(excess * excess) / (std * std)) - 1.0


def stance_width_penalty_phased(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    stand_width: float = 0.20,
    squat_width: float = 0.28,
    std: float = 0.08,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """足の左右間隔が深さ相応かどうか [-1, 0]。

    固定目標版 (stance_width_penalty) と違い、しゃがむにつれて
    わずかな足幅拡大を許容する。許容幅を超えた分のみ罰する (片側)。

    NOTE: robot_cfg は足の body_names を含む SceneEntityCfg を params で渡すこと。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    depth = _squat_depth(env, period)
    target = stand_width + (squat_width - stand_width) * depth

    feet_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]      # (N, 2, 3)
    rel = feet_w[:, 0, :] - feet_w[:, 1, :]
    rel_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel)
    width = torch.abs(rel_b[:, 1])

    # 片側のみ: 目標より狭い分は罰しない。
    excess = (width - target).clamp(min=0.0)
    return torch.exp(-(excess * excess) / (std * std)) - 1.0

# ---------------------------------------------------------------------------
# v5: 手を「膝の前」に「膝幅」で出す
# ---------------------------------------------------------------------------
# hands_forward_tracking は骨盤基準の絶対位置しか見ておらず、y(左右)を
# 一切拘束していなかった。hands_symmetry_penalty も「左右対称」しか
# 要求しないので、両手が中央で重なっていても満点になってしまう。
#
# ここでは基準を骨盤から「膝リンク」に変える:
#   - 膝はしゃがみと一緒に前下方へ動くので、オフセットが位相によらず安定する
#   - 「膝幅」「膝の前」という指示をそのまま数式にできる
#
# すべて左右の平均・間隔で評価するので find_bodies が返す左右の並び順に
# 依存しない (hands と knees で順序が違っても正しく動く)。
# ---------------------------------------------------------------------------


def _bodies_in_yaw_frame(
    env: "ManagerBasedRLEnv",
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """骨盤原点・ヨーのみ揃えた座標系での body 位置 (N, K, 3)。"""
    robot: Articulation = env.scene[body_cfg.name]
    pos_w = robot.data.body_pos_w[:, body_cfg.body_ids, :]
    rel_w = pos_w - robot.data.root_pos_w.unsqueeze(1)

    k = rel_w.shape[1]
    q = yaw_quat(robot.data.root_quat_w).unsqueeze(1).expand(-1, k, -1)
    rel_b = quat_apply_inverse(q.reshape(-1, 4), rel_w.reshape(-1, 3))
    return rel_b.reshape(rel_w.shape)


def hands_at_knee_front(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    stand_forward: float = 0.03,
    squat_forward: float = 0.15,
    stand_up: float = 0.20,
    squat_up: float = -0.10,
    std: float = 0.12,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """両手の中心が「膝の中心 + オフセット」に来ているほど良い [0, 1]。

    しゃがむにつれて手が膝より前・膝より下へ移動する。
    骨盤基準ではなく膝基準なので、深さが変わってもオフセットの意味が変わらない。

    NOTE: hand_cfg / knee_cfg はそれぞれ body_names を持つ SceneEntityCfg を
          params で渡すこと。
    """
    depth = _squat_depth(env, period)
    hands = _bodies_in_yaw_frame(env, hand_cfg)      # (N, 2, 3)
    knees = _bodies_in_yaw_frame(env, knee_cfg)      # (N, 2, 3)

    t_fwd = stand_forward + (squat_forward - stand_forward) * depth
    t_up = stand_up + (squat_up - stand_up) * depth

    err_fwd = hands[:, :, 0].mean(dim=-1) - (knees[:, :, 0].mean(dim=-1) + t_fwd)
    err_up = hands[:, :, 2].mean(dim=-1) - (knees[:, :, 2].mean(dim=-1) + t_up)

    return torch.exp(-(err_fwd ** 2 + err_up ** 2) / (std * std))


def hands_width_match(
    env: "ManagerBasedRLEnv",
    width_scale: float = 1.0,
    min_width: float = 0.16,
    std: float = 0.06,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """両手の左右間隔が膝の間隔に一致しているほど良い [0, 1]。

    これが無いと「両手を中央で揃える」解が対称性報酬を満点にしてしまう。
    膝が閉じている立ち位相でも最低 min_width は開くよう下限を設ける。

    左右の差の絶対値だけを使うので body_ids の並び順に依存しない。
    """
    hands = _bodies_in_yaw_frame(env, hand_cfg)
    knees = _bodies_in_yaw_frame(env, knee_cfg)

    hand_w = torch.abs(hands[:, 0, 1] - hands[:, 1, 1])
    knee_w = torch.abs(knees[:, 0, 1] - knees[:, 1, 1])
    target = (knee_w * width_scale).clamp(min=min_width)

    err = hand_w - target
    return torch.exp(-(err * err) / (std * std))

# ---------------------------------------------------------------------------
# v6: 腕の伸展 (しゃがみ切った時に肘が曲がっていたら罰する)
# ---------------------------------------------------------------------------
# 「腕が伸びている」を肘の関節角で判定すると、G1 の elbow のゼロ位置が
# 「真っ直ぐ」なのかどうか USD を見ないと分からず危険。
# そこで肩・肘・手の3点の幾何で測る:
#
#     straightness = ||肩 -> 手|| / (||肩 -> 肘|| + ||肘 -> 手||)
#
# 3点が一直線なら 1.0、曲がるほど小さくなる。肘の屈曲角を f とすると
# 厳密に cos(f/2) に一致する (上腕と前腕の長さが違っても単調性は保たれる)。
# リンク長も関節の符号規約も知らなくてよいのが利点。
# ---------------------------------------------------------------------------


def _sorted_by_lateral(rel_b: torch.Tensor) -> torch.Tensor:
    """(N, 2, 3) を y 座標の昇順に並べ替える。

    肩・肘・手を別々の正規表現で引くと find_bodies が返す左右の順序が
    一致する保証がない。y でソートすれば必ず [右腕, 左腕] の順に揃うので、
    3リンクを同じ腕どうしで対応付けられる。
    """
    idx = torch.argsort(rel_b[:, :, 1], dim=1)
    return torch.gather(rel_b, 1, idx.unsqueeze(-1).expand(-1, -1, 3))


def arm_extension_penalty(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    min_straightness: float = 0.97,
    std: float = 0.06,
    shoulder_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    elbow_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """しゃがみが深いほど、腕が伸びていないことを強く罰する [-1, 0]。

    深さでゲートしているので:
      立ち位相 (depth=0) -> ペナルティ 0 (腕は自然に下ろしていてよい)
      しゃがみ切り (depth=1) -> 伸展不足がそのままマイナス

    min_straightness=0.97 は肘の屈曲 28 度までを許容する値。
    それを超えた分だけ罰する片側ペナルティなので、伸ばしすぎは罰しない。

    NOTE: 3つの SceneEntityCfg は必ず params で渡すこと。
    """
    depth = _squat_depth(env, period)

    sh = _sorted_by_lateral(_bodies_in_yaw_frame(env, shoulder_cfg))   # (N, 2, 3)
    el = _sorted_by_lateral(_bodies_in_yaw_frame(env, elbow_cfg))
    hd = _sorted_by_lateral(_bodies_in_yaw_frame(env, hand_cfg))

    upper = torch.linalg.norm(el - sh, dim=-1)      # 上腕 (N, 2)
    fore = torch.linalg.norm(hd - el, dim=-1)       # 前腕 (N, 2)
    direct = torch.linalg.norm(hd - sh, dim=-1)     # 肩から手までの直線距離

    straightness = direct / (upper + fore + 1e-6)   # 1.0 で完全伸展
    deficit = (min_straightness - straightness).clamp(min=0.0).mean(dim=-1)

    shortfall = torch.exp(-(deficit * deficit) / (std * std)) - 1.0
    return depth * shortfall

# ---------------------------------------------------------------------------
# v7: 腕を「前方へ振る」 (肩関節を動かす動機を直接与える)
# ---------------------------------------------------------------------------
# hands_at_knee_front は手の「到達点」を目標にしていたが、
#   - 目標が腕の長さの外にあると勾配が薄くなる
#   - 腕を真下に垂らしていても部分点が入る
# ため、肩を回す動機が弱かった。
#
# ここでは腕の「向き」そのものを報酬にする:
#
#     forward = (手 - 肩) の単位ベクトルの前方(x)成分
#
#   腕を真下に垂らす      -> 0.00
#   鉛直から 37 度前へ振る -> 0.60
#   鉛直から 53 度前へ振る -> 0.80
#   真横(水平)に前へ伸ばす -> 1.00
#
# 向きなので必ず到達可能。全域で単調な勾配が出るため、肩関節を回す方向へ
# 素直に学習が進む。腕の長さもリンク数も知らなくてよい。
# ---------------------------------------------------------------------------


def arm_forward_direction(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    stand_forward: float = 0.00,
    squat_forward: float = 0.65,
    std: float = 0.15,
    shoulder_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """腕(肩->手)がどれだけ前方を向いているか [0, 1]。

    立ち位相では真下(0.0)、しゃがみ切りでは squat_forward を目標にする。
    左右それぞれで誤差を取ってから平均するので、片腕だけ前に出して
    平均でごまかす解が成立しない。

    NOTE: shoulder_cfg / hand_cfg は body_names を持つ SceneEntityCfg を
          params で渡すこと。
    """
    depth = _squat_depth(env, period)

    sh = _sorted_by_lateral(_bodies_in_yaw_frame(env, shoulder_cfg))   # (N, 2, 3)
    hd = _sorted_by_lateral(_bodies_in_yaw_frame(env, hand_cfg))

    v = hd - sh                                                        # (N, 2, 3)
    fwd = v[:, :, 0] / (torch.linalg.norm(v, dim=-1) + 1e-6)           # (N, 2)

    target = (stand_forward + (squat_forward - stand_forward) * depth).unsqueeze(-1)
    err_sq = ((fwd - target) ** 2).mean(dim=-1)
    return torch.exp(-err_sq / (std * std))

