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

from typing import TYPE_CHECKING

import torch
import math

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    return robot.data.body_pos_w[:, body_ids, :]  # (N, K, 3)


def is_grasped(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    hand_body_names: list[str],
    contact_sensor_name: str,
    grasp_dist: float = 0.18,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """0/1 tensor: both hands close to box AND both hands in contact."""
    box: RigidObject = env.scene[object_cfg.name]
    hands_w = _hand_positions_w(env, robot_cfg, list(hand_body_names))
    d = torch.linalg.norm(hands_w - box.data.root_pos_w.unsqueeze(1), dim=-1)
    both_near = (d < grasp_dist).all(dim=-1)

    cs: ContactSensor = env.scene.sensors[contact_sensor_name]
    forces = torch.linalg.norm(cs.data.net_forces_w[:, :, :], dim=-1)
    both_touch = (forces > force_threshold).sum(dim=-1) >= 2
    return (both_near & both_touch).float()


# ---------------------------------------------------------------------------
# Phase 1: approach
# ---------------------------------------------------------------------------


def approach_box(
    env: "ManagerBasedRLEnv",
    target_distance: float = 0.55,
    std: float = 0.35,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    err = dist - target_distance
    return torch.exp(-(err * err) / (std * std))


# ---------------------------------------------------------------------------
# Phase 2: face
# ---------------------------------------------------------------------------


def face_box(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    return dx / dist  # cos(theta) — 1 when facing the box


# ---------------------------------------------------------------------------
# Phase 3: squat
# ---------------------------------------------------------------------------


def squat_when_near_box(
    env: "ManagerBasedRLEnv",
    target_height: float = 0.50,
    near_threshold: float = 0.75,
    std: float = 0.10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    near = (dist < near_threshold).float()

    h = robot.data.root_pos_w[:, 2]
    err = h - target_height
    return near * torch.exp(-(err * err) / (std * std))


def hold_still_when_squatting(
    env: "ManagerBasedRLEnv",
    near_threshold: float = 0.75,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    near = (dist < near_threshold).float()

    lin_speed = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
    ang_speed = torch.abs(robot.data.root_ang_vel_b[:, 2])
    return -near * (lin_speed + 0.5 * ang_speed)


# ---------------------------------------------------------------------------
# Phase 4: hands to box
# ---------------------------------------------------------------------------


def hands_near_box(
    env: "ManagerBasedRLEnv",
    std: float = 0.15,
    hand_body_names: list[str] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    near_threshold: float = 0.75,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian on mean(hand→box distance), gated by base-to-box proximity."""
    box: RigidObject = env.scene[object_cfg.name]
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist_base = torch.sqrt(dx * dx + dy * dy + 1e-8)
    near = (dist_base < near_threshold).float()

    hands_w = _hand_positions_w(env, robot_cfg, list(hand_body_names))
    d = torch.linalg.norm(hands_w - box.data.root_pos_w.unsqueeze(1), dim=-1)
    score = torch.exp(-(d * d) / (std * std)).mean(dim=-1)
    return near * score


def hands_contact_box(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("hand_contact", body_names=[".*_wrist_yaw_link"]),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """0 / 0.3 / 1.0 depending on how many hands are in contact."""
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.linalg.norm(cs.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    hit = (forces > force_threshold).float()
    both = (hit.sum(dim=-1) >= 2).float()
    any_ = (hit.sum(dim=-1) >= 1).float()
    return 0.3 * (any_ - both) + 1.0 * both


# ---------------------------------------------------------------------------
# Phase 5: grasp
# ---------------------------------------------------------------------------


def grasp_bonus(
    env: "ManagerBasedRLEnv",
    hand_body_names: list[str] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    contact_sensor_name: str = "hand_contact",
    grasp_dist: float = 0.18,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return is_grasped(env, object_cfg, robot_cfg, list(hand_body_names), contact_sensor_name, grasp_dist)


# ---------------------------------------------------------------------------
# Phase 6: lift / stand up while grasped
# ---------------------------------------------------------------------------


def lift_box(
    env: "ManagerBasedRLEnv",
    initial_z: float = 0.1,
    target_lift: float = 0.4,
    std: float = 0.12,
    hand_body_names: list[str] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    contact_sensor_name: str = "hand_contact",
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    box: RigidObject = env.scene[object_cfg.name]
    lifted = box.data.root_pos_w[:, 2] - initial_z
    err = lifted - target_lift
    gauss = torch.exp(-(err * err) / (std * std))
    gated = is_grasped(env, object_cfg, robot_cfg, list(hand_body_names), contact_sensor_name)
    return gated * gauss


def stand_up_when_lifting(
    env: "ManagerBasedRLEnv",
    stand_height: float = 0.75,
    std: float = 0.10,
    hand_body_names: list[str] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    contact_sensor_name: str = "hand_contact",
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    err = robot.data.root_pos_w[:, 2] - stand_height
    gauss = torch.exp(-(err * err) / (std * std))
    gated = is_grasped(env, object_cfg, robot_cfg, list(hand_body_names), contact_sensor_name)
    return gated * gauss


# ---------------------------------------------------------------------------
# Phase 7: carry
# ---------------------------------------------------------------------------


def carry_box_velocity(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    std: float = 0.4,
    hand_body_names: list[str] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    contact_sensor_name: str = "hand_contact",
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    err = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2] - cmd, dim=-1)
    gauss = torch.exp(-(err * err) / (std * std))
    gated = is_grasped(env, object_cfg, robot_cfg, list(hand_body_names), contact_sensor_name)
    return gated * gauss


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


def drop_box_penalty(
    env: "ManagerBasedRLEnv",
    min_z: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    box: RigidObject = env.scene[object_cfg.name]
    return (box.data.root_pos_w[:, 2] < min_z).float() * -1.0


def box_collision_penalty(
    env: "ManagerBasedRLEnv",
    min_distance: float = 0.30,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    intrusion = (min_distance - dist).clamp(min=0.0)
    return -intrusion

# ---------------------------------------------------------------------------
# Posture shaping: 
# ---------------------------------------------------------------------------


def hip_abduction_penalty(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=["left_hip_roll_joint", "right_hip_roll_joint"],
    ),
) -> torch.Tensor:
    """hip_roll The further the hip abduction deviates from the neutral position (0), the greater the penalty."""
    robot: Articulation = env.scene[robot_cfg.name]
    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    return -(q ** 2).sum(dim=-1)


def feet_lateral_distance_penalty(
    env: "ManagerBasedRLEnv",
    max_stance_width: float = 0.30,
    foot_body_names: list[str] = (".*_ankle_roll_link",),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """A linear penalty is applied for the amount by which the distance between the left and right feet exceeds `max_stance_width`."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies(list(foot_body_names))
    feet_w = robot.data.body_pos_w[:, body_ids, :]  # (N, 2, 3)
    rel = feet_w[:, 0, :] - feet_w[:, 1, :]
    rel_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel)
    lateral = torch.abs(rel_b[:, 1])
    excess = (lateral - max_stance_width).clamp(min=0.0)
    return -excess


def knee_flexion_when_squatting(
    env: "ManagerBasedRLEnv",
    target_knee_angle: float = 1.2,
    std: float = 0.3,
    near_threshold: float = 0.75,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=["left_knee_joint", "right_knee_joint"],
    ),
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """The higher the reward, the more your knee is bent toward the target_knee_angle when you're near the box."""
    robot: Articulation = env.scene[robot_cfg.name]
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    near = (dist < near_threshold).float()

    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    err = q.mean(dim=-1) - target_knee_angle
    return near * torch.exp(-(err * err) / (std * std))


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
    """The more symmetrical the left and right ankle joints are, the better (= the square of the difference)"""
    robot: Articulation = env.scene[robot_cfg_roll.name]
    q_roll = robot.data.joint_pos[:, robot_cfg_roll.joint_ids]
    q_pitch = robot.data.joint_pos[:, robot_cfg_pitch.joint_ids]
    q_knee = robot.data.joint_pos[:, robot_cfg_knee.joint_ids]
    d_roll = q_roll[:, 0] + q_roll[:, 1]         
    d_pitch = q_pitch[:, 0] - q_pitch[:, 1]     
    d_knee = q_knee[:, 0] - q_knee[:, 1]
    return -(d_roll ** 2 + d_pitch ** 2 + d_knee ** 2)


# ---------------------------------------------------------------------------
# Unconditional posture rewards (linear, always-on gradient)
# ---------------------------------------------------------------------------

def base_height_low(
    env: "ManagerBasedRLEnv",
    max_height: float = 0.78,
    min_height: float = 0.40,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    h = robot.data.root_pos_w[:, 2]
    return ((max_height - h) / (max_height - min_height)).clamp(0.0, 1.0)


def knee_bent(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=["left_knee_joint", "right_knee_joint"],
    ),
    max_angle: float = 1.6,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    return q.clamp(0.0, max_angle).mean(dim=-1)

# ---------------------------------------------------------------------------
# Periodic squat-stand cycle 
# ---------------------------------------------------------------------------
def _squat_phase(env: "ManagerBasedRLEnv", period: float) -> torch.Tensor:
    return ((env.episode_length_buf * env.step_dt) % period) / period


def periodic_squat_height(
    env: "ManagerBasedRLEnv",
    period: float = 3.0,
    stand_height: float = 0.75,
    squat_height: float = 0.50,
    std: float = 0.06,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    phi = _squat_phase(env, period)                              
    mid = 0.5 * (stand_height + squat_height)
    amp = 0.5 * (stand_height - squat_height)
    target = mid + amp * torch.cos(2 * math.pi * phi)              

    h = robot.data.root_pos_w[:, 2]
    err = h - target
    return torch.exp(-(err * err) / (std * std))


def periodic_knee_bend(
    env: "ManagerBasedRLEnv",
    period: float = 3.0,
    stand_knee: float = 0.1,
    squat_knee: float = 1.3,
    std: float = 0.25,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_knee_joint", "right_knee_joint"]
    ),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    phi = _squat_phase(env, period)
    mid = 0.5 * (stand_knee + squat_knee)
    amp = 0.5 * (squat_knee - stand_knee)
    target = mid - amp * torch.cos(2 * math.pi * phi)   

    q = robot.data.joint_pos[:, robot_cfg.joint_ids].mean(dim=-1)
    err = q - target
    return torch.exp(-(err * err) / (std * std))


def squat_motion_penalty(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    freeze_penalty: float = 0.5,
) -> torch.Tensor:
    
    robot: Articulation = env.scene[robot_cfg.name]
    vz = torch.abs(robot.data.root_lin_vel_w[:, 2])
    # vz < 0.05 m/s 
    is_frozen = (vz < 0.05).float()
    return -freeze_penalty * is_frozen

def periodic_squat_height(
    env: "ManagerBasedRLEnv",
    period: float = 3.0,
    stand_height: float = 0.75,
    squat_height: float = 0.50,
    std: float = 0.06,
    # --- 膝ゲート ---
    knee_gate_min: float = 0.4,          # squat期でこの角度以上曲がってないと報酬ゼロ
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knee_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_knee_joint", "right_knee_joint"]
    ),
) -> torch.Tensor:
    """胴体高さが目標に追従＋膝が十分曲がっているときのみ報酬。"""
    robot: Articulation = env.scene[robot_cfg.name]
    phi = _squat_phase(env, period)
    mid = 0.5 * (stand_height + squat_height)
    amp = 0.5 * (stand_height - squat_height)
    target = mid + amp * torch.cos(2 * math.pi * phi)

    h = robot.data.root_pos_w[:, 2]
    err = h - target
    height_score = torch.exp(-(err * err) / (std * std))

    # squat期の深さ(0=立ち, 1=しゃがみ)
    squat_depth = 0.5 - 0.5 * torch.cos(2 * math.pi * phi)  # 0..1

    # 膝が要求角度以上曲がっているか(squat期のみ厳しく)
    knee = robot.data.joint_pos[:, knee_cfg.joint_ids].mean(dim=-1)
    required_knee = squat_depth * knee_gate_min  # 立ち期は0でOK、しゃがみ期は0.4以上要求
    knee_ok = torch.sigmoid(10.0 * (knee - required_knee))  # 満たすほど1、そうでないと0

    return height_score * knee_ok

def hip_roll_magnitude_penalty(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_hip_roll_joint", "right_hip_roll_joint"],
    ),
) -> torch.Tensor:
    """hip_roll の絶対値の和(=どちらか一方でも外転していたらペナルティ)。"""
    robot: Articulation = env.scene[robot_cfg.name]
    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    return -q.abs().sum(dim=-1)

def feet_air_time_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("foot_contact", body_names=[".*_ankle_roll_link"]),
    grace_period: float = 0.2,   # 0.2秒までの浮きは無罰(自然な歩行余裕)
) -> torch.Tensor:
    """空中滞在時間が grace_period を超えた分だけリニアにペナルティ。"""
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air = cs.data.current_air_time[:, sensor_cfg.body_ids]   # (N, 2)
    excess = (air - grace_period).clamp(min=0.0).sum(dim=-1)
    return -excess