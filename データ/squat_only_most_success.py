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
    return ((env.episode_length_buf * env.step_dt) % period) / period


def _squat_depth(env: "ManagerBasedRLEnv", period: float) -> torch.Tensor:
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


def knee_bent_reward(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_knee_joint", "right_knee_joint"]
    ),
    max_angle: float = 1.5,
) -> torch.Tensor:
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
    robot: Articulation = env.scene[robot_cfg.name]
    h = robot.data.root_pos_w[:, 2]
    height_score = ((max_height - h) / (max_height - min_height)).clamp(0.0, 1.0)

    knee = robot.data.joint_pos[:, knee_cfg.joint_ids].mean(dim=-1)
    knee_ok = torch.sigmoid(10.0 * (knee - knee_gate_min))  # 0.5rad未満で0、以上で1

    return height_score * knee_ok



def periodic_height_target(
    env: "ManagerBasedRLEnv",
    period: float = 3.0,
    stand_height: float = 0.75,
    squat_height: float = 0.50,
    std: float = 0.08,
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


def periodic_knee_target(
    env: "ManagerBasedRLEnv",
    period: float = 3.0,
    stand_knee: float = 0.1,
    squat_knee: float = 1.3,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_knee_joint", "right_knee_joint"]
    ),
) -> torch.Tensor:
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
    robot: Articulation = env.scene[robot_cfg.name]
    depth = _squat_depth(env, period)
    target = stand_hip + (squat_hip - stand_hip) * depth
    q = robot.data.joint_pos[:, robot_cfg.joint_ids].mean(dim=-1)
    return -torch.abs(q - target)


def hip_abduction_penalty(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_hip_roll_joint", "right_hip_roll_joint"]
    ),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    return -(q ** 2).sum(dim=-1)


def hip_roll_magnitude_penalty(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_hip_roll_joint", "right_hip_roll_joint"]
    ),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    return -q.abs().sum(dim=-1)


def feet_lateral_distance_penalty(
    env: "ManagerBasedRLEnv",
    max_stance_width: float = 0.25,
    foot_body_names: list[str] = (".*_ankle_roll_link",),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
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
    robot: Articulation = env.scene[robot_cfg_roll.name]
    q_roll = robot.data.joint_pos[:, robot_cfg_roll.joint_ids]
    q_pitch = robot.data.joint_pos[:, robot_cfg_pitch.joint_ids]
    q_knee = robot.data.joint_pos[:, robot_cfg_knee.joint_ids]
    d_roll = q_roll[:, 0] + q_roll[:, 1]
    d_pitch = q_pitch[:, 0] - q_pitch[:, 1]
    d_knee = q_knee[:, 0] - q_knee[:, 1]
    return -(d_roll ** 2 + d_pitch ** 2 + d_knee ** 2)



def feet_air_time_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "foot_contact", body_names=[".*_ankle_roll_link"]
    ),
    grace_period: float = 0.2,
) -> torch.Tensor:
    air = cs.data.current_air_time[:, sensor_cfg.body_ids]
    excess = (air - grace_period).clamp(min=0.0).sum(dim=-1)
    return -excess


def freeze_penalty(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vz_threshold: float = 0.05,
) -> torch.Tensor:
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
    robot: Articulation = env.scene[robot_cfg.name]
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    near = (dist < near_threshold).float()

    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    err = q.mean(dim=-1) - target_knee_angle
    return near * torch.exp(-(err * err) / (std * std))


def upright_bonus(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    g_z = robot.data.projected_gravity_b[:, 2]  
    return (1.0 - g_z) * 0.5


def fallen_penalty(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tilt_threshold: float = -0.3,   
    height_threshold: float = 0.30,  
) -> torch.Tensor:

    robot: RigidObject = env.scene[robot_cfg.name]
    g_z = robot.data.projected_gravity_b[:, 2]
    h = robot.data.root_pos_w[:, 2]

    too_tilted = (g_z > tilt_threshold).float()
    too_low = (h < height_threshold).float()
    return -(too_tilted + too_low)   

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

    robot: Articulation = env.scene[hip_pitch_cfg.name]
    depth = _squat_depth(env, period).unsqueeze(-1)  

    err_sq = torch.zeros(env.num_envs, device=robot.data.joint_pos.device)

    for cfg, q_stand, q_squat in (
        (hip_pitch_cfg, stand_hip_pitch, squat_hip_pitch),
        (knee_cfg, stand_knee, squat_knee),
        (ankle_cfg, stand_ankle, squat_ankle),
    ):
        target = q_stand + (q_squat - q_stand) * depth      
        q = robot.data.joint_pos[:, cfg.joint_ids]          
        err_sq = err_sq + ((q - target) ** 2).mean(dim=-1)

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
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.linalg.norm(cs.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    return (forces > force_threshold).float().mean(dim=-1)


def feet_stance_width(
    env: "ManagerBasedRLEnv",
    target_width: float = 0.20,
    std: float = 0.12,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    feet_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]   # (N, 2, 3)
    rel = feet_w[:, 0, :] - feet_w[:, 1, :]
    rel_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel)
    width = torch.abs(rel_b[:, 1])
    err = width - target_width
    return torch.exp(-(err * err) / (std * std))


def stay_in_place(
    env: "ManagerBasedRLEnv",
    std: float = 0.25,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    offset = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    d_sq = (offset * offset).sum(dim=-1)
    return torch.exp(-d_sq / (std * std))


def low_base_speed(
    env: "ManagerBasedRLEnv",
    std: float = 0.30,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    v_sq = (robot.data.root_lin_vel_b[:, :2] ** 2).sum(dim=-1)
    return torch.exp(-v_sq / (std * std))


def heading_hold(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
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
    robot: Articulation = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    forces = torch.linalg.norm(cs.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    in_contact = (forces > force_threshold).float()                      # (N, 2)

    foot_v = torch.linalg.norm(
        robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1
    )                                                                    # (N, 2)

    slip_sq = ((in_contact * foot_v) ** 2).sum(dim=-1)
    return torch.exp(-slip_sq / (std * std))


def drift_penalty(
    env: "ManagerBasedRLEnv",
    std: float = 0.25,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return stay_in_place(env, std=std, robot_cfg=robot_cfg) - 1.0


def base_speed_penalty(
    env: "ManagerBasedRLEnv",
    std: float = 0.30,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return low_base_speed(env, std=std, robot_cfg=robot_cfg) - 1.0


def heading_penalty(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return heading_hold(env, robot_cfg=robot_cfg) - 1.0


def feet_slip_penalty(
    env: "ManagerBasedRLEnv",
    std: float = 0.15,
    force_threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("foot_contact"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
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
    return feet_stance_width(
        env, target_width=target_width, std=std, robot_cfg=robot_cfg
    ) - 1.0


def _hands_in_yaw_frame(
    env: "ManagerBasedRLEnv",
    hand_cfg: SceneEntityCfg,
) -> torch.Tensor:
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
    robot: Articulation = env.scene[robot_cfg.name]
    g_y = robot.data.projected_gravity_b[:, 1]
    return torch.exp(-(g_y * g_y) / (std * std)) - 1.0

def hip_abduction_tracking(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    stand_abduction: float = 0.00,
    squat_abduction: float = 0.18,
    std: float = 0.12,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    robot: Articulation = env.scene[robot_cfg.name]
    depth = _squat_depth(env, period)
    target = stand_abduction + (squat_abduction - stand_abduction) * depth

    abd = robot.data.joint_pos[:, robot_cfg.joint_ids].abs().mean(dim=-1)

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
    robot: Articulation = env.scene[robot_cfg.name]
    depth = _squat_depth(env, period)
    target = stand_width + (squat_width - stand_width) * depth

    feet_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]      # (N, 2, 3)
    rel = feet_w[:, 0, :] - feet_w[:, 1, :]
    rel_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel)
    width = torch.abs(rel_b[:, 1])

    excess = (width - target).clamp(min=0.0)
    return torch.exp(-(excess * excess) / (std * std)) - 1.0


def _bodies_in_yaw_frame(
    env: "ManagerBasedRLEnv",
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
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

    depth = _squat_depth(env, period)
    hands = _bodies_in_yaw_frame(env, hand_cfg)      
    knees = _bodies_in_yaw_frame(env, knee_cfg)      

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

    hands = _bodies_in_yaw_frame(env, hand_cfg)
    knees = _bodies_in_yaw_frame(env, knee_cfg)

    hand_w = torch.abs(hands[:, 0, 1] - hands[:, 1, 1])
    knee_w = torch.abs(knees[:, 0, 1] - knees[:, 1, 1])
    target = (knee_w * width_scale).clamp(min=min_width)

    err = hand_w - target
    return torch.exp(-(err * err) / (std * std))



def _sorted_by_lateral(rel_b: torch.Tensor) -> torch.Tensor:
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
 
    depth = _squat_depth(env, period)

    sh = _sorted_by_lateral(_bodies_in_yaw_frame(env, shoulder_cfg))   
    el = _sorted_by_lateral(_bodies_in_yaw_frame(env, elbow_cfg))
    hd = _sorted_by_lateral(_bodies_in_yaw_frame(env, hand_cfg))

    upper = torch.linalg.norm(el - sh, dim=-1)      
    fore = torch.linalg.norm(hd - el, dim=-1)       
    direct = torch.linalg.norm(hd - sh, dim=-1)     

    straightness = direct / (upper + fore + 1e-6)   
    deficit = (min_straightness - straightness).clamp(min=0.0).mean(dim=-1)

    shortfall = torch.exp(-(deficit * deficit) / (std * std)) - 1.0
    return depth * shortfall


def arm_forward_direction(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    stand_forward: float = 0.00,
    squat_forward: float = 0.65,
    std: float = 0.15,
    shoulder_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    elbow_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    depth = _squat_depth(env, period)

    sh = _sorted_by_lateral(_bodies_in_yaw_frame(env, shoulder_cfg))   
    el = _sorted_by_lateral(_bodies_in_yaw_frame(env, elbow_cfg))

    v = el - sh                                                        
    fwd = v[:, :, 0] / (torch.linalg.norm(v, dim=-1) + 1e-6)          

    target = (stand_forward + (squat_forward - stand_forward) * depth).unsqueeze(-1)
    err_sq = ((fwd - target) ** 2).mean(dim=-1)
    return torch.exp(-err_sq / (std * std))

def torso_pitch_tracking(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    stand_pitch: float = 0.00,
    squat_pitch: float = 0.65,
    std: float = 0.15,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    depth = _squat_depth(env, period)
    target = torch.sin(torch.as_tensor(stand_pitch, device=depth.device)
                       + (squat_pitch - stand_pitch) * depth)
    err = robot.data.projected_gravity_b[:, 0] - target
    return torch.exp(-(err * err) / (std * std))
