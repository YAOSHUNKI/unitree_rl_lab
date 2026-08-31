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


def _squat_phase(env: "ManagerBasedRLEnv", period: float, phase_offset: float = 0.0) -> torch.Tensor:
    return (((env.episode_length_buf * env.step_dt) % period) / period + phase_offset) % 1.0


def _squat_depth(env: "ManagerBasedRLEnv", period: float, phase_offset: float = 0.0) -> torch.Tensor:
    phi = _squat_phase(env, period, phase_offset)
    return 0.5 - 0.5 * torch.cos(2 * math.pi * phi)


def _relative_track(
    err_sq: torch.Tensor, idle_err_sq: torch.Tensor, std: float, floor: float = 0.25
) -> torch.Tensor:
    base = torch.exp(-idle_err_sq / (std * std))
    raw = torch.exp(-err_sq / (std * std))
    return ((raw - base) / (1.0 - base).clamp(min=floor)).clamp(min=0.0, max=1.0)


def _knee_gate(
    env: "ManagerBasedRLEnv",
    knee_cfg: SceneEntityCfg,
    stand_knee: float = 0.30,
    gate_knee: float = 1.20,
    gate_min: float = 0.30,
) -> torch.Tensor:
    robot: Articulation = env.scene[knee_cfg.name]
    knee = robot.data.joint_pos[:, knee_cfg.joint_ids].mean(dim=-1)
    prog = ((knee - stand_knee) / (gate_knee - stand_knee)).clamp(min=0.0, max=1.0)
    return gate_min + (1.0 - gate_min) * prog



def upright_bonus(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    g_z = robot.data.projected_gravity_b[:, 2] 
    return (1.0 - g_z) * 0.5


def squat_pose_tracking(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    phase_offset: float = 0.0,
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
    depth = _squat_depth(env, period, phase_offset).unsqueeze(-1)  

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

    idle_sq = depth.squeeze(-1) ** 2 * (
        (squat_hip_pitch - stand_hip_pitch) ** 2
        + (squat_knee - stand_knee) ** 2
        + (squat_ankle - stand_ankle) ** 2
    )
    return _relative_track(err_sq, idle_sq, std)


def squat_height_tracking(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    phase_offset: float = 0.0,
    stand_height: float = 0.74,
    squat_height: float = 0.52,
    std: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    depth = _squat_depth(env, period, phase_offset)
    target = stand_height + (squat_height - stand_height) * depth
    err = robot.data.root_pos_w[:, 2] - target
    idle_sq = depth ** 2 * (squat_height - stand_height) ** 2
    return _relative_track(err * err, idle_sq, std)


def feet_grounded(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("foot_contact"),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.linalg.norm(cs.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    return (forces > force_threshold).float().mean(dim=-1)

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
    in_contact = (forces > force_threshold).float()                      

    foot_v = torch.linalg.norm(
        robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1
    )                                                                   

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


def yaw_rate_penalty(
    env: "ManagerBasedRLEnv",
    std: float = 0.50,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    wz = robot.data.root_ang_vel_b[:, 2]
    return torch.exp(-(wz * wz) / (std * std)) - 1.0


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


def _hands_in_yaw_frame(
    env: "ManagerBasedRLEnv",
    hand_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[hand_cfg.name]
    hands_w = robot.data.body_pos_w[:, hand_cfg.body_ids, :]         
    rel_w = hands_w - robot.data.root_pos_w.unsqueeze(1)              

    n_hand = rel_w.shape[1]
    q = yaw_quat(robot.data.root_quat_w).unsqueeze(1).expand(-1, n_hand, -1)
    rel_b = quat_apply_inverse(q.reshape(-1, 4), rel_w.reshape(-1, 3))
    return rel_b.reshape(rel_w.shape)


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
    phase_offset: float = 0.0,
    stand_abduction: float = 0.00,
    squat_abduction: float = 0.18,
    std: float = 0.12,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    depth = _squat_depth(env, period, phase_offset)
    target = stand_abduction + (squat_abduction - stand_abduction) * depth

    abd = robot.data.joint_pos[:, robot_cfg.joint_ids].abs().mean(dim=-1)
    excess = (abd - target).clamp(min=0.0)
    return depth * (torch.exp(-(excess * excess) / (std * std)) - 1.0)


def stance_width_penalty_phased(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    phase_offset: float = 0.0,
    stand_width: float = 0.20,
    squat_width: float = 0.28,
    std: float = 0.08,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    depth = _squat_depth(env, period, phase_offset)
    target = stand_width + (squat_width - stand_width) * depth

    feet_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]    
    rel = feet_w[:, 0, :] - feet_w[:, 1, :]
    rel_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel)
    width = torch.abs(rel_b[:, 1])

    excess = (width - target).clamp(min=0.0)
    return depth * (torch.exp(-(excess * excess) / (std * std)) - 1.0)


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


def hands_width_match(
    env: "ManagerBasedRLEnv",
    width_scale: float = 1.0,
    min_width: float = 0.16,
    std: float = 0.06,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knee_gate_cfg: SceneEntityCfg | None = None,
    gate_stand_knee: float = 0.30,
    gate_knee: float = 1.20,
    gate_min: float = 0.30,
) -> torch.Tensor:
    hands = _bodies_in_yaw_frame(env, hand_cfg)
    knees = _bodies_in_yaw_frame(env, knee_cfg)

    hand_w = torch.abs(hands[:, 0, 1] - hands[:, 1, 1])
    knee_w = torch.abs(knees[:, 0, 1] - knees[:, 1, 1])
    target = (knee_w * width_scale).clamp(min=min_width)

    err = hand_w - target
    r = torch.exp(-(err * err) / (std * std))
    if knee_gate_cfg is not None:
        r = r * _knee_gate(env, knee_gate_cfg, gate_stand_knee, gate_knee, gate_min)
    return r


def _sorted_by_lateral(rel_b: torch.Tensor) -> torch.Tensor:
    idx = torch.argsort(rel_b[:, :, 1], dim=1)
    return torch.gather(rel_b, 1, idx.unsqueeze(-1).expand(-1, -1, 3))


def arm_extension_penalty(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    phase_offset: float = 0.0,
    min_straightness: float = 0.97,
    std: float = 0.06,
    shoulder_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    elbow_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    depth = _squat_depth(env, period, phase_offset)

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
    phase_offset: float = 0.0,
    stand_forward: float = 0.00,
    squat_forward: float = 0.65,
    std: float = 0.15,
    shoulder_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    elbow_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knee_gate_cfg: SceneEntityCfg | None = None,
    gate_stand_knee: float = 0.30,
    gate_knee: float = 1.20,
    gate_min: float = 0.30,
) -> torch.Tensor:
    depth = _squat_depth(env, period, phase_offset)

    sh = _sorted_by_lateral(_bodies_in_yaw_frame(env, shoulder_cfg))   
    el = _sorted_by_lateral(_bodies_in_yaw_frame(env, elbow_cfg))

    v = el - sh                                                        
    fwd = v[:, :, 0] / (torch.linalg.norm(v, dim=-1) + 1e-6)           

    target = (stand_forward + (squat_forward - stand_forward) * depth).unsqueeze(-1)
    err_sq = ((fwd - target) ** 2).mean(dim=-1)
    idle = target.squeeze(-1) - stand_forward
    r = _relative_track(err_sq, idle * idle, std)
    if knee_gate_cfg is not None:
        r = r * _knee_gate(env, knee_gate_cfg, gate_stand_knee, gate_knee, gate_min)
    return r

def torso_pitch_tracking(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    phase_offset: float = 0.0,
    stand_pitch: float = 0.00,
    squat_pitch: float = 0.65,
    std: float = 0.15,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    depth = _squat_depth(env, period, phase_offset)
    target = torch.sin(torch.as_tensor(stand_pitch, device=depth.device)
                       + (squat_pitch - stand_pitch) * depth)
    err = robot.data.projected_gravity_b[:, 0] - target
    idle = target - torch.sin(torch.as_tensor(stand_pitch, device=depth.device))
    return _relative_track(err * err, idle * idle, std)


def torso_backlean_penalty(
    env: "ManagerBasedRLEnv",
    margin: float = 0.10,
    std: float = 0.15,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    gx = robot.data.projected_gravity_b[:, 0]
    deficit = (-margin - gx).clamp(min=0.0)
    return torch.exp(-(deficit * deficit) / (std * std)) - 1.0


def joint_default_deviation_penalty(
    env: "ManagerBasedRLEnv",
    margin: float = 0.15,
    std: float = 0.30,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    q0 = robot.data.default_joint_pos[:, robot_cfg.joint_ids]
    excess = ((q - q0).abs() - margin).clamp(min=0.0).mean(dim=-1)
    return torch.exp(-(excess * excess) / (std * std)) - 1.0


def lateral_offset_penalty(
    env: "ManagerBasedRLEnv",
    std: float = 0.06,
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    rel = _bodies_in_yaw_frame(env, body_cfg)         
    mid_y = rel[:, :, 1].mean(dim=-1)                  
    return torch.exp(-(mid_y * mid_y) / (std * std)) - 1.0

def arm_forward_shortfall_penalty(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    phase_offset: float = 0.0,
    min_forward: float = 0.85,
    std: float = 0.60,
    shoulder_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    elbow_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    depth = _squat_depth(env, period, phase_offset)

    sh = _sorted_by_lateral(_bodies_in_yaw_frame(env, shoulder_cfg))
    el = _sorted_by_lateral(_bodies_in_yaw_frame(env, elbow_cfg))

    v = el - sh
    fwd = v[:, :, 0] / (torch.linalg.norm(v, dim=-1) + 1e-6)          

    threshold = min_forward * depth
    deficit = (threshold.unsqueeze(-1) - fwd).clamp(min=0.0).mean(dim=-1)
    return depth * (torch.exp(-(deficit * deficit) / (std * std)) - 1.0)


def hands_knee_clearance_penalty(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    min_distance: float = 0.18,
    std: float = 0.08,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    hands = _bodies_in_yaw_frame(env, hand_cfg)                       
    knees = _bodies_in_yaw_frame(env, knee_cfg)                       

    d = torch.linalg.norm(hands.unsqueeze(2) - knees.unsqueeze(1), dim=-1)  
    nearest = d.min(dim=-1).values                                    

    deficit = (min_distance - nearest).clamp(min=0.0).mean(dim=-1)
    depth = _squat_depth(env, period)
    return depth * (torch.exp(-(deficit * deficit) / (std * std)) - 1.0)

def waist_pitch_penalty(
    env: "ManagerBasedRLEnv",
    max_abs: float = 0.10,
    std: float = 0.12,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    excess = (q.abs() - max_abs).clamp(min=0.0).mean(dim=-1)
    return torch.exp(-(excess * excess) / (std * std)) - 1.0

def wrist_neutral_penalty(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    max_abs: float = 0.15,
    std: float = 0.25,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    excess = (q.abs() - max_abs).clamp(min=0.0).mean(dim=-1)
    depth = _squat_depth(env, period)
    return depth * (torch.exp(-(excess * excess) / (std * std)) - 1.0)


def arm_pose_tracking(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    std: float = 0.50,
    stand_shoulder_pitch: float = 0.20,
    squat_shoulder_pitch: float = -0.45,
    stand_elbow: float = 0.97,
    squat_elbow: float = 1.25,
    shoulder_pitch_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    elbow_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    shoulder_yaw_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knee_gate_cfg: SceneEntityCfg | None = None,
    gate_stand_knee: float = 0.30,
    gate_knee: float = 1.20,
    gate_min: float = 0.30,
) -> torch.Tensor:
    
    robot: Articulation = env.scene[shoulder_pitch_cfg.name]
    depth = _squat_depth(env, period).unsqueeze(-1)

    err_sq = torch.zeros(env.num_envs, device=robot.data.joint_pos.device)
    for cfg, q_stand, q_squat in (
        (shoulder_pitch_cfg, stand_shoulder_pitch, squat_shoulder_pitch),
        (elbow_cfg, stand_elbow, squat_elbow),
    ):
        target = q_stand + (q_squat - q_stand) * depth
        q = robot.data.joint_pos[:, cfg.joint_ids]
        err_sq = err_sq + ((q - target) ** 2).mean(dim=-1)

    q_yaw = robot.data.joint_pos[:, shoulder_yaw_cfg.joint_ids]
    err_sq = err_sq + (q_yaw ** 2).mean(dim=-1)

    idle_sq = depth.squeeze(-1) ** 2 * (
        (squat_shoulder_pitch - stand_shoulder_pitch) ** 2
        + (squat_elbow - stand_elbow) ** 2
    )
    r = _relative_track(err_sq, idle_sq, std)
    if knee_gate_cfg is not None:
        r = r * _knee_gate(env, knee_gate_cfg, gate_stand_knee, gate_knee, gate_min)
    return r

def squat_depth_shortfall_penalty(
    env: "ManagerBasedRLEnv",
    period: float = 6.0,
    stand_knee: float = 0.30,
    squat_knee: float = 2.20,
    min_ratio: float = 0.85,
    std: float = 0.90,
    knee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[knee_cfg.name]
    depth = _squat_depth(env, period)

    target = stand_knee + (squat_knee - stand_knee) * depth * min_ratio
    knee = robot.data.joint_pos[:, knee_cfg.joint_ids].mean(dim=-1)

    deficit = (target - knee).clamp(min=0.0)
    return depth * (torch.exp(-(deficit * deficit) / (std * std)) - 1.0)

