"""Observation terms for the pickup-and-carry task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import math

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Box pose (base frame)
# ---------------------------------------------------------------------------


def box_position_in_base_frame(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Box origin expressed in the yaw-only base frame. Shape: (N, 3)."""
    box: RigidObject = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]

    rel_w = box.data.root_pos_w - robot.data.root_pos_w
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel_w)


def box_distance_and_heading(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(xy-distance, sin(yaw), cos(yaw)) to the box. Shape: (N, 3)."""
    rel_b = box_position_in_base_frame(env, object_cfg, robot_cfg)
    dx, dy = rel_b[:, 0], rel_b[:, 1]
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    yaw = torch.atan2(dy, dx)
    return torch.stack([dist, torch.sin(yaw), torch.cos(yaw)], dim=-1)


# ---------------------------------------------------------------------------
# Hands
# ---------------------------------------------------------------------------


def hand_positions_in_base_frame(
    env: "ManagerBasedRLEnv",
    hand_body_names: list[str] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Both hand positions in the yaw-only base frame. Shape: (N, 3*K)."""
    robot = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies(list(hand_body_names))
    hands_w = robot.data.body_pos_w[:, body_ids, :]
    base_pos = robot.data.root_pos_w.unsqueeze(1)
    rel_w = hands_w - base_pos
    q = yaw_quat(robot.data.root_quat_w).unsqueeze(1).expand(-1, rel_w.shape[1], -1).reshape(-1, 4)
    rel_b = quat_apply_inverse(q, rel_w.reshape(-1, 3)).reshape(rel_w.shape)
    return rel_b.reshape(rel_b.shape[0], -1)


def box_in_hand_frame(
    env: "ManagerBasedRLEnv",
    hand_body_names: list[str] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vector from each hand to the box (world frame). Shape: (N, 3*K)."""
    robot = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[object_cfg.name]
    body_ids, _ = robot.find_bodies(list(hand_body_names))
    hands_w = robot.data.body_pos_w[:, body_ids, :]
    return (box.data.root_pos_w.unsqueeze(1) - hands_w).reshape(hands_w.shape[0], -1)


def hand_contact_flags(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("hand_contact", body_names=[".*_wrist_yaw_link"]),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """0/1 per hand indicating contact. Shape: (N, K)."""
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    f = torch.linalg.norm(cs.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    return (f > force_threshold).float()

def squat_phase_obs(
    env: "ManagerBasedRLEnv",
    period: float = 3.0,
) -> torch.Tensor:
    phi = ((env.episode_length_buf * env.step_dt) % period) / period
    return torch.stack([torch.sin(2 * math.pi * phi), torch.cos(2 * math.pi * phi)], dim=-1)