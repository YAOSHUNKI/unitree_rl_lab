"""Reward terms for the "squat before a box" task.

The desired behaviour is a three-phase policy:
  1. Walk toward the box.
  2. Stop in front of it, facing it, at a target stand-off distance.
  3. Lower the base (squat) while remaining stable.

Rewards are designed to be **gated**: shaping terms for later phases only fire
once earlier-phase preconditions are met, so the agent cannot squat in the
middle of the arena to farm reward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box_relative_xy(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg):
    box: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    rel_w = box.data.root_pos_w - robot.data.root_pos_w
    rel_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel_w)
    return rel_b[:, 0], rel_b[:, 1]


# ---------------------------------------------------------------------------
# Phase 1: approach
# ---------------------------------------------------------------------------


def approach_box(
    env: "ManagerBasedRLEnv",
    target_distance: float = 0.5,
    std: float = 0.4,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian-shaped reward for being at ``target_distance`` from the box (xy).

    Positive when the base->box distance approaches ``target_distance``;
    encourages walking up but not colliding.
    """
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    err = dist - target_distance
    return torch.exp(-(err * err) / (std * std))


# ---------------------------------------------------------------------------
# Phase 2: face the box
# ---------------------------------------------------------------------------


def face_box(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for pointing the robot's +x axis at the box (in base frame).

    Returns cos(theta) where theta is the yaw angle between the base heading
    and the direction to the box. Range: [-1, 1].
    """
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    # Unit vector to box in base frame == (cos(theta), sin(theta))
    return dx / dist


# ---------------------------------------------------------------------------
# Phase 3: squat when close
# ---------------------------------------------------------------------------


def squat_when_near_box(
    env: "ManagerBasedRLEnv",
    target_height: float = 0.55,
    stand_height: float = 0.78,
    near_threshold: float = 0.7,
    std: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for lowering its base height *only when near the box*.

    * Far from the box (> near_threshold): 0 reward (agent should stand and walk).
    * Near the box: Gaussian around ``target_height``, so a full squat pays out.

    The gate keeps the agent from squatting in place forever; it must first
    satisfy phases 1 and 2.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    near = (dist < near_threshold).float()

    h = robot.data.root_pos_w[:, 2]
    err = h - target_height
    squat_score = torch.exp(-(err * err) / (std * std))

    # Also penalize *height above standing* by clipping (agent shouldn't jump).
    _ = stand_height  # kept for documentation; not used in the gaussian
    return near * squat_score


def hold_still_when_squatting(
    env: "ManagerBasedRLEnv",
    near_threshold: float = 0.7,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty on base velocity when close to the box (must be steady in squat)."""
    robot: Articulation = env.scene[robot_cfg.name]
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    near = (dist < near_threshold).float()

    lin_speed = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
    ang_speed = torch.abs(robot.data.root_ang_vel_b[:, 2])
    return -near * (lin_speed + 0.5 * ang_speed)


# ---------------------------------------------------------------------------
# Safety: don't ram the box
# ---------------------------------------------------------------------------


def box_collision_penalty(
    env: "ManagerBasedRLEnv",
    min_distance: float = 0.35,
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Negative reward when the base gets closer than ``min_distance`` (xy)."""
    dx, dy = _box_relative_xy(env, object_cfg, robot_cfg)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    intrusion = (min_distance - dist).clamp(min=0.0)
    return -intrusion
