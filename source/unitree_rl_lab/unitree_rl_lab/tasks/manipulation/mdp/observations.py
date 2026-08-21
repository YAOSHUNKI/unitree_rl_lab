"""Observation terms for the "squat before a box" task.

The policy receives:
  * a low-resolution depth image from a head-mounted RealSense-like camera
    (Isaac Lab TiledCamera, `data_types=["depth"]`), flattened to a vector so
    the standard rsl_rl MLP actor-critic can consume it.
  * proprioceptive observations (defined in the env cfg using the standard
    `mdp.*` terms).

The critic additionally receives the box's pose relative to the robot base
as privileged information (asymmetric actor-critic).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Camera observation
# ---------------------------------------------------------------------------


def depth_image_flat(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("head_camera"),
    clip: tuple[float, float] = (0.1, 3.0),
) -> torch.Tensor:
    """Flatten TiledCamera depth into a 1-D observation per env.

    The RealSense D435 gives depth in metres; we mirror that with a physical
    clip range and normalize to [0, 1] so that the MLP does not have to learn
    huge input scales.

    Returns:
        Tensor of shape ``(num_envs, H*W)``.
    """
    sensor: TiledCamera = env.scene.sensors[sensor_cfg.name]
    # TiledCamera.data.output["depth"] has shape (N, H, W, 1)
    depth = sensor.data.output["depth"].clone()
    # Replace inf / nan (sky / no-hit pixels) with the far clip so the network
    # sees a bounded value instead of garbage.
    depth = torch.nan_to_num(depth, nan=clip[1], posinf=clip[1], neginf=clip[1])
    depth = depth.clamp(min=clip[0], max=clip[1])
    depth = (depth - clip[0]) / (clip[1] - clip[0])
    return depth.reshape(depth.shape[0], -1)


def rgb_image_flat(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("head_camera"),
) -> torch.Tensor:
    """Flatten TiledCamera RGB into a 1-D observation (optional).

    Only enable this if you switch the actor to a CNN backbone — the raw
    flattened tensor grows very fast with resolution.
    """
    sensor: TiledCamera = env.scene.sensors[sensor_cfg.name]
    rgb = sensor.data.output["rgb"].clone().float() / 255.0
    return rgb.reshape(rgb.shape[0], -1)


# ---------------------------------------------------------------------------
# Privileged observations for the critic
# ---------------------------------------------------------------------------


def box_position_in_base_frame(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the box origin expressed in the robot base frame (yaw only)."""
    box: RigidObject = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]

    box_pos_w = box.data.root_pos_w
    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w

    rel_w = box_pos_w - base_pos_w
    rel_b = quat_apply_inverse(yaw_quat(base_quat_w), rel_w)
    return rel_b


def box_distance_and_heading(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Scalar features that summarize the box pose: (distance_xy, sin, cos).

    Cheap, low-dim, works even without the camera — useful as a "no-vision"
    ablation or to bootstrap training.
    """
    rel_b = box_position_in_base_frame(env, object_cfg, robot_cfg)
    dx, dy = rel_b[:, 0], rel_b[:, 1]
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    yaw = torch.atan2(dy, dx)
    return torch.stack([dist, torch.sin(yaw), torch.cos(yaw)], dim=-1)
