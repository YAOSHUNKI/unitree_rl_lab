"""Reset-time events for the pickup-and-carry task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_box_pose_uniform(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = (0.5, 0.9),
    y_range: tuple[float, float] = (-0.2, 0.2),
    z: float = 0.1,
    yaw_range: tuple[float, float] = (-0.5, 0.5),
    object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> None:
    """Sample a fresh box pose in front of each robot at reset.

    The box is placed **relative to each env's origin** (which follows the
    robot's spawn) so on reset it always appears in front of the robot
    within the given ranges.
    """
    box: RigidObject = env.scene[object_cfg.name]
    device = box.data.root_pos_w.device
    n = env_ids.numel()

    def rand(lo, hi):
        return torch.empty(n, device=device).uniform_(lo, hi)

    origins = env.scene.env_origins[env_ids]

    pos = origins.clone()
    pos[:, 0] += rand(*x_range)
    pos[:, 1] += rand(*y_range)
    pos[:, 2] = origins[:, 2] + z

    yaw = rand(*yaw_range)
    zero = torch.zeros_like(yaw)
    quat = torch.stack([torch.cos(yaw * 0.5), zero, zero, torch.sin(yaw * 0.5)], dim=-1)  # (w,x,y,z)

    root_state = box.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] = pos
    root_state[:, 3:7] = quat
    root_state[:, 7:] = 0.0

    box.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    box.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
