"""周期スクワットタスクの観測項。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def squat_phase_obs(
    env: "ManagerBasedRLEnv",
    period: float = 3.0,
    phase_offset: float = 0.0,
) -> torch.Tensor:
    phi = (((env.episode_length_buf * env.step_dt) % period) / period + phase_offset) % 1.0
    return torch.stack([torch.sin(2 * math.pi * phi), torch.cos(2 * math.pi * phi)], dim=-1)

