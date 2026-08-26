"""Gym registration for the G1 29DOF pickup-and-carry task.

Uses string entry points so gym.register does NOT trigger the omni-dependent
import chain during ``scripts/list_envs.py`` (which runs before Isaac Sim is
launched). The actual modules are loaded lazily when gym.make is called.
"""

import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-PickupCarry",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.tasks.pickup_carry.robots.g1.29dof.pickup_carry_env_cfg:G1PickupCarryEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.pickup_carry.agents.rsl_rl_ppo_cfg:G1PickupCarryPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-PickupCarry-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "play_env_cfg_entry_point": "unitree_rl_lab.tasks.pickup_carry.robots.g1.29dof.pickup_carry_env_cfg:G1PickupCarryEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.pickup_carry.agents.rsl_rl_ppo_cfg:G1PickupCarryPPORunnerCfg",
    },
)

# ----- PeriodicSquat ------------------------------------------------
gym.register(
    id="Unitree-G1-29dof-PeriodicSquat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.tasks.pickup_carry.robots.g1.29dof.squat_only_env_cfg:G1PeriodicSquatEnvCfg",
        "play_env_cfg_entry_point": "unitree_rl_lab.tasks.pickup_carry.robots.g1.29dof.squat_only_env_cfg:G1PeriodicSquatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.pickup_carry.agents.rsl_rl_ppo_cfg:G1PickupCarryPPORunnerCfg",
    },
)
gym.register(
    id="Unitree-G1-29dof-PeriodicSquat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "play_env_cfg_entry_point": "unitree_rl_lab.tasks.pickup_carry.robots.g1.29dof.squat_only_env_cfg:G1PeriodicSquatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.pickup_carry.agents.rsl_rl_ppo_cfg:G1PickupCarryPPORunnerCfg",
    },
)