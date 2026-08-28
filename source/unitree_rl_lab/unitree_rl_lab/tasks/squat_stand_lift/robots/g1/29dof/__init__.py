"""Gym registration for the G1 29DOF SquatStandLift task."""

import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-SquatStandLift",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.tasks.squat_stand_lift.robots.g1.29dof.squat_stand_lift_env_cfg:G1SquatStandLiftEnvCfg",
        "play_env_cfg_entry_point": "unitree_rl_lab.tasks.squat_stand_lift.robots.g1.29dof.squat_stand_lift_env_cfg:G1SquatStandLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.squat_stand_lift.agents.rsl_rl_ppo_cfg:G1SquatStandLiftPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-SquatStandLift-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "play_env_cfg_entry_point": "unitree_rl_lab.tasks.squat_stand_lift.robots.g1.29dof.squat_stand_lift_env_cfg:G1SquatStandLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.squat_stand_lift.agents.rsl_rl_ppo_cfg:G1SquatStandLiftPPORunnerCfg",
    },
)
