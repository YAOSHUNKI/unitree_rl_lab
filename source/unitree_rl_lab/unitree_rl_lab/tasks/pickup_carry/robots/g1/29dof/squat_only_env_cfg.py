from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.pickup_carry.mdp as mdp
from .pickup_carry_env_cfg import G1PickupCarryEnvCfg, FOOT_BODY_REGEX

SQUAT_PERIOD = 3.0   # 秒。1周期で「立ち→しゃがみ→立ち」


@configclass
class PeriodicSquatRewardsCfg:
    # --- 周期的スクワット(主報酬) ---
    periodic_height = RewTerm(
        func=mdp.periodic_squat_height, weight=10.0,
        params=dict(period=SQUAT_PERIOD, stand_height=0.75, squat_height=0.50, std=0.06),
    )
    periodic_knee = RewTerm(
        func=mdp.periodic_knee_bend, weight=5.0,
        params=dict(period=SQUAT_PERIOD, stand_knee=0.1, squat_knee=1.3, std=0.25),
    )
    no_freeze = RewTerm(
        func=mdp.squat_motion_penalty, weight=1.0,
        params=dict(freeze_penalty=0.5),
    )

    # --- 姿勢の質 ---
    feet_width = RewTerm(
        func=mdp.feet_lateral_distance_penalty, weight=1.0,
        params=dict(max_stance_width=0.30, foot_body_names=[FOOT_BODY_REGEX]),
    )
    hip_abduct = RewTerm(func=mdp.hip_abduction_penalty, weight=0.5)
    leg_sym    = RewTerm(func=mdp.leg_symmetry_penalty, weight=0.3)

    # --- 転倒防止 ---
    flat_orient = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)

    # --- 正則化(弱め) ---
    lin_vel_z    = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1)   # ← しゃがみ中の上下動を許すため弱く
    ang_vel_xy   = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02)
    action_rate  = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_acc    = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    dof_pos_lim  = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    arm_default = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.5,
        params=dict(
            asset_cfg=SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]
            )
        ),
    )


@configclass
class G1PeriodicSquatEnvCfg(G1PickupCarryEnvCfg):
    rewards: PeriodicSquatRewardsCfg = PeriodicSquatRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 速度指令ゼロ固定
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0

        # 周期の整数倍にするとカリキュラム的に扱いやすい(9秒 = 3周期)
        self.episode_length_s = 9.0

        # 位相を観測に加える
        self.observations.policy.squat_phase = ObsTerm(
            func=mdp.squat_phase_obs,
            params=dict(period=SQUAT_PERIOD),
        )


@configclass
class G1PeriodicSquatEnvCfg_PLAY(G1PeriodicSquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False