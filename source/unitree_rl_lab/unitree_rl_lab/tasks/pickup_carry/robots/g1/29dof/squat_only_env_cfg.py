from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.pickup_carry.mdp as mdp
from .pickup_carry_env_cfg import (
    G1PickupCarryEnvCfg,
    FOOT_BODY_REGEX,
)


@configclass
class SquatOnlyRewardsCfg:
    # --- コア: しゃがみ ---
    squat = RewTerm(
        func=mdp.squat_when_near_box, weight=5.0,   # ← 大きめ
        params=dict(target_height=0.55, near_threshold=10.0, std=0.08),
        # near_threshold を 10m にして「常時ON」に。箱位置に関係なくしゃがませる
    )

    # --- 姿勢の質: 膝で下げる、脚を開かない ---
    knee_flex = RewTerm(
        func=mdp.knee_flexion_when_squatting, weight=3.0,
        params=dict(target_knee_angle=1.2, std=0.3, near_threshold=10.0),
    )
    feet_width = RewTerm(
        func=mdp.feet_lateral_distance_penalty, weight=3.0,
        params=dict(max_stance_width=0.28, foot_body_names=[FOOT_BODY_REGEX]),
    )
    hip_abduct = RewTerm(func=mdp.hip_abduction_penalty, weight=1.0)
    leg_sym = RewTerm(func=mdp.leg_symmetry_penalty, weight=0.5)

    # --- 生存と最低限の正則化 ---
    alive = RewTerm(func=mdp.is_alive, weight=1.0)          # 大きめに(倒れないでいる)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_pos_lim = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    # 腕はデフォルト姿勢に固定(邪魔しないように強めに)
    arm_default = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.5,   
        params=dict(
            asset_cfg=SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]
            )
        ),
    )


@configclass
class G1SquatOnlyEnvCfg(G1PickupCarryEnvCfg):
    rewards: SquatOnlyRewardsCfg = SquatOnlyRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 速度指令はゼロに固定(移動学習を止める)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0

        # エピソードを短くしてサンプル効率UP
        self.episode_length_s = 6.0


@configclass
class G1SquatOnlyEnvCfg_PLAY(G1SquatOnlyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False