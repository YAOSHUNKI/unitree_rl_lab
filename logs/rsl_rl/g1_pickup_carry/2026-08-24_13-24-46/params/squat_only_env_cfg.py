from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.pickup_carry.mdp as mdp
from .pickup_carry_env_cfg import G1PickupCarryEnvCfg, FOOT_BODY_REGEX

SQUAT_PERIOD = 3.0   # 秒。1周期で「立ち→しゃがみ→立ち」

def __post_init__(self):
    super().__post_init__()
    # ... 既存の設定 ...

    # 初期姿勢を半しゃがみに(膝の存在をロボットに知らせる)
    default = dict(self.scene.robot.init_state.joint_pos)
    for k in ["left_hip_pitch_joint", "right_hip_pitch_joint"]:
        default[k] = -0.3
    for k in ["left_knee_joint", "right_knee_joint"]:
        default[k] = 0.6
    for k in ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]:
        default[k] = -0.3
    self.scene.robot.init_state.joint_pos = default

@configclass
class PeriodicSquatRewardsCfg:
    # === 主報酬: 膝を曲げること (常時ON, リニア) ===
    knee_bent  = RewTerm(func=mdp.knee_bent_reward, weight=8.0)
    hip_pitch  = RewTerm(func=mdp.hip_pitch_bent_reward, weight=4.0)

    # === 周期目標 (立ち↔しゃがみの動きを強制) ===
    period_height = RewTerm(
        func=mdp.periodic_height_target, weight=6.0,
        params=dict(period=3.0, stand_height=0.75, squat_height=0.50, std=0.08),
    )
    period_knee = RewTerm(
        func=mdp.periodic_knee_target, weight=4.0,
        params=dict(period=3.0, stand_knee=0.1, squat_knee=1.3),
    )
    period_hip = RewTerm(
        func=mdp.periodic_hip_pitch_target, weight=2.0,
        params=dict(period=3.0, stand_hip=0.0, squat_hip=-0.7),
    )

    # === 高さ報酬 (膝ゲート付き。開脚で下げても報酬ゼロ) ===
    height_gated = RewTerm(
        func=mdp.height_low_gated_by_knee, weight=4.0,
        params=dict(max_height=0.78, min_height=0.40, knee_gate_min=0.5),
    )

    # === 開脚抑制 (多角的) ===
    hip_abduct   = RewTerm(func=mdp.hip_abduction_penalty, weight=5.0)
    hip_roll_mag = RewTerm(func=mdp.hip_roll_magnitude_penalty, weight=5.0)
    feet_width = RewTerm(
        func=mdp.feet_lateral_distance_penalty, weight=10.0,
        params=dict(max_stance_width=0.25, foot_body_names=[FOOT_BODY_REGEX]),
    )
    leg_sym = RewTerm(func=mdp.leg_symmetry_penalty, weight=0.3)

    # === 姿勢 & 足浮き ===
    flat_orient = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    feet_air = RewTerm(
        func=mdp.feet_air_time_penalty, weight=1.0,
        params=dict(
            sensor_cfg=SceneEntityCfg("foot_contact", body_names=[FOOT_BODY_REGEX]),
            grace_period=0.2,
        ),
    )

    # === 正則化 ===
    lin_vel_z    = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1)
    ang_vel_xy   = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02)
    action_rate  = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_acc    = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    dof_pos_lim  = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    arm_default = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.5,
        params=dict(asset_cfg=SceneEntityCfg(
            "robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]
        )),
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