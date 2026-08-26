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
    """静的ディープスクワット保持タスク(周期は一旦諦める)。

    目標:
      1. 膝を深く曲げる (target 1.5 rad)
      2. 胴体を低く (target 0.45 m)
      3. 開脚せず膝で下げる
      4. 転倒しない
    """
    # === 主報酬: 深く曲げる (常時ON, 巨大な weight) ===
    knee_bent   = RewTerm(func=mdp.knee_bent_reward, weight=20.0)
    hip_pitch   = RewTerm(func=mdp.hip_pitch_bent_reward, weight=10.0)
    height_gated = RewTerm(
        func=mdp.height_low_gated_by_knee, weight=15.0,
        params=dict(max_height=0.78, min_height=0.40, knee_gate_min=0.5),
    )

    # === 開脚抑制(緩め) ===
    hip_abduct   = RewTerm(func=mdp.hip_abduction_penalty, weight=1.5)
    hip_roll_mag = RewTerm(func=mdp.hip_roll_magnitude_penalty, weight=1.5)
    feet_width = RewTerm(
        func=mdp.feet_lateral_distance_penalty, weight=3.0,
        params=dict(max_stance_width=0.30, foot_body_names=[FOOT_BODY_REGEX]),
    )
    leg_sym = RewTerm(func=mdp.leg_symmetry_penalty, weight=0.3)

    # === 転倒防止(緩め) ===
    flat_orient = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)

    # === 正則化(最小限) ===
    ang_vel_xy   = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)
    action_rate  = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_acc    = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    dof_pos_lim  = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    arm_default = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1,
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

        self.episode_length_s = 9.0

        # 位相観測
        self.observations.policy.squat_phase = ObsTerm(
            func=mdp.squat_phase_obs,
            params=dict(period=SQUAT_PERIOD),
        )

        # ==== warm start: 半しゃがみ姿勢からリセット ====
        # UNITREE_G1_29DOF_CFG の既存キー形式に合わせて上書き
        # (hip_pitch は完全一致、knee/ankle_pitch は regex キーで登録済み)
        default = dict(self.scene.robot.init_state.joint_pos or {})
        default["left_hip_pitch_joint"] = -0.75
        default["right_hip_pitch_joint"] = -0.75
        default[".*_knee_joint"] = 1.5
        default[".*_ankle_pitch_joint"] = -0.75
        self.scene.robot.init_state.joint_pos = default

        # デバッグ: 実際に反映されたか確認(初回のみ出力される)
        print(">>> warm start joint_pos:", 
              {k: v for k, v in self.scene.robot.init_state.joint_pos.items()
               if "hip_pitch" in k or "knee" in k or "ankle_pitch" in k})


@configclass
class G1PeriodicSquatEnvCfg_PLAY(G1PeriodicSquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False