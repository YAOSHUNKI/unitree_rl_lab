"""その場スクワット(立ち <-> 完全しゃがみ)の学習環境 v3。

目標動作:
  - 膝を限界近くまで曲げきる (knee 2.4 rad ~ 137 度)
  - しゃがむにつれて両手を左右対称に前へ出す
  - 胴は正面を向いたまま (ヨー・ロールは 0、前傾ピッチのみ許可)
  - その場から動かない

設計原則:
  1. 正報酬 = タスク達成のみ。定位置保持はペナルティで与える
     (棒立ちでタダでもらえる報酬を作らない)。
  2. ペナルティはすべて [-1, 0] に有界。合計が負に振れると
     エージェントは早期終了(わざと転倒)で return を最大化する。
  3. 姿勢追従は coarse(広い std) + fine(狭い std) の2段構え。
     coarse が遠方からの勾配を供給し、fine が精度を要求する。
  4. 「手を前に出す」は関節角ではなくタスク空間(手の位置)で指定。
     shoulder_pitch の符号規約に依存しないため。
  5. SceneEntityCfg は必ず RewTerm(params=...) に書く。
     デフォルト引数に置くと resolve されず全29関節を指してしまう。
"""

from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.pickup_carry.mdp as mdp
from .pickup_carry_env_cfg import (
    G1PickupCarryEnvCfg,
    FOOT_BODY_REGEX,
    HAND_BODY_REGEX,
)

# === 周期 ===================================================================
SQUAT_PERIOD = 6.0        # 秒 / 1周期 (3秒かけて沈み、3秒かけて立つ)

# === 脚の参照姿勢 ===========================================================
# G1 29DOF の soft 関節限界 (soft_joint_pos_limit_factor = 0.9):
#   knee        : 2.73   (物理 2.880)
#   ankle_pitch : -0.803 (物理 -0.873)
#   hip_pitch   : -2.26  (物理 -2.531)
# 下記はすべて soft 限界の内側。
STAND_HIP_PITCH, SQUAT_HIP_PITCH = -0.10, -1.95
STAND_KNEE,      SQUAT_KNEE      = 0.30, 2.40    # 137 度 = ほぼ曲げきり
STAND_ANKLE,     SQUAT_ANKLE     = -0.20, -0.80  # soft 限界 -0.803 ぎりぎり

# hip_pitch + knee + ankle_pitch が胴の前傾角を決める:
#   立ち  : -0.10 + 0.30 - 0.20 =  0.00  -> 垂直
#   完全しゃがみ: -1.95 + 2.40 - 0.80 = -0.35  -> 前傾 20 度
# 深いスクワットは前傾しないと重心が踵より後ろに抜けて転ぶ。

# 骨盤高さ = 0.3*cos(ankle) + 0.3*cos(knee - ankle) + 0.13
#   knee 0.30 -> 0.73 m
#   knee 2.40 -> 0.33 m  (大腿がほぼ水平 = 完全しゃがみ)
STAND_HEIGHT, SQUAT_HEIGHT = 0.73, 0.33

# === 手の参照位置 (骨盤原点・ヨー座標系) ====================================
# x = 前方 / z = 上下。関節角ではなく位置で指定するので符号規約に依存しない。
# NOTE: STAND_* は G1 の腕の自然位置の推定値。PLAY で実測して合わせると精度が上がる。
STAND_HAND_X, SQUAT_HAND_X = 0.05, 0.35   # しゃがむと手が前に出る
STAND_HAND_Z, SQUAT_HAND_Z = -0.15, -0.20  # 同時に下がる (骨盤0.33m時 -> 地上0.13m)

# === 開脚の許容量 ===========================================================
# 完全に 0 は非現実的 (大腿が水平近くまで来ると胴の入るスペースが要る)。
# 深さ相応のわずかな開きだけ許し、それを超えたら強く罰する。
STAND_ABDUCTION, SQUAT_ABDUCTION = 0.00, 0.18   # |hip_roll| [rad] (約10度)
STAND_WIDTH,     SQUAT_WIDTH     = 0.20, 0.28   # 足の左右間隔 [m]

# === SceneEntityCfg (必ず params で渡す) ====================================
HIP_PITCH_CFG = SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint"])
KNEE_CFG      = SceneEntityCfg("robot", joint_names=[".*_knee_joint"])
ANKLE_CFG     = SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"])
# 0 に固定したい関節: 脚のねじれ(hip_yaw)・胴のねじれと横傾き(waist)
# waist_pitch は前傾に使うので入れない。
# hip_roll(開脚) はここに入れると平均で薄まるので専用項 HIP_ROLL_CFG に分離した。
LATERAL_CFG   = SceneEntityCfg("robot", joint_names=[
    ".*_hip_yaw_joint", "waist_yaw_joint", "waist_roll_joint",
])
HIP_ROLL_CFG  = SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])
FEET_BODY_CFG   = SceneEntityCfg("robot", body_names=[FOOT_BODY_REGEX])
HAND_BODY_CFG   = SceneEntityCfg("robot", body_names=[HAND_BODY_REGEX])
FOOT_SENSOR_CFG = SceneEntityCfg("foot_contact", body_names=[FOOT_BODY_REGEX])

_POSE_PARAMS = dict(
    period=SQUAT_PERIOD,
    stand_hip_pitch=STAND_HIP_PITCH, squat_hip_pitch=SQUAT_HIP_PITCH,
    stand_knee=STAND_KNEE,           squat_knee=SQUAT_KNEE,
    stand_ankle=STAND_ANKLE,         squat_ankle=SQUAT_ANKLE,
    hip_pitch_cfg=HIP_PITCH_CFG,
    knee_cfg=KNEE_CFG,
    ankle_cfg=ANKLE_CFG,
    lateral_cfg=LATERAL_CFG,
)


@configclass
class PeriodicSquatRewardsCfg:
    # ================= 正報酬: タスク達成 (最大 18.0) =================
    # 姿勢追従は 2段構え。
    #   coarse(std 0.85): 目標から遠くても勾配が残る -> 学習初期の誘導
    #   fine  (std 0.35): 精度を出さないと入らない   -> 収束後の精度
    pose_coarse = RewTerm(
        func=mdp.squat_pose_tracking, weight=4.0,
        params=dict(std=0.85, **_POSE_PARAMS),
    )
    pose_fine = RewTerm(
        func=mdp.squat_pose_tracking, weight=8.0,
        params=dict(std=0.35, **_POSE_PARAMS),
    )
    height_track = RewTerm(
        func=mdp.squat_height_tracking, weight=3.0,
        params=dict(
            period=SQUAT_PERIOD, std=0.15,
            stand_height=STAND_HEIGHT, squat_height=SQUAT_HEIGHT,
            robot_cfg=SceneEntityCfg("robot"),
        ),
    )
    # しゃがむにつれて両手を前へ
    hands_forward = RewTerm(
        func=mdp.hands_forward_tracking, weight=3.0,
        params=dict(
            period=SQUAT_PERIOD, std=0.13,
            stand_x=STAND_HAND_X, squat_x=SQUAT_HAND_X,
            stand_z=STAND_HAND_Z, squat_z=SQUAT_HAND_Z,
            hand_cfg=HAND_BODY_CFG,
        ),
    )
    # 合計を正に保つ床 (転倒判定は終了条件が担当するので小さく)
    upright = RewTerm(
        func=mdp.upright_bonus, weight=0.5,
        params=dict(robot_cfg=SceneEntityCfg("robot")),
    )
    grounded = RewTerm(
        func=mdp.feet_grounded, weight=0.5,
        params=dict(sensor_cfg=FOOT_SENSOR_CFG, force_threshold=1.0),
    )

    # ================= ペナルティ: 崩れるとマイナス (最小 -12.0) =================
    # すべて値域 [-1, 0]。静止・対称・正面向きなら 0 なので「タダ取り」できない。
    drift_pen = RewTerm(
        func=mdp.drift_penalty, weight=1.5,
        params=dict(std=0.25, robot_cfg=SceneEntityCfg("robot")),
    )
    slip_pen = RewTerm(
        func=mdp.feet_slip_penalty, weight=1.5,
        params=dict(std=0.15, force_threshold=1.0,
                    sensor_cfg=FOOT_SENSOR_CFG, asset_cfg=FEET_BODY_CFG),
    )
    # 手が左右非対称だとマイナス
    hands_sym_pen = RewTerm(
        func=mdp.hands_symmetry_penalty, weight=1.0,
        params=dict(std=0.10, hand_cfg=HAND_BODY_CFG),
    )
    # 胴が左右に傾いたらマイナス (前傾ピッチは罰しない)
    torso_roll_pen = RewTerm(
        func=mdp.torso_roll_penalty, weight=1.0,
        params=dict(std=0.15, robot_cfg=SceneEntityCfg("robot")),
    )
    # 胴が正面からヨー方向にずれたらマイナス
    heading_pen = RewTerm(
        func=mdp.heading_penalty, weight=1.0,
        params=dict(robot_cfg=SceneEntityCfg("robot")),
    )
    speed_pen = RewTerm(
        func=mdp.base_speed_penalty, weight=0.5,
        params=dict(std=0.30, robot_cfg=SceneEntityCfg("robot")),
    )
    # --- 開脚抑制 (深いスクワットでは開脚が「安い抜け道」になるので強く) ---
    hip_abduction_pen = RewTerm(
        func=mdp.hip_abduction_tracking, weight=6.0,
        params=dict(
            period=SQUAT_PERIOD, std=0.12,
            stand_abduction=STAND_ABDUCTION, squat_abduction=SQUAT_ABDUCTION,
            robot_cfg=HIP_ROLL_CFG,
        ),
    )
    stance_pen = RewTerm(
        func=mdp.stance_width_penalty_phased, weight=5.0,
        params=dict(
            period=SQUAT_PERIOD, std=0.08,
            stand_width=STAND_WIDTH, squat_width=SQUAT_WIDTH,
            robot_cfg=FEET_BODY_CFG,
        ),
    )

    # ================= 正則化 =================
    ang_vel_xy   = RewTerm(func=mdp.ang_vel_xy_l2,     weight=-0.02)
    action_rate  = RewTerm(func=mdp.action_rate_l2,    weight=-0.005)
    joint_acc    = RewTerm(func=mdp.joint_acc_l2,      weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2,  weight=-1.0e-6)
    dof_pos_lim  = RewTerm(func=mdp.joint_pos_limits,  weight=-1.0)
    # 肩は前へ伸ばすので拘束しない。手首だけ暴れないよう弱く抑える。
    wrist_default = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.05,
        params=dict(asset_cfg=SceneEntityCfg("robot", joint_names=[".*_wrist_.*"])),
    )
    # 棒立ち   : 約 5.5 / 18.0
    # 正しいスクワット: 約 11.5 / 18.0
    # 開脚スクワット  : 約 6.0  (開脚ペナルティ -5.5 で相殺)


@configclass
class G1PeriodicSquatEnvCfg(G1PickupCarryEnvCfg):
    rewards: PeriodicSquatRewardsCfg = PeriodicSquatRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- 移動はさせない ---
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0

        self.episode_length_s = 12.0   # 2周期

        # --- 位相観測 (policy が今どの位相か知る必要がある) ---
        self.observations.policy.squat_phase = ObsTerm(
            func=mdp.squat_phase_obs,
            params=dict(period=SQUAT_PERIOD),
        )

        # --- 初期姿勢は参照の phi=0 (立ち) に合わせる ---
        default = dict(self.scene.robot.init_state.joint_pos or {})
        default["left_hip_pitch_joint"]  = STAND_HIP_PITCH
        default["right_hip_pitch_joint"] = STAND_HIP_PITCH
        default[".*_knee_joint"]         = STAND_KNEE
        default[".*_ankle_pitch_joint"]  = STAND_ANKLE
        self.scene.robot.init_state.joint_pos = default

        # --- 転倒は罰ではなく終了で扱う ---
        self.terminations.base_contact = None
        self.terminations.fell_over = DoneTerm(
            func=mdp.bad_orientation,
            params=dict(limit_angle=1.2),          # 約 69 度
        )
        # 完全しゃがみで骨盤 0.33m まで下がるので閾値を下げる
        self.terminations.collapsed = DoneTerm(
            func=mdp.root_height_below_minimum,
            params=dict(minimum_height=0.20,
                        asset_cfg=SceneEntityCfg("robot")),
        )

        print(f">>> PeriodicSquat v3: period={SQUAT_PERIOD}s")
        print(f"    knee   {STAND_KNEE} -> {SQUAT_KNEE} rad")
        print(f"    height {STAND_HEIGHT} -> {SQUAT_HEIGHT} m")
        print(f"    hand x {STAND_HAND_X} -> {SQUAT_HAND_X} m (骨盤基準・前方)")


@configclass
class G1PeriodicSquatEnvCfg_PLAY(G1PeriodicSquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
