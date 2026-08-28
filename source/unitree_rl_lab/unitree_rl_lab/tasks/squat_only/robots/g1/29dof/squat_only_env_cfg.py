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

import unitree_rl_lab.tasks.squat_only.mdp as mdp
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
STAND_HIP_PITCH, SQUAT_HIP_PITCH = -0.10, -2.10
STAND_KNEE,      SQUAT_KNEE      = 0.30, 2.20    # 126 度
STAND_ANKLE,     SQUAT_ANKLE     = -0.20, -0.75  # soft 限界 -0.803 に余裕を残す

# hip_pitch + knee + ankle_pitch が胴の前傾角を決める:
#   立ち  : -0.10 + 0.30 - 0.20 =  0.00  -> 垂直
#   完全しゃがみ: -2.10 + 2.20 - 0.75 = -0.65  -> 前傾 37 度
# 深いスクワットは前傾しないと重心が踵より後ろに抜けて転ぶ。

# 骨盤高さ = 0.3*cos(ankle) + 0.3*cos(knee - ankle) + 0.13
#   knee 0.30 -> 0.73 m
#   knee 2.20 -> 0.39 m  (股関節は足首の 8cm 後ろ。前傾37度で重心を戻す)
STAND_HEIGHT, SQUAT_HEIGHT = 0.73, 0.39

# === 腕の参照姿勢 ===========================================================
# 腕の姿勢は次の3つだけで完全に決まる:
#   1. 向き   arm_forward   (肩 -> 手 の単位ベクトルの前方成分)
#   2. 伸展   arm_ext_pen   (肩/肘/手 の3点の一直線度)
#   3. 間隔   hands_width   (左右の手の距離 = 膝幅)
# いずれもスケールフリーなので、腕の長さや肩の高さを知らなくてよい。
# 手の「絶対位置」を別途指定すると腕長の推定値に依存し、腕を前に振ると
# 手が必然的に上がる分だけ arm_forward と逆方向に引っ張り合うので指定しない。

# 上腕(肩->肘)の前方成分。0=真下, 0.95=鉛直から72度前, 1.0=水平前方
# 0.95 なら腕が伸びていれば手は肩の約11cm下 = 胸の高さに来る。
# 0.55 では手の到達点が膝とほぼ同座標になり、腕が膝にめり込む。
STAND_ARM_FWD,  SQUAT_ARM_FWD  = 0.00,  0.95

# 肩・肘の関節目標 (MuJoCo モデル deploy/mujoco_py/g1_model/g1_29dof.xml から実測)
#   shoulder_pitch: 負が前方。0.194 で真下。-0.45 は前傾37度と合わせて world fwd 0.962
#   elbow         : 0 は 73 度曲がった姿勢。1.276 で完全伸展 (デフォルト 0.97 は 17.7 度)
STAND_SHOULDER_PITCH, SQUAT_SHOULDER_PITCH = 0.20, -0.45
STAND_ELBOW,          SQUAT_ELBOW          = 0.97,  1.25
ARM_FWD_MIN = 0.85          # これを下回ると arm_shortfall_pen で大幅減点

# 胴の前傾 [rad]。股関節が足首の真上に来る条件は knee = 2 x |ankle|。
# ankle の soft 限界が 0.803 なので、踵接地のまま股関節を足の上に保てるのは
# knee <= 1.61 まで。それより深いと股関節は必ず後ろへ抜けるので前傾で戻す。
# 採用値での重心: COM_x +0.046 / 踵余裕 0.106 / つま先余裕 0.104 (ほぼ均衡)
TORSO_STAND_PITCH, TORSO_SQUAT_PITCH = 0.00, 0.65

HAND_WIDTH_SCALE = 1.0                        # 手の間隔 = 膝の間隔 x これ
HAND_WIDTH_MIN   = 0.16                       # 立ち位相でも最低これだけ開く [m]

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
# waist_pitch は専用項 waist_pitch_pen が担当するのでここには入れない
# (グループ平均に混ぜるとシグナルが薄まるため)。
# hip_roll(開脚) はここに入れると平均で薄まるので専用項 HIP_ROLL_CFG に分離した。
LATERAL_CFG   = SceneEntityCfg("robot", joint_names=[
    ".*_hip_yaw_joint", "waist_yaw_joint", "waist_roll_joint",
])
HIP_ROLL_CFG  = SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])
FEET_BODY_CFG   = SceneEntityCfg("robot", body_names=[FOOT_BODY_REGEX])
HAND_BODY_CFG   = SceneEntityCfg("robot", body_names=[HAND_BODY_REGEX])
KNEE_BODY_CFG   = SceneEntityCfg("robot", body_names=[".*_knee_link"])
SHOULDER_CFG    = SceneEntityCfg("robot", body_names=[".*_shoulder_yaw_link"])
ELBOW_CFG       = SceneEntityCfg("robot", body_names=[".*_elbow_link"])
WAIST_PITCH_CFG = SceneEntityCfg("robot", joint_names=["waist_pitch_joint"])
WRIST_CFG       = SceneEntityCfg("robot", joint_names=[".*_wrist_.*_joint"])
SH_PITCH_CFG    = SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint"])
SH_YAW_CFG      = SceneEntityCfg("robot", joint_names=[".*_shoulder_yaw_joint"])
ELBOW_JOINT_CFG = SceneEntityCfg("robot", joint_names=[".*_elbow_joint"])
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


# 腕の正報酬を開ける「膝の曲がり」ゲート。
# しゃがまずに腕だけ伸ばして稼ぐ局所解を塞ぐ (落とし穴 13)。
GATE_KNEE = 1.20            # 目標 SQUAT_KNEE=2.20 の約半分で満開
GATE_MIN  = 0.30            # 立っていても残す最低倍率 (落とし穴 15)
_GATE_PARAMS = dict(
    knee_gate_cfg=KNEE_CFG, gate_stand_knee=STAND_KNEE, gate_knee=GATE_KNEE,
    gate_min=GATE_MIN,
)

_ARM_PARAMS = dict(
    period=SQUAT_PERIOD,
    stand_shoulder_pitch=STAND_SHOULDER_PITCH, squat_shoulder_pitch=SQUAT_SHOULDER_PITCH,
    stand_elbow=STAND_ELBOW,                   squat_elbow=SQUAT_ELBOW,
    shoulder_pitch_cfg=SH_PITCH_CFG,
    elbow_cfg=ELBOW_JOINT_CFG,
    shoulder_yaw_cfg=SH_YAW_CFG,
    **_GATE_PARAMS,
)


@configclass
class PeriodicSquatRewardsCfg:
    # ================= 正報酬: タスク達成 (最大 25.0) =================
    # 姿勢追従は 2段構え。
    #   coarse(std 0.85): 目標から遠くても勾配が残る -> 学習初期の誘導
    #   fine  (std 0.35): 精度を出さないと入らない   -> 収束後の精度
    pose_coarse = RewTerm(
        func=mdp.squat_pose_tracking, weight=5.0,
        params=dict(std=1.80, **_POSE_PARAMS),
    )
    pose_fine = RewTerm(
        func=mdp.squat_pose_tracking, weight=8.0,
        params=dict(std=0.35, **_POSE_PARAMS),
    )
    height_track = RewTerm(
        func=mdp.squat_height_tracking, weight=3.0,
        params=dict(
            period=SQUAT_PERIOD, std=0.24,
            stand_height=STAND_HEIGHT, squat_height=SQUAT_HEIGHT,
            robot_cfg=SceneEntityCfg("robot"),
        ),
    )
    # 胴の前傾 (重心を踵より前に保つ要。崩れると後方転倒する)
    torso_pitch = RewTerm(
        func=mdp.torso_pitch_tracking, weight=3.0,
        params=dict(
            period=SQUAT_PERIOD, std=0.40,
            stand_pitch=TORSO_STAND_PITCH, squat_pitch=TORSO_SQUAT_PITCH,
            robot_cfg=SceneEntityCfg("robot"),
        ),
    )
    # 腕を前方へ振る (肩関節を動かす動機。位置目標より勾配が素直)
    # 2段構え。目標を 0.95 まで上げたことで、腕が真下のときの誤差が
    # 0.95 に達し、狭い std だけでは勾配が完全に消える (1e-28 桁)。
    # coarse が遠方からの誘導を、fine が精度を担当する。
    # 肩・肘の関節目標を直接追従 (腕の主報酬)。coarse+fine の2段構え。
    arm_pose_coarse = RewTerm(
        func=mdp.arm_pose_tracking, weight=5.0,
        params=dict(std=0.60, **_ARM_PARAMS),
    )
    arm_pose_fine = RewTerm(
        func=mdp.arm_pose_tracking, weight=8.0,
        params=dict(std=0.25, **_ARM_PARAMS),
    )
    # world 座標での向きの確認 (関節目標だけでは胴の傾き次第で向きがずれる)
    arm_forward = RewTerm(
        func=mdp.arm_forward_direction, weight=3.0,
        params=dict(
            **_GATE_PARAMS,
            period=SQUAT_PERIOD, std=0.30,
            stand_forward=STAND_ARM_FWD, squat_forward=SQUAT_ARM_FWD,
            shoulder_cfg=SHOULDER_CFG, elbow_cfg=ELBOW_CFG,
        ),
    )
    # 両手の間隔を「膝幅」に合わせる
    # (これが無いと hands_sym_pen だけでは両手が中央で重なっても満点になる)
    hands_width = RewTerm(
        func=mdp.hands_width_match, weight=1.0,
        params=dict(
            width_scale=HAND_WIDTH_SCALE, min_width=HAND_WIDTH_MIN, std=0.06,
            hand_cfg=HAND_BODY_CFG, knee_cfg=KNEE_BODY_CFG, **_GATE_PARAMS,
        ),
    )
    # 合計を正に保つ床 (転倒判定は終了条件が担当するので小さく)
    upright = RewTerm(
        func=mdp.upright_bonus, weight=3.0,
        params=dict(robot_cfg=SceneEntityCfg("robot")),
    )
    grounded = RewTerm(
        func=mdp.feet_grounded, weight=3.0,
        params=dict(sensor_cfg=FOOT_SENSOR_CFG, force_threshold=1.0),
    )

    # ================= ペナルティ: 崩れるとマイナス (最小 -19.5) =================
    # すべて値域 [-1, 0]。静止・対称・正面向きなら 0 なので「タダ取り」できない。
    drift_pen = RewTerm(
        func=mdp.drift_penalty, weight=1.0,
        params=dict(std=0.25, robot_cfg=SceneEntityCfg("robot")),
    )
    slip_pen = RewTerm(
        func=mdp.feet_slip_penalty, weight=1.0,
        params=dict(std=0.30, force_threshold=1.0,
                    sensor_cfg=FOOT_SENSOR_CFG, asset_cfg=FEET_BODY_CFG),
    )
    # 手が左右非対称だとマイナス (中心が揃っているか。間隔は hands_width が担当)
    hands_sym_pen = RewTerm(
        func=mdp.hands_symmetry_penalty, weight=0.5,
        params=dict(std=0.10, hand_cfg=HAND_BODY_CFG),
    )
    # しゃがみ切った時に肘が曲がっていたらマイナス (立ち位相では罰しない)
    arm_ext_pen = RewTerm(
        func=mdp.arm_extension_penalty, weight=1.5,
        params=dict(
            period=SQUAT_PERIOD, min_straightness=0.97, std=0.10,
            shoulder_cfg=SHOULDER_CFG, elbow_cfg=ELBOW_CFG, hand_cfg=HAND_BODY_CFG,
        ),
    )
    # しゃがみ切りで腕が前方に出ていなければ大幅減点 (正報酬だけでは
    # 「取らなくても損しない」ため、必須要件はコスト側にも置く)
    arm_shortfall_pen = RewTerm(
        func=mdp.arm_forward_shortfall_penalty, weight=4.0,
        params=dict(
            period=SQUAT_PERIOD, min_forward=ARM_FWD_MIN, std=0.80,
            shoulder_cfg=SHOULDER_CFG, elbow_cfg=ELBOW_CFG,
        ),
    )
    # しゃがみが浅ければ大幅減点 (腕と同じく必須要件をコスト側にも置く)
    squat_shortfall_pen = RewTerm(
        func=mdp.squat_depth_shortfall_penalty, weight=3.0,
        params=dict(
            period=SQUAT_PERIOD, stand_knee=STAND_KNEE, squat_knee=SQUAT_KNEE,
            min_ratio=0.85, std=0.90, knee_cfg=KNEE_CFG,
        ),
    )
    # 手が膝にめり込んだらマイナス
    knee_clear_pen = RewTerm(
        func=mdp.hands_knee_clearance_penalty, weight=4.0,
        params=dict(
            period=SQUAT_PERIOD, min_distance=0.18, std=0.08,
            hand_cfg=HAND_BODY_CFG, knee_cfg=KNEE_BODY_CFG,
        ),
    )
    # 胴を反らせたらマイナス (骨盤基準の projected_gravity では検出できない)
    # 胴を後ろに反らす (torso_pitch は _relative_track なので反っても 0 点止まりで無罰)
    backlean_pen = RewTerm(
        func=mdp.torso_backlean_penalty, weight=3.0,
        params=dict(margin=0.10, std=0.15, robot_cfg=SceneEntityCfg("robot")),
    )
    waist_pitch_pen = RewTerm(
        func=mdp.waist_pitch_penalty, weight=4.0,
        params=dict(max_abs=0.10, std=0.12, robot_cfg=WAIST_PITCH_CFG),
    )
    # 胴が左右に傾いたらマイナス (前傾ピッチは罰しない)
    torso_roll_pen = RewTerm(
        func=mdp.torso_roll_penalty, weight=1.0,
        params=dict(std=0.25, robot_cfg=SceneEntityCfg("robot")),
    )
    # 胴が正面からヨー方向にずれたらマイナス
    heading_pen = RewTerm(
        func=mdp.heading_penalty, weight=2.0,
        params=dict(robot_cfg=SceneEntityCfg("robot")),
    )
    # ang_vel_xy_l2 は z (ヨー) を見ないため、その場回転がほぼ無料だった。
    yaw_rate_pen = RewTerm(
        func=mdp.yaw_rate_penalty, weight=2.0,
        params=dict(std=0.50, robot_cfg=SceneEntityCfg("robot")),
    )
    speed_pen = RewTerm(
        func=mdp.base_speed_penalty, weight=0.5,
        params=dict(std=0.40, robot_cfg=SceneEntityCfg("robot")),
    )
    # --- 開脚抑制 (深いスクワットでは開脚が「安い抜け道」になるので強く) ---
    hip_abduction_pen = RewTerm(
        func=mdp.hip_abduction_tracking, weight=3.0,
        params=dict(
            period=SQUAT_PERIOD, std=0.12,
            stand_abduction=STAND_ABDUCTION, squat_abduction=SQUAT_ABDUCTION,
            robot_cfg=HIP_ROLL_CFG,
        ),
    )
    stance_pen = RewTerm(
        func=mdp.stance_width_penalty_phased, weight=2.5,
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
    # 手首を中立に固定。肩の勾配が弱いと方策は手首をひねって
    # 手先位置の項を満たそうとするので、その逃げ道を塞ぐ。
    wrist_pen = RewTerm(
        func=mdp.wrist_neutral_penalty, weight=1.5,
        params=dict(period=SQUAT_PERIOD, max_abs=0.15, std=0.25, robot_cfg=WRIST_CFG),
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
        print(f"    arm    上腕の前方成分 {STAND_ARM_FWD} -> {SQUAT_ARM_FWD} (下限 {ARM_FWD_MIN})")
        print(f"    torso  前傾 {TORSO_STAND_PITCH} -> {TORSO_SQUAT_PITCH} rad")


@configclass
class G1PeriodicSquatEnvCfg_PLAY(G1PeriodicSquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
