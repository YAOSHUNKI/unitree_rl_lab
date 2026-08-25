"""その場スクワット(立ち <-> しゃがみ)の学習環境。

設計原則 (これまでの失敗から得た教訓):

 1. 報酬は「すべて正値・有界 [0,1]」にする。
    合計が負になると、エージェントは早期終了(= 転倒・寝転がり)で
    return を最大化しようとする。正値だけなら「生き続ける」が常に最適。

 2. 転倒は「罰」ではなく「終了条件」で扱う。

 3. 脚の姿勢は個別の関節ごとにバラバラの項で要求せず、
    「1本の参照姿勢へのトラッキング」1項にまとめる。
    膝・股・足首・内転が互いに矛盾せず、開脚も自動的に潰れる。

 4. SceneEntityCfg は必ず RewTerm(params=...) に書く。
    関数のデフォルト引数に置いた SceneEntityCfg は resolve されず
    joint_ids が slice(None) のまま = 29関節すべてを指してしまう。
"""

from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.pickup_carry.mdp as mdp
from .pickup_carry_env_cfg import G1PickupCarryEnvCfg, FOOT_BODY_REGEX

# --- チューニングノブ ------------------------------------------------------
SQUAT_PERIOD = 6.0        # 秒 / 1周期 (3秒かけて下がり、3秒かけて上がる)

STAND_HIP_PITCH, SQUAT_HIP_PITCH = -0.10, -1.50
STAND_KNEE,      SQUAT_KNEE      = 0.30, 2.00
STAND_ANKLE,     SQUAT_ANKLE     = -0.20, -0.70
# hip_pitch + knee + ankle_pitch が胴体ピッチを決める:
#   立ち  : -0.10 + 0.30 - 0.20 =  0.00  -> 垂直
#   しゃがみ: -1.50 + 2.00 - 0.70 = -0.20  -> 前傾 11.5 度
# 深いスクワットは前傾しないと重心が後ろに抜けて転ぶので意図的に負にしている。

# 骨盤高さ = 0.6*cos(knee/2) + 0.13  (大腿 0.30m / 下腿 0.30m の運動学から)
#   knee 0.30 -> 0.73 m
#   knee 2.00 -> 0.45 m
STAND_HEIGHT, SQUAT_HEIGHT = 0.73, 0.45
# ---------------------------------------------------------------------------

# SceneEntityCfg は必ず params で渡す (下で使い回す)
HIP_PITCH_CFG = SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint"])
KNEE_CFG      = SceneEntityCfg("robot", joint_names=[".*_knee_joint"])
ANKLE_CFG     = SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"])
LATERAL_CFG   = SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])
FEET_BODY_CFG = SceneEntityCfg("robot", body_names=[FOOT_BODY_REGEX])
FOOT_SENSOR_CFG = SceneEntityCfg("foot_contact", body_names=[FOOT_BODY_REGEX])


@configclass
class PeriodicSquatRewardsCfg:
    # ================= 主報酬: 参照姿勢トラッキング (正値・有界) =================
    pose_track = RewTerm(
        func=mdp.squat_pose_tracking, weight=8.0,
        params=dict(
            period=SQUAT_PERIOD, std=0.50,
            stand_hip_pitch=STAND_HIP_PITCH, squat_hip_pitch=SQUAT_HIP_PITCH,
            stand_knee=STAND_KNEE,           squat_knee=SQUAT_KNEE,
            stand_ankle=STAND_ANKLE,         squat_ankle=SQUAT_ANKLE,
            hip_pitch_cfg=HIP_PITCH_CFG,
            knee_cfg=KNEE_CFG,
            ankle_cfg=ANKLE_CFG,
            lateral_cfg=LATERAL_CFG,
        ),
    )
    height_track = RewTerm(
        func=mdp.squat_height_tracking, weight=3.0,
        params=dict(
            period=SQUAT_PERIOD, std=0.10,
            stand_height=STAND_HEIGHT, squat_height=SQUAT_HEIGHT,
            robot_cfg=SceneEntityCfg("robot"),
        ),
    )

    # ================= 補助 (すべて正値) =================
    upright = RewTerm(
        func=mdp.upright_bonus, weight=2.0,
        params=dict(robot_cfg=SceneEntityCfg("robot")),
    )
    grounded = RewTerm(
        func=mdp.feet_grounded, weight=1.0,
        params=dict(sensor_cfg=FOOT_SENSOR_CFG, force_threshold=1.0),
    )
    stance_width = RewTerm(
        func=mdp.feet_stance_width, weight=1.0,
        params=dict(target_width=0.20, std=0.12, robot_cfg=FEET_BODY_CFG),
    )

    # ================= その場に留まる (定位置保持) =================
    stay_put = RewTerm(
        func=mdp.stay_in_place, weight=2.0,
        params=dict(std=0.25, robot_cfg=SceneEntityCfg("robot")),
    )
    no_slip = RewTerm(
        func=mdp.feet_no_slip, weight=1.5,
        params=dict(std=0.15, force_threshold=1.0,
                    sensor_cfg=FOOT_SENSOR_CFG, asset_cfg=FEET_BODY_CFG),
    )
    heading = RewTerm(
        func=mdp.heading_hold, weight=1.0,
        params=dict(robot_cfg=SceneEntityCfg("robot")),
    )
    low_speed = RewTerm(
        func=mdp.low_base_speed, weight=0.5,
        params=dict(std=0.30, robot_cfg=SceneEntityCfg("robot")),
    )

    # ================= 正則化 (小さく) =================
    ang_vel_xy   = RewTerm(func=mdp.ang_vel_xy_l2,     weight=-0.02)
    action_rate  = RewTerm(func=mdp.action_rate_l2,    weight=-0.005)
    joint_acc    = RewTerm(func=mdp.joint_acc_l2,      weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2,  weight=-1.0e-6)
    dof_pos_lim  = RewTerm(func=mdp.joint_pos_limits,  weight=-1.0)
    arm_default  = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1,
        params=dict(asset_cfg=SceneEntityCfg(
            "robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*", "waist_.*"]
        )),
    )
    # 正報酬の最大 = 8+3+2+1+1 (姿勢) + 2+1.5+1+0.5 (定位置) = 20/step
    # 正則化は通常 -0.3 程度 -> 合計は常に大きく正 -> 生存が常に最適


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

        # --- 位相を観測に (これが無いと policy は「今しゃがむ番か」を知れない) ---
        self.observations.policy.squat_phase = ObsTerm(
            func=mdp.squat_phase_obs,
            params=dict(period=SQUAT_PERIOD),
        )

        # --- 初期姿勢は通常の立ち姿勢 (warm start しない) ---
        # 参照姿勢の phi=0 と一致させておく
        default = dict(self.scene.robot.init_state.joint_pos or {})
        default["left_hip_pitch_joint"]  = STAND_HIP_PITCH
        default["right_hip_pitch_joint"] = STAND_HIP_PITCH
        default[".*_knee_joint"]         = STAND_KNEE
        default[".*_ankle_pitch_joint"]  = STAND_ANKLE
        self.scene.robot.init_state.joint_pos = default

        # --- 転倒は「罰」ではなく「終了」で扱う ---
        self.terminations.base_contact = None          # 偶発的な骨盤接触では終了しない
        self.terminations.fell_over = DoneTerm(
            func=mdp.bad_orientation,
            params=dict(limit_angle=1.2),              # 約 69 度傾いたら終了(深い前傾を許容)
        )
        self.terminations.collapsed = DoneTerm(
            func=mdp.root_height_below_minimum,
            params=dict(minimum_height=0.25,
                        asset_cfg=SceneEntityCfg("robot")),
        )

        print(f">>> PeriodicSquat v2: period={SQUAT_PERIOD}s  "
              f"knee {STAND_KNEE}->{SQUAT_KNEE}  h {STAND_HEIGHT}->{SQUAT_HEIGHT}")


@configclass
class G1PeriodicSquatEnvCfg_PLAY(G1PeriodicSquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
