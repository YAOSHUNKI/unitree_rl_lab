"""しゃがみ状態から箱を掴んで立ち上がる単発動作の学習環境 (SquatStandLift)。

目標動作:
  - spawn 時点で完全しゃがみ姿勢 (骨盤 0.397 m, 胴前傾 0.65 rad)
  - 目の前 (前方 0.34 m) の 20 cm 立方体に両手を触れて掴む
  - 掴んだまま立ち上がる (骨盤 0.73 m)
  - その間ずっとその場から動かない

設計方針 (PeriodicSquat + 箱把持・持ち上げの融合):
  - PeriodicSquat の姿勢追従群を period=6.0 s / phase_offset=0.5 で半周期だけ使い、
    しゃがみ (phi=0.5) -> 立ち (phi=1.0) の単調な追従にする。
  - episode_length_s = T_task = 3.0 s で切ることで戻り位相 (立ち->しゃがみ) に
    入らせない。
  - 箱関連の報酬 (hands_near, hands_touch, grasp, lift, stand_up) は
    _squat_depth の余相 s = 1 - depth をゲートに使い、しゃがみ位相では効かせず、
    立ち上がり位相で効かせる。
  - 「腕を前に振る」系の項 (arm_forward_direction / hands_width /
    arm_extension_penalty / arm_forward_shortfall_penalty / hands_knee_clearance)
    は箱に手をつける動機と競合するので落とす (落とし穴 7)。

原則 (g1-squat-reward-reference.md より):
  1. 正報酬 = タスク達成のみ。定位置保持はペナルティで。
  2. ペナルティは有界 exp(-x^2/sigma^2) - 1 in [-1, 0]。
  3. 転倒は罰ではなく終了で扱う。
  4. 参照姿勢は静的に安定でなければならない。
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.squat_stand_lift.mdp as mdp

# NOTE: フォルダ名が数字で始まる (29dof) ため通常の import では読めない。
# 実体クラスは entry_point 文字列経由で読まれるので、この env_cfg の
# 基底クラスを import するために importlib を使う。
import importlib
import math
_pickup = importlib.import_module(
    "unitree_rl_lab.tasks.squat_stand_lift.robots.g1.29dof.pickup_carry_env_cfg"
)
G1PickupCarryEnvCfg = _pickup.G1PickupCarryEnvCfg
FOOT_BODY_REGEX = _pickup.FOOT_BODY_REGEX
HAND_BODY_REGEX = _pickup.HAND_BODY_REGEX
HAND_BODY_NAMES = _pickup.HAND_BODY_NAMES


# === 位相設計 ===============================================================
T_TASK = 3.0                    # しゃがみ->立ちにかける時間 [s]
SQUAT_PERIOD = 2.0 * T_TASK     # 6.0 s
PHASE_OFFSET = 0.5              # phi(t=0)=0.5 -> depth=1 (完全しゃがみ) から始まる


# === 脚の参照姿勢 (PeriodicSquat と同一) =====================================
STAND_HIP_PITCH, SQUAT_HIP_PITCH = -0.10, -2.10
STAND_KNEE,      SQUAT_KNEE      = 0.30,  2.20
STAND_ANKLE,     SQUAT_ANKLE     = -0.20, -0.75
# NOTE: These are articulation-root (pelvis) heights, not foot heights.
# With the squat joint pose and 0.65-rad torso pitch below, the lowest
# ankle-roll visual mesh point is 0.3948 m below the pelvis (measured from
# g1_29dof_rev_1_0). Keep a 2 mm clearance to avoid an initial
# penetration/contact impulse.
STAND_HEIGHT,    SQUAT_HEIGHT    = 0.73,  0.397
TORSO_STAND_PITCH, TORSO_SQUAT_PITCH = 0.00, 0.65

# === 開脚許容 (PeriodicSquat と同一) =========================================
STAND_ABDUCTION, SQUAT_ABDUCTION = 0.00, 0.18
STAND_WIDTH,     SQUAT_WIDTH     = 0.20, 0.28

# === 腕 spawn 用の関節角 ====================================================
# G1 の shoulder_pitch は負方向が前方。root を 0.65 rad 前傾させた
# 状態で、この組み合わせは両手首をおよそ x=0.3 m（前方）へ置く。
# shoulder_roll は左右の手を箱（幅 0.2 m）の側面に合わせて内側へ寄せる。
SQUAT_SHOULDER_PITCH = -0.80
SQUAT_LEFT_SHOULDER_ROLL = -0.20
SQUAT_RIGHT_SHOULDER_ROLL = 0.20
SQUAT_ELBOW = 1.00


# === SceneEntityCfg =========================================================
HIP_PITCH_CFG = SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint"])
KNEE_CFG      = SceneEntityCfg("robot", joint_names=[".*_knee_joint"])
ANKLE_CFG     = SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"])
LATERAL_CFG   = SceneEntityCfg("robot", joint_names=[
    ".*_hip_yaw_joint", "waist_yaw_joint", "waist_roll_joint",
])
HIP_ROLL_CFG  = SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])
FEET_BODY_CFG = SceneEntityCfg("robot", body_names=[FOOT_BODY_REGEX])
FOOT_SENSOR_CFG = SceneEntityCfg("foot_contact", body_names=[FOOT_BODY_REGEX])
HAND_SENSOR_CFG = SceneEntityCfg("hand_contact", body_names=[HAND_BODY_REGEX])
WAIST_PITCH_CFG = SceneEntityCfg("robot", joint_names=["waist_pitch_joint"])
WRIST_CFG       = SceneEntityCfg("robot", joint_names=[".*_wrist_.*_joint"])


_POSE_PARAMS = dict(
    period=SQUAT_PERIOD,
    phase_offset=PHASE_OFFSET,
    stand_hip_pitch=STAND_HIP_PITCH, squat_hip_pitch=SQUAT_HIP_PITCH,
    stand_knee=STAND_KNEE,           squat_knee=SQUAT_KNEE,
    stand_ankle=STAND_ANKLE,         squat_ankle=SQUAT_ANKLE,
    hip_pitch_cfg=HIP_PITCH_CFG,
    knee_cfg=KNEE_CFG,
    ankle_cfg=ANKLE_CFG,
    lateral_cfg=LATERAL_CFG,
)


@configclass
class SquatStandLiftRewardsCfg:
    # ================= 姿勢追従 (PeriodicSquat 流用、位相オフセット付き) =====
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
            period=SQUAT_PERIOD, phase_offset=PHASE_OFFSET, std=0.15,
            stand_height=STAND_HEIGHT, squat_height=SQUAT_HEIGHT,
            robot_cfg=SceneEntityCfg("robot"),
        ),
    )
    torso_pitch = RewTerm(
        func=mdp.torso_pitch_tracking, weight=3.0,
        params=dict(
            period=SQUAT_PERIOD, phase_offset=PHASE_OFFSET, std=0.15,
            stand_pitch=TORSO_STAND_PITCH, squat_pitch=TORSO_SQUAT_PITCH,
            robot_cfg=SceneEntityCfg("robot"),
        ),
    )
    upright = RewTerm(
        func=mdp.upright_bonus, weight=0.5,
        params=dict(robot_cfg=SceneEntityCfg("robot")),
    )
    grounded = RewTerm(
        func=mdp.feet_grounded, weight=0.5,
        params=dict(sensor_cfg=FOOT_SENSOR_CFG, force_threshold=1.0),
    )

    # ================= 箱関連 (このタスク内 MDP) ===========================
    # NOTE: 元関数はゲートを持たないので、しゃがみ位相でも部分点が入る。
    # これで問題があれば SquatStandLift 専用のゲート付きラッパを作る。
    # まずは弱めの重みで生の値を投入して挙動を観察する。
    hands_near = RewTerm(
        func=mdp.hands_near_box, weight=1.5,
        params=dict(std=0.15, hand_body_names=HAND_BODY_NAMES),
    )
    hands_touch = RewTerm(
        func=mdp.hands_contact_box, weight=1.5,
        params=dict(sensor_cfg=HAND_SENSOR_CFG),
    )
    grasp = RewTerm(
        func=mdp.grasp_bonus, weight=3.0,
        params=dict(hand_body_names=HAND_BODY_NAMES,
                    contact_sensor_name="hand_contact"),
    )
    lift = RewTerm(
        func=mdp.lift_box, weight=4.0,
        params=dict(
            initial_z=0.10, target_lift=0.35, std=0.12,
            hand_body_names=HAND_BODY_NAMES,
            contact_sensor_name="hand_contact",
        ),
    )
    stand_up = RewTerm(
        func=mdp.stand_up_when_lifting, weight=3.0,
        params=dict(
            stand_height=STAND_HEIGHT, std=0.10,
            hand_body_names=HAND_BODY_NAMES,
            contact_sensor_name="hand_contact",
        ),
    )
    drop_pen = RewTerm(func=mdp.drop_box_penalty, weight=1.0)

    # ================= 定位置保持ペナルティ =================================
    drift_pen = RewTerm(
        func=mdp.drift_penalty, weight=1.5,
        params=dict(std=0.25, robot_cfg=SceneEntityCfg("robot")),
    )
    slip_pen = RewTerm(
        func=mdp.feet_slip_penalty, weight=1.5,
        params=dict(std=0.15, force_threshold=1.0,
                    sensor_cfg=FOOT_SENSOR_CFG, asset_cfg=FEET_BODY_CFG),
    )
    heading_pen = RewTerm(
        func=mdp.heading_penalty, weight=1.0,
        params=dict(robot_cfg=SceneEntityCfg("robot")),
    )
    speed_pen = RewTerm(
        func=mdp.base_speed_penalty, weight=0.5,
        params=dict(std=0.30, robot_cfg=SceneEntityCfg("robot")),
    )

    # ================= 胴の姿勢ペナルティ ===================================
    waist_pitch_pen = RewTerm(
        func=mdp.waist_pitch_penalty, weight=4.0,
        params=dict(max_abs=0.10, std=0.12, robot_cfg=WAIST_PITCH_CFG),
    )
    torso_roll_pen = RewTerm(
        func=mdp.torso_roll_penalty, weight=1.0,
        params=dict(std=0.15, robot_cfg=SceneEntityCfg("robot")),
    )

    # ================= 開脚抑制 (位相付き) ==================================
    hip_abduction_pen = RewTerm(
        func=mdp.hip_abduction_tracking, weight=6.0,
        params=dict(
            period=SQUAT_PERIOD, phase_offset=PHASE_OFFSET, std=0.12,
            stand_abduction=STAND_ABDUCTION, squat_abduction=SQUAT_ABDUCTION,
            robot_cfg=HIP_ROLL_CFG,
        ),
    )
    stance_pen = RewTerm(
        func=mdp.stance_width_penalty_phased, weight=5.0,
        params=dict(
            period=SQUAT_PERIOD, phase_offset=PHASE_OFFSET, std=0.08,
            stand_width=STAND_WIDTH, squat_width=SQUAT_WIDTH,
            robot_cfg=FEET_BODY_CFG,
        ),
    )

    # ================= 正則化 ================================================
    ang_vel_xy   = RewTerm(func=mdp.ang_vel_xy_l2,     weight=-0.02)
    action_rate  = RewTerm(func=mdp.action_rate_l2,    weight=-0.005)
    joint_acc    = RewTerm(func=mdp.joint_acc_l2,      weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2,  weight=-1.0e-6)
    dof_pos_lim  = RewTerm(func=mdp.joint_pos_limits,  weight=-1.0)
    wrist_pen = RewTerm(
        func=mdp.wrist_neutral_penalty, weight=3.0,
        params=dict(max_abs=0.15, std=0.25, robot_cfg=WRIST_CFG),
    )


@configclass
class G1SquatStandLiftEnvCfg(G1PickupCarryEnvCfg):
    rewards: SquatStandLiftRewardsCfg = SquatStandLiftRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- 移動はさせない ---
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0

        # --- エピソード長 = 半周期 (戻り位相に入らせない) ---
        self.episode_length_s = T_TASK

        # --- 位相観測 ---
        self.observations.policy.squat_phase = ObsTerm(
            func=mdp.squat_phase_obs,
            params=dict(period=SQUAT_PERIOD, phase_offset=PHASE_OFFSET),
        )

        # --- 初期姿勢: 完全しゃがみ ---
        default = dict(self.scene.robot.init_state.joint_pos or {})
        default["left_hip_pitch_joint"]  = SQUAT_HIP_PITCH
        default["right_hip_pitch_joint"] = SQUAT_HIP_PITCH
        default[".*_knee_joint"]         = SQUAT_KNEE
        default[".*_ankle_pitch_joint"]  = SQUAT_ANKLE
        default[".*_shoulder_pitch_joint"] = SQUAT_SHOULDER_PITCH
        default["left_shoulder_roll_joint"] = SQUAT_LEFT_SHOULDER_ROLL
        default["right_shoulder_roll_joint"] = SQUAT_RIGHT_SHOULDER_ROLL
        default[".*_elbow_joint"]          = SQUAT_ELBOW
        self.scene.robot.init_state.joint_pos = default
        self.scene.robot.init_state.pos = (0.0, 0.0, SQUAT_HEIGHT)
        # TORSO_SQUAT_PITCH is a root/base pitch. The tracking reward alone
        # does not apply it to the initial state. Isaac Lab uses (w, x, y, z).
        half_pitch = 0.5 * TORSO_SQUAT_PITCH
        self.scene.robot.init_state.rot = (
            math.cos(half_pitch), 0.0, math.sin(half_pitch), 0.0
        )

        # --- 箱: 前へ伸ばした手の少し先に spawn ---
        # 箱の一辺は 0.20 m。中心を 0.46--0.52 m に置くと、手前面は
        # x=0.36--0.42 m となり、spawn 時の手首より少し前になる。
        self.events.reset_box.params.update(
            x_range=(0.46, 0.52),
            y_range=(-0.03, 0.03),
            z=0.10,
            yaw_range=(-0.1, 0.1),
        )

        # --- 終了条件 ---
        self.terminations.base_contact = None
        self.terminations.fell_over = DoneTerm(
            func=mdp.bad_orientation,
            params=dict(limit_angle=1.2),
        )
        # spawn 時の骨盤高は 0.397 m なので閾値 0.20 は余裕あり
        self.terminations.collapsed = DoneTerm(
            func=mdp.root_height_below_minimum,
            params=dict(minimum_height=0.20,
                        asset_cfg=SceneEntityCfg("robot")),
        )

        print(f">>> SquatStandLift: T_task={T_TASK}s, period={SQUAT_PERIOD}s, phase_offset={PHASE_OFFSET}")
        print(f"    spawn: knee={SQUAT_KNEE} height={SQUAT_HEIGHT}m torso_pitch={TORSO_SQUAT_PITCH}rad")
        print(
            "    spawn arm: "
            f"shoulder_pitch={SQUAT_SHOULDER_PITCH} "
            f"shoulder_roll=({SQUAT_LEFT_SHOULDER_ROLL}, "
            f"{SQUAT_RIGHT_SHOULDER_ROLL}) elbow={SQUAT_ELBOW}"
        )
        print(f"    goal:  knee={STAND_KNEE} height={STAND_HEIGHT}m")


@configclass
class G1SquatStandLiftEnvCfg_PLAY(G1SquatStandLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        # play では箱位置を固定
        self.events.reset_box.params["x_range"] = (0.49, 0.49)
        self.events.reset_box.params["y_range"] = (0.0, 0.0)
        self.events.reset_box.params["yaw_range"] = (0.0, 0.0)
