"""その場スクワット (立ち <-> 完全しゃがみ) の学習環境 v5。

**このファイル 1 つで環境が完結する**（シーン / 行動 / 観測 / イベント /
報酬 / 終了条件 / 定数）。以前は基底を base_env_cfg.py に分けていたが、
参照が 1 タスクだけなので統合した。

目標動作:
  - 膝を限界近くまで曲げきる (knee 0.30 -> 2.20 rad = 126 度)
  - 胴は正面を向いたまま (ヨー・ロールは 0、前傾ピッチのみ許可)
  - その場から動かない

腕は学習対象外。初期姿勢「肩の高さで前へ伸ばした形」で固定し、
`arm_hold_pen` がそこから外れた分だけを罰する。腕の動作学習は
脚の学習まで阻害したため 08-31 に切り離した。

設計・配点・落とし穴の解説は ../../../README.md にまとめてある。
**報酬関数・定数・環境設定を変更したら、同じ作業の中で README も更新すること。**

要点だけ再掲:
  1. 正報酬 = タスク達成のみ。追従項は `_relative_track` で
     「何もしない状態」を 0 点に正規化する (落とし穴 11)。
  2. ペナルティはすべて [-1, 0] に有界。合計が負に振れると
     エージェントは早期終了 (わざと転倒) で return を最大化する。
  3. 姿勢追従は coarse (広い std) + fine (狭い std) の 2 段構え。
  4. 29 関節すべてに役割を与える。無拘束の関節は逃げ道になる (落とし穴 19)。
     脚 10 = 参照姿勢 / 開脚抑制、胴 3 + 脚 4 = 0 固定・デフォルト維持、
     腕 14 = arm_hold_pen で固定。
  5. SceneEntityCfg は必ず RewTerm(params=...) に書く。
     デフォルト引数に置くと resolve されず全 29 関節を指してしまう。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unitree_rl_lab.tasks.squat_only.mdp as mdp
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG

# === ボディ名 (USD に合わせて調整) ==========================================
HAND_BODY_REGEX = ".*_wrist_yaw_link"
PELVIS_BODY_REGEX = ["pelvis", "torso_link"]
FOOT_BODY_REGEX = ".*_ankle_roll_link"

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

# === 腕の姿勢 (固定・学習対象外) =============================================
# 腕の動作は学習が難しく脚の学習も阻害したため、**初期姿勢で固定**する方針に変更した。
# 立ち姿勢で上腕が水平前方・肘が伸びきった「肩の高さで前へ伸ばした」形。
#
# MuJoCo モデルから、shoulder_pitch 関節の原点を基準に実測した:
#   肩pitch関節 -> 肘   (0.0158, 0.0442, -0.1975)  長さ 0.203 m
#   肘 -> 手先(wrist_yaw) (0.1840, 0.0019, -0.0100)  長さ 0.184 m
#   -> 上腕が真下を向くのは shoulder_pitch = +0.0797
#   -> 前腕が上腕と一直線になるのは elbow = +1.4368
# 腕全体を水平前方にする幾何値は shoulder_pitch -1.4911 / elbow +1.4368。
#
# ただし腕の関節は ImplicitActuator (stiffness 40 N*m/rad) で位置制御されるので、
# 目標をそのまま与えると自重で垂れる:
#   肩: 片腕 3.52 kg x 肩からの重心 0.1535 m -> 5.30 N*m -> 0.133 rad (7.6 度)
#   肘: 前腕 0.82 kg x 肘からの重心 0.1487 m -> 1.20 N*m -> 0.030 rad (1.7 度)
# たわむ分だけ目標を先回りさせて、実際の姿勢が水平になるようにする。
ARM_SHOULDER_PITCH = -1.6236  # = -1.4911 - 0.1325 (たわみ補償, soft 限界 -2.801 内)
ARM_ELBOW          =  1.4668  # = +1.4368 + 0.0300 (たわみ補償, soft 限界 +1.937 内)
#
# 腕は関節角で固定するので、しゃがんで胴が前傾すると腕も一緒に傾く。
# 完全しゃがみ (前傾 37 度) では腕は水平から 37 度下向き = 前下方へのリーチ姿勢になり、
# 将来の箱拾いへ素直につながる。
#
# 重心への影響: 腕 7.04 kg (全体の 20%) が肩から 0.098 m の位置で前方へ回るので
# 重心は前へ 2.0 cm 移動する。完全しゃがみでの余裕は
#   踵まで 0.126 m / つま先まで 0.084 m (前傾 0.65 のままで成立)。

# 胴の前傾 [rad]。股関節が足首の真上に来る条件は knee = 2 x |ankle|。
# ankle の soft 限界が 0.803 なので、踵接地のまま股関節を足の上に保てるのは
# knee <= 1.61 まで。それより深いと股関節は必ず後ろへ抜けるので前傾で戻す。
# 採用値での重心: COM_x +0.046 / 踵余裕 0.106 / つま先余裕 0.104 (ほぼ均衡)
TORSO_STAND_PITCH, TORSO_SQUAT_PITCH = 0.00, 0.65

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
WAIST_PITCH_CFG = SceneEntityCfg("robot", joint_names=["waist_pitch_joint"])
# 腕は学習対象から外し、この 14 関節をまとめてデフォルト姿勢に保つ
ARM_HOLD_CFG    = SceneEntityCfg("robot", joint_names=[
    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
    ".*_elbow_joint", ".*_wrist_.*_joint",
])
# 参照姿勢を持たないが放置すると逃げ道になる関節 (落とし穴 19)
ANKLE_ROLL_CFG  = SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])
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
class SquatSceneCfg(InteractiveSceneCfg):
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + FOOT_BODY_REGEX,
        track_air_time=True,
        history_length=3,
    )
    hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + HAND_BODY_REGEX,
        history_length=3,
    )
    body_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(pelvis|torso_link)",
        history_length=3,
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9)),
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.4,
        rel_heading_envs=0.5,
        heading_command=True,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.8),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )


# ---------------------------------------------------------------------------
# Actions 
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    # 目標関節角 = default_joint_pos + scale * action。
    #
    # 一律 0.25 だと膝の目標 2.20 rad に必要な action が 7.6 になり、方策の
    # 探索幅 3σ ≈ 2.2 では物理的に届かず「中腰」で頭打ちになる (落とし穴 16)。
    # 可動域が大きい関節だけスケールを上げ、必要な action を ±2.5 以内に収める。
    # 上げすぎると探索ノイズ (= scale * init_noise_std) も比例して増えて
    # ロボットがよろけるので 0.8 まで (落とし穴 18)。
    #
    # NOTE: dict を渡すと「マッチしなかった関節は 1.0」になる。29 関節すべてを
    #       明示すること。正規表現が二重マッチすると例外が飛ぶ。
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        use_default_offset=True,
        scale={
            # 可動域が大きい関節
            ".*_hip_pitch_joint":      0.8,   # -0.10 -> -2.10 (必要 a = -2.5)
            ".*_knee_joint":           0.8,   #  0.30 ->  2.20 (必要 a = +2.4)
            ".*_ankle_pitch_joint":    0.5,   # -0.20 -> -0.75 (必要 a = -1.1)
            # 腕は固定姿勢を保つだけなので小さく (探索ノイズを減らす)
            ".*_shoulder_pitch_joint": 0.25,
            ".*_elbow_joint":          0.25,
            # 0 付近に留めたい関節
            ".*_hip_roll_joint":       0.25,
            ".*_hip_yaw_joint":        0.25,
            ".*_ankle_roll_joint":     0.25,
            "waist_yaw_joint":         0.25,
            "waist_roll_joint":        0.25,
            "waist_pitch_joint":       0.25,
            ".*_shoulder_roll_joint":  0.25,
            ".*_shoulder_yaw_joint":   0.25,
            ".*_wrist_roll_joint":     0.25,
            ".*_wrist_pitch_joint":    0.25,
            ".*_wrist_yaw_joint":      0.25,
        },
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        pass

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@configclass
class EventsCfg:
    reset_robot = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


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
    # 合計を正に保つ床 (転倒判定は終了条件が担当するので小さく)
    # --- 腕は学習させず、初期姿勢 (肩の高さで前方に伸ばした形) を保たせるだけ ---
    # 目標＝デフォルトなので action 0 でそのまま維持される。
    # margin は自重によるたわみ (肩 0.133 rad) を無罰にする幅にしてある。
    arm_hold_pen = RewTerm(
        func=mdp.joint_default_deviation_penalty, weight=4.0,
        params=dict(margin=0.25, std=0.35, robot_cfg=ARM_HOLD_CFG),
    )
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
        func=mdp.drift_penalty, weight=3.0,
        params=dict(std=0.60, robot_cfg=SceneEntityCfg("robot")),
    )
    slip_pen = RewTerm(
        func=mdp.feet_slip_penalty, weight=1.0,
        params=dict(std=0.30, force_threshold=1.0,
                    sensor_cfg=FOOT_SENSOR_CFG, asset_cfg=FEET_BODY_CFG),
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
        func=mdp.hands_knee_clearance_penalty, weight=5.0,
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
    # ankle_roll は横方向の連鎖で唯一まったく拘束されていなかった。
    ankle_roll_pen = RewTerm(
        func=mdp.joint_default_deviation_penalty, weight=2.0,
        params=dict(margin=0.10, std=0.20, robot_cfg=ANKLE_ROLL_CFG),
    )
    # --- 左右で打ち消し合う非対称姿勢を潰す ---
    # 「膝が左・胴が右」は倒れないので、個別の関節罰だけでは抜け出せない。
    # 左右の中点で見るので、対称な開脚は無罰。
    knee_lateral_pen = RewTerm(
        func=mdp.lateral_offset_penalty, weight=3.0,
        params=dict(std=0.06, body_cfg=KNEE_BODY_CFG),
    )
    feet_lateral_pen = RewTerm(
        func=mdp.lateral_offset_penalty, weight=2.0,
        params=dict(std=0.08, body_cfg=FEET_BODY_CFG),
    )
    torso_roll_pen = RewTerm(
        func=mdp.torso_roll_penalty, weight=3.0,
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
    action_rate  = RewTerm(func=mdp.action_rate_l2,    weight=-0.015)
    joint_acc    = RewTerm(func=mdp.joint_acc_l2,      weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2,  weight=-1.0e-6)
    dof_pos_lim  = RewTerm(func=mdp.joint_pos_limits,  weight=-1.0)
    # 棒立ち   : 約 5.5 / 18.0
    # 正しいスクワット: 約 11.5 / 18.0
    # 開脚スクワット  : 約 6.0  (開脚ペナルティ -5.5 で相殺)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params=dict(
            sensor_cfg=SceneEntityCfg("body_contact", body_names=PELVIS_BODY_REGEX),
            threshold=1.0,
        ),
    )


@configclass
class G1PeriodicSquatEnvCfg(ManagerBasedRLEnvCfg):
    scene: SquatSceneCfg = SquatSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    rewards: PeriodicSquatRewardsCfg = PeriodicSquatRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # --- シミュレーション ---
        self.decimation = 4
        self.sim.dt = 0.005                    # 制御 50 Hz
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.5, 2.5, 1.5)

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

        # --- 腕は「肩の高さで前へ伸ばした」姿勢で固定 ---
        # use_default_offset=True なので、これが action 0 の姿勢になる。
        # NOTE: UNITREE_G1_29DOF_CFG が持つ既存キーを上書きする。
        #       新しい正規表現を足すと二重マッチで ValueError になる。
        default[".*_shoulder_pitch_joint"] = ARM_SHOULDER_PITCH
        default["left_shoulder_roll_joint"]  = 0.0     # 元 +0.25 (外へ開く)
        default["right_shoulder_roll_joint"] = 0.0     # 元 -0.25
        default[".*_elbow_joint"]          = ARM_ELBOW
        default["left_wrist_roll_joint"]   = 0.0       # 元 +0.15
        default["right_wrist_roll_joint"]  = 0.0       # 元 -0.15
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

        print(f">>> PeriodicSquat v6 (腕は水平前方で固定): period={SQUAT_PERIOD}s")
        print("    action scale: hip_pitch/knee 0.8, shoulder 0.6, ankle/elbow 0.5, 他 0.25")
        print(f"    knee   {STAND_KNEE} -> {SQUAT_KNEE} rad")
        print(f"    height {STAND_HEIGHT} -> {SQUAT_HEIGHT} m")
        print(f"    arm    固定 shoulder_pitch={ARM_SHOULDER_PITCH} elbow={ARM_ELBOW} (学習対象外)")
        print(f"    torso  前傾 {TORSO_STAND_PITCH} -> {TORSO_SQUAT_PITCH} rad")


@configclass
class G1PeriodicSquatEnvCfg_PLAY(G1PeriodicSquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
