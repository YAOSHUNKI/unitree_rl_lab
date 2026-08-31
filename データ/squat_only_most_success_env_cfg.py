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

# === Cycle ===================================================================
SQUAT_PERIOD = 6.0        

# === Reference Posture for the Legs ===========================================================
STAND_HIP_PITCH, SQUAT_HIP_PITCH = -0.10, -2.10
STAND_KNEE,      SQUAT_KNEE      = 0.30, 2.20   
STAND_ANKLE,     SQUAT_ANKLE     = -0.20, -0.75  

STAND_HEIGHT, SQUAT_HEIGHT = 0.73, 0.39

# === Reference Posture for the Arms ===========================================================
STAND_ARM_FWD,  SQUAT_ARM_FWD  = 0.00,  0.55

TORSO_STAND_PITCH, TORSO_SQUAT_PITCH = 0.00, 0.65

HAND_WIDTH_SCALE = 1.0                        
HAND_WIDTH_MIN   = 0.16                      

# === Maximum Leg Spread ===========================================================
STAND_ABDUCTION, SQUAT_ABDUCTION = 0.00, 0.18   
STAND_WIDTH,     SQUAT_WIDTH     = 0.20, 0.28  
# === SceneEntityCfg  ====================================
HIP_PITCH_CFG = SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint"])
KNEE_CFG      = SceneEntityCfg("robot", joint_names=[".*_knee_joint"])
ANKLE_CFG     = SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"])
LATERAL_CFG   = SceneEntityCfg("robot", joint_names=[
    ".*_hip_yaw_joint", "waist_yaw_joint", "waist_roll_joint",
])
HIP_ROLL_CFG  = SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])
FEET_BODY_CFG   = SceneEntityCfg("robot", body_names=[FOOT_BODY_REGEX])
HAND_BODY_CFG   = SceneEntityCfg("robot", body_names=[HAND_BODY_REGEX])
KNEE_BODY_CFG   = SceneEntityCfg("robot", body_names=[".*_knee_link"])
SHOULDER_CFG    = SceneEntityCfg("robot", body_names=[".*_shoulder_yaw_link"])
ELBOW_CFG       = SceneEntityCfg("robot", body_names=[".*_elbow_link"])
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
    torso_pitch = RewTerm(
        func=mdp.torso_pitch_tracking, weight=3.0,
        params=dict(
            period=SQUAT_PERIOD, std=0.15,
            stand_pitch=TORSO_STAND_PITCH, squat_pitch=TORSO_SQUAT_PITCH,
            robot_cfg=SceneEntityCfg("robot"),
        ),
    )
    arm_forward = RewTerm(
        func=mdp.arm_forward_direction, weight=5.0,
        params=dict(
            period=SQUAT_PERIOD, std=0.12,
            stand_forward=STAND_ARM_FWD, squat_forward=SQUAT_ARM_FWD,
            shoulder_cfg=SHOULDER_CFG, elbow_cfg=ELBOW_CFG,
        ),
    )

    hands_width = RewTerm(
        func=mdp.hands_width_match, weight=2.0,
        params=dict(
            width_scale=HAND_WIDTH_SCALE, min_width=HAND_WIDTH_MIN, std=0.06,
            hand_cfg=HAND_BODY_CFG, knee_cfg=KNEE_BODY_CFG,
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
    drift_pen = RewTerm(
        func=mdp.drift_penalty, weight=1.5,
        params=dict(std=0.25, robot_cfg=SceneEntityCfg("robot")),
    )
    slip_pen = RewTerm(
        func=mdp.feet_slip_penalty, weight=1.5,
        params=dict(std=0.15, force_threshold=1.0,
                    sensor_cfg=FOOT_SENSOR_CFG, asset_cfg=FEET_BODY_CFG),
    )
    hands_sym_pen = RewTerm(
        func=mdp.hands_symmetry_penalty, weight=1.0,
        params=dict(std=0.10, hand_cfg=HAND_BODY_CFG),
    )
    arm_ext_pen = RewTerm(
        func=mdp.arm_extension_penalty, weight=3.0,
        params=dict(
            period=SQUAT_PERIOD, min_straightness=0.97, std=0.06,
            shoulder_cfg=SHOULDER_CFG, elbow_cfg=ELBOW_CFG, hand_cfg=HAND_BODY_CFG,
        ),
    )
    torso_roll_pen = RewTerm(
        func=mdp.torso_roll_penalty, weight=1.0,
        params=dict(std=0.15, robot_cfg=SceneEntityCfg("robot")),
    )
    heading_pen = RewTerm(
        func=mdp.heading_penalty, weight=1.0,
        params=dict(robot_cfg=SceneEntityCfg("robot")),
    )
    speed_pen = RewTerm(
        func=mdp.base_speed_penalty, weight=0.5,
        params=dict(std=0.30, robot_cfg=SceneEntityCfg("robot")),
    )
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

    ang_vel_xy   = RewTerm(func=mdp.ang_vel_xy_l2,     weight=-0.02)
    action_rate  = RewTerm(func=mdp.action_rate_l2,    weight=-0.005)
    joint_acc    = RewTerm(func=mdp.joint_acc_l2,      weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2,  weight=-1.0e-6)
    dof_pos_lim  = RewTerm(func=mdp.joint_pos_limits,  weight=-1.0)
    wrist_default = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.30,
        params=dict(asset_cfg=SceneEntityCfg("robot", joint_names=[".*_wrist_.*"])),
    )


@configclass
class G1PeriodicSquatEnvCfg(G1PickupCarryEnvCfg):
    rewards: PeriodicSquatRewardsCfg = PeriodicSquatRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0

        self.episode_length_s = 12.0   # 2周期

        self.observations.policy.squat_phase = ObsTerm(
            func=mdp.squat_phase_obs,
            params=dict(period=SQUAT_PERIOD),
        )

        default = dict(self.scene.robot.init_state.joint_pos or {})
        default["left_hip_pitch_joint"]  = STAND_HIP_PITCH
        default["right_hip_pitch_joint"] = STAND_HIP_PITCH
        default[".*_knee_joint"]         = STAND_KNEE
        default[".*_ankle_pitch_joint"]  = STAND_ANKLE
        self.scene.robot.init_state.joint_pos = default

        self.terminations.base_contact = None
        self.terminations.fell_over = DoneTerm(
            func=mdp.bad_orientation,
            params=dict(limit_angle=1.2),          
        )
        self.terminations.collapsed = DoneTerm(
            func=mdp.root_height_below_minimum,
            params=dict(minimum_height=0.20,
                        asset_cfg=SceneEntityCfg("robot")),
        )

        print(f">>> PeriodicSquat v3: period={SQUAT_PERIOD}s")
        print(f"    knee   {STAND_KNEE} -> {SQUAT_KNEE} rad")
        print(f"    height {STAND_HEIGHT} -> {SQUAT_HEIGHT} m")
        print(f"    arm    上腕の前方成分 {STAND_ARM_FWD} -> {SQUAT_ARM_FWD}")
        print(f"    torso  前傾 {TORSO_STAND_PITCH} -> {TORSO_SQUAT_PITCH} rad")


@configclass
class G1PeriodicSquatEnvCfg_PLAY(G1PeriodicSquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False