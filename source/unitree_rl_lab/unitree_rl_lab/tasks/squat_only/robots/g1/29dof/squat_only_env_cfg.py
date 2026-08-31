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

# === Body name ==========================================
HAND_BODY_REGEX = ".*_wrist_yaw_link"
PELVIS_BODY_REGEX = ["pelvis", "torso_link"]
FOOT_BODY_REGEX = ".*_ankle_roll_link"

# === Period ===================================================================
SQUAT_PERIOD = 6.0      

# === Reference Posture for the Legs ===========================================================
STAND_HIP_PITCH, SQUAT_HIP_PITCH = -0.10, -2.10
STAND_KNEE,      SQUAT_KNEE      = 0.30, 2.20   
STAND_ANKLE,     SQUAT_ANKLE     = -0.20, -0.75  

STAND_HEIGHT, SQUAT_HEIGHT = 0.73, 0.39

# === Arm Position=============================================
ARM_SHOULDER_PITCH = -1.6236  
ARM_ELBOW          =  1.4668  

TORSO_STAND_PITCH, TORSO_SQUAT_PITCH = 0.00, 0.65

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
WAIST_PITCH_CFG = SceneEntityCfg("robot", joint_names=["waist_pitch_joint"])

ARM_HOLD_CFG    = SceneEntityCfg("robot", joint_names=[
    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
    ".*_elbow_joint", ".*_wrist_.*_joint",
])

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
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        use_default_offset=True,
        scale={
            # Joints with a wide range of motion
            ".*_hip_pitch_joint":      0.8,  
            ".*_knee_joint":           0.8,  
            ".*_ankle_pitch_joint":    0.5, 
            
            ".*_shoulder_pitch_joint": 0.25,
            ".*_elbow_joint":          0.25,
            # Joints that should be kept around 0
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
    # ================= Base Reward: Task Completion (Max 25.0) =================
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
 
    torso_pitch = RewTerm(
        func=mdp.torso_pitch_tracking, weight=3.0,
        params=dict(
            period=SQUAT_PERIOD, std=0.40,
            stand_pitch=TORSO_STAND_PITCH, squat_pitch=TORSO_SQUAT_PITCH,
            robot_cfg=SceneEntityCfg("robot"),
        ),
    )

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

    # ================= Handicap: (minimum -19.5) =================
    drift_pen = RewTerm(
        func=mdp.drift_penalty, weight=3.0,
        params=dict(std=0.60, robot_cfg=SceneEntityCfg("robot")),
    )
    slip_pen = RewTerm(
        func=mdp.feet_slip_penalty, weight=1.0,
        params=dict(std=0.30, force_threshold=1.0,
                    sensor_cfg=FOOT_SENSOR_CFG, asset_cfg=FEET_BODY_CFG),
    )
    squat_shortfall_pen = RewTerm(
        func=mdp.squat_depth_shortfall_penalty, weight=3.0,
        params=dict(
            period=SQUAT_PERIOD, stand_knee=STAND_KNEE, squat_knee=SQUAT_KNEE,
            min_ratio=0.85, std=0.90, knee_cfg=KNEE_CFG,
        ),
    )
    knee_clear_pen = RewTerm(
        func=mdp.hands_knee_clearance_penalty, weight=5.0,
        params=dict(
            period=SQUAT_PERIOD, min_distance=0.18, std=0.08,
            hand_cfg=HAND_BODY_CFG, knee_cfg=KNEE_BODY_CFG,
        ),
    )
    backlean_pen = RewTerm(
        func=mdp.torso_backlean_penalty, weight=3.0,
        params=dict(margin=0.10, std=0.15, robot_cfg=SceneEntityCfg("robot")),
    )
    waist_pitch_pen = RewTerm(
        func=mdp.waist_pitch_penalty, weight=4.0,
        params=dict(max_abs=0.10, std=0.12, robot_cfg=WAIST_PITCH_CFG),
    )
    ankle_roll_pen = RewTerm(
        func=mdp.joint_default_deviation_penalty, weight=2.0,
        params=dict(margin=0.10, std=0.20, robot_cfg=ANKLE_ROLL_CFG),
    )
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
    heading_pen = RewTerm(
        func=mdp.heading_penalty, weight=2.0,
        params=dict(robot_cfg=SceneEntityCfg("robot")),
    )
    yaw_rate_pen = RewTerm(
        func=mdp.yaw_rate_penalty, weight=2.0,
        params=dict(std=0.50, robot_cfg=SceneEntityCfg("robot")),
    )
    speed_pen = RewTerm(
        func=mdp.base_speed_penalty, weight=0.5,
        params=dict(std=0.40, robot_cfg=SceneEntityCfg("robot")),
    )
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

    # ================= Regularization =================
    ang_vel_xy   = RewTerm(func=mdp.ang_vel_xy_l2,     weight=-0.02)
    action_rate  = RewTerm(func=mdp.action_rate_l2,    weight=-0.015)
    joint_acc    = RewTerm(func=mdp.joint_acc_l2,      weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2,  weight=-1.0e-6)
    dof_pos_lim  = RewTerm(func=mdp.joint_pos_limits,  weight=-1.0)

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
        self.decimation = 4
        self.sim.dt = 0.005                  
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.5, 2.5, 1.5)

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0

        self.episode_length_s = 12.0   

        self.observations.policy.squat_phase = ObsTerm(
            func=mdp.squat_phase_obs,
            params=dict(period=SQUAT_PERIOD),
        )

        default = dict(self.scene.robot.init_state.joint_pos or {})
        default["left_hip_pitch_joint"]  = STAND_HIP_PITCH
        default["right_hip_pitch_joint"] = STAND_HIP_PITCH
        default[".*_knee_joint"]         = STAND_KNEE
        default[".*_ankle_pitch_joint"]  = STAND_ANKLE

        default[".*_shoulder_pitch_joint"] = ARM_SHOULDER_PITCH
        default["left_shoulder_roll_joint"]  = 0.0     
        default["right_shoulder_roll_joint"] = 0.0     
        default[".*_elbow_joint"]          = ARM_ELBOW
        default["left_wrist_roll_joint"]   = 0.0      
        default["right_wrist_roll_joint"]  = 0.0      
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

        print(f">>> PeriodicSquat : period={SQUAT_PERIOD}s")
        print("    action scale: hip_pitch/knee 0.8, shoulder 0.6, ankle/elbow 0.5,  0.25")
        print(f"    knee   {STAND_KNEE} -> {SQUAT_KNEE} rad")
        print(f"    height {STAND_HEIGHT} -> {SQUAT_HEIGHT} m")
        print(f"    arm     shoulder_pitch={ARM_SHOULDER_PITCH} elbow={ARM_ELBOW}")
        print(f"    torso   {TORSO_STAND_PITCH} -> {TORSO_SQUAT_PITCH} rad")


@configclass
class G1PeriodicSquatEnvCfg_PLAY(G1PeriodicSquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
