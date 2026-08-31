from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
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

import unitree_rl_lab.tasks.squat_stand_lift.mdp as mdp
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG

HAND_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
HAND_BODY_REGEX = ".*_wrist_yaw_link"
PELVIS_BODY_REGEX = ["pelvis", "torso_link"]
FOOT_BODY_REGEX = ".*_ankle_roll_link"

T_TASK = 3.0                  
SQUAT_PERIOD = 2.0 * T_TASK     
PHASE_OFFSET = 0.5              

STAND_HIP_PITCH, SQUAT_HIP_PITCH = -0.10, -2.10
STAND_KNEE,      SQUAT_KNEE      = 0.30,  2.20
STAND_ANKLE,     SQUAT_ANKLE     = -0.20, -0.75
STAND_HEIGHT,    SQUAT_HEIGHT    = 0.73,  0.39
TORSO_STAND_PITCH, TORSO_SQUAT_PITCH = 0.00, 0.65

STAND_ABDUCTION, SQUAT_ABDUCTION = 0.00, 0.18
STAND_WIDTH,     SQUAT_WIDTH     = 0.20, 0.28

SQUAT_SHOULDER_PITCH = 0.30
SQUAT_ELBOW          = 1.00


# ===========================================================================
# SceneEntityCfg
# ===========================================================================
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


# ===========================================================================
# Scene 
# ===========================================================================
@configclass
class SquatStandLiftSceneCfg(InteractiveSceneCfg):
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

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2, dynamic_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.4, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.34, 0.0, 0.1)),
    )

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


# ===========================================================================
# Commands 
# ===========================================================================
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=True,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


# ===========================================================================
# Actions 
# ===========================================================================
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


# ===========================================================================
# Observations
# ===========================================================================
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        squat_phase = ObsTerm(
            func=mdp.squat_phase_obs,
            params=dict(period=SQUAT_PERIOD, phase_offset=PHASE_OFFSET),
        )

        box_rel = ObsTerm(func=mdp.box_position_in_base_frame)
        box_dist_heading = ObsTerm(func=mdp.box_distance_and_heading)
        hand_pos = ObsTerm(
            func=mdp.hand_positions_in_base_frame,
            params={"hand_body_names": HAND_BODY_NAMES},
        )
        box_in_hands = ObsTerm(
            func=mdp.box_in_hand_frame,
            params={"hand_body_names": HAND_BODY_NAMES},
        )
        hand_touch = ObsTerm(
            func=mdp.hand_contact_flags,
            params={"sensor_cfg": SceneEntityCfg("hand_contact", body_names=[HAND_BODY_REGEX])},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        pass

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ===========================================================================
# Events 
# ===========================================================================
@configclass
class EventsCfg:
    reset_robot = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_box = EventTerm(
        func=mdp.reset_box_pose_uniform, mode="reset",
        params=dict(
            x_range=(0.30, 0.38),
            y_range=(-0.03, 0.03),
            z=0.10,
            yaw_range=(-0.1, 0.1),
        ),
    )


# ===========================================================================
# Rewards
# ===========================================================================
@configclass
class SquatStandLiftRewardsCfg:
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

    waist_pitch_pen = RewTerm(
        func=mdp.waist_pitch_penalty, weight=4.0,
        params=dict(max_abs=0.10, std=0.12, robot_cfg=WAIST_PITCH_CFG),
    )
    torso_roll_pen = RewTerm(
        func=mdp.torso_roll_penalty, weight=1.0,
        params=dict(std=0.15, robot_cfg=SceneEntityCfg("robot")),
    )
    wrist_pen = RewTerm(
        func=mdp.wrist_neutral_penalty, weight=3.0,
        params=dict(max_abs=0.15, std=0.25, robot_cfg=WRIST_CFG),
    )


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

    ang_vel_xy   = RewTerm(func=mdp.ang_vel_xy_l2,     weight=-0.02)
    action_rate  = RewTerm(func=mdp.action_rate_l2,    weight=-0.005)
    joint_acc    = RewTerm(func=mdp.joint_acc_l2,      weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2,  weight=-1.0e-6)
    dof_pos_lim  = RewTerm(func=mdp.joint_pos_limits,  weight=-1.0)


# ===========================================================================
# Terminations
# ===========================================================================
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(
        func=mdp.bad_orientation, params=dict(limit_angle=1.2),
    )
    collapsed = DoneTerm(
        func=mdp.root_height_below_minimum,
        params=dict(minimum_height=0.20, asset_cfg=SceneEntityCfg("robot")),
    )


# ===========================================================================
# Env
# ===========================================================================
@configclass
class G1SquatStandLiftEnvCfg(ManagerBasedRLEnvCfg):
    scene: SquatStandLiftSceneCfg = SquatStandLiftSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    rewards: SquatStandLiftRewardsCfg = SquatStandLiftRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = T_TASK   
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 1.2)

        default = dict(self.scene.robot.init_state.joint_pos or {})
        default["left_hip_pitch_joint"]    = SQUAT_HIP_PITCH
        default["right_hip_pitch_joint"]   = SQUAT_HIP_PITCH
        default[".*_knee_joint"]           = SQUAT_KNEE
        default[".*_ankle_pitch_joint"]    = SQUAT_ANKLE
        default[".*_shoulder_pitch_joint"] = SQUAT_SHOULDER_PITCH
        default[".*_elbow_joint"]          = SQUAT_ELBOW
        self.scene.robot.init_state.joint_pos = default
        self.scene.robot.init_state.pos = (0.0, 0.0, SQUAT_HEIGHT)

        print(f">>> SquatStandLift: T_task={T_TASK}s, period={SQUAT_PERIOD}s, phase_offset={PHASE_OFFSET}")
        print(f"    spawn: knee={SQUAT_KNEE} height={SQUAT_HEIGHT}m torso_pitch={TORSO_SQUAT_PITCH}rad")
        print(f"    spawn arm (PROVISIONAL): shoulder_pitch={SQUAT_SHOULDER_PITCH} elbow={SQUAT_ELBOW}")
        print(f"    goal:  knee={STAND_KNEE} height={STAND_HEIGHT}m")


@configclass
class G1SquatStandLiftEnvCfg_PLAY(G1SquatStandLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.reset_box.params["x_range"] = (0.34, 0.34)
        self.events.reset_box.params["y_range"] = (0.0, 0.0)
        self.events.reset_box.params["yaw_range"] = (0.0, 0.0)
