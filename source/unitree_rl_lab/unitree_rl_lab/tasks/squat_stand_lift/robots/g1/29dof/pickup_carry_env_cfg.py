"""G1 29DOF whole-body pickup-and-carry env cfg.

The policy drives ALL 29 joints (legs + waist + arms + wrists) so the arms
are actually part of what is being learned.

Staged behaviour:
  1. Approach a box on the ground
  2. Face it and squat
  3. Bring both hands onto the box
  4. Grasp it (both hands in contact, near box centre)
  5. Stand back up while grasped (= lift the box)
  6. Walk while carrying it, tracking the base velocity command
"""

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


# --- knobs you may need to tweak for your USD -------------------------------
HAND_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
HAND_BODY_REGEX = ".*_wrist_yaw_link"
PELVIS_BODY_REGEX = ["pelvis", "torso_link"]
FOOT_BODY_REGEX = ".*_ankle_roll_link"
# ----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class PickupSceneCfg(InteractiveSceneCfg):
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

    # A 20 cm cube, 1 kg, orange.
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2, dynamic_friction=1.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.4, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.1)),
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

    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9)))


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
# Actions — whole-body (all 29 joints)
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

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


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@configclass
class EventsCfg:
    reset_robot = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_box = EventTerm(
        func=mdp.reset_box_pose_uniform,
        mode="reset",
        params=dict(x_range=(0.5, 0.9), y_range=(-0.2, 0.2), z=0.1, yaw_range=(-0.5, 0.5)),
    )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@configclass
class RewardsCfg:
    # 1. Approach
    approach = RewTerm(
        func=mdp.approach_box, weight=1.0,
        params=dict(target_distance=0.55, std=0.35),
    )
    face = RewTerm(func=mdp.face_box, weight=0.5)

    # 2. Squat
    squat = RewTerm(
        func=mdp.squat_when_near_box, weight=2.0,
        params=dict(target_height=0.50, near_threshold=0.75, std=0.10),
    )

    # 3. Hands on box
    hands_near = RewTerm(
        func=mdp.hands_near_box, weight=1.5,
        params=dict(std=0.15, hand_body_names=HAND_BODY_NAMES),
    )
    hands_touch = RewTerm(
        func=mdp.hands_contact_box, weight=1.5,
        params=dict(sensor_cfg=SceneEntityCfg("hand_contact", body_names=[HAND_BODY_REGEX])),
    )

    # 4. Grasp
    grasp = RewTerm(
        func=mdp.grasp_bonus, weight=3.0,
        params=dict(hand_body_names=HAND_BODY_NAMES, contact_sensor_name="hand_contact"),
    )

    # 5. Lift / stand up while grasped
    lift = RewTerm(
        func=mdp.lift_box, weight=4.0,
        params=dict(
            initial_z=0.1, target_lift=0.4, std=0.12,
            hand_body_names=HAND_BODY_NAMES, contact_sensor_name="hand_contact",
        ),
    )
    stand_up = RewTerm(
        func=mdp.stand_up_when_lifting, weight=2.0,
        params=dict(
            stand_height=0.75, std=0.10,
            hand_body_names=HAND_BODY_NAMES, contact_sensor_name="hand_contact",
        ),
    )

    # 6. Carry
    carry = RewTerm(
        func=mdp.carry_box_velocity, weight=2.0,
        params=dict(
            std=0.4,
            hand_body_names=HAND_BODY_NAMES, contact_sensor_name="hand_contact",
        ),
    )

    # Safety
    drop_pen = RewTerm(func=mdp.drop_box_penalty, weight=1.0)
    box_hit = RewTerm(
        func=mdp.box_collision_penalty, weight=0.5,
        params=dict(min_distance=0.30),
    )

    # Regularization
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_pos_lim = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    arm_default = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.05,
        params=dict(
            asset_cfg=SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"]
            )
        ),
    )
     # Posture shaping
    feet_width = RewTerm(
        func=mdp.feet_lateral_distance_penalty, weight=2.0,
        params=dict(max_stance_width=0.30, foot_body_names=[FOOT_BODY_REGEX]),
    )
    hip_abduct = RewTerm(func=mdp.hip_abduction_penalty, weight=0.5)
    knee_flex = RewTerm(
        func=mdp.knee_flexion_when_squatting, weight=1.5,
        params=dict(target_knee_angle=1.2, std=0.3, near_threshold=0.75),
    )
    leg_sym = RewTerm(func=mdp.leg_symmetry_penalty, weight=0.2)


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


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


@configclass
class G1PickupCarryEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickupSceneCfg = PickupSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.5, 2.5, 1.5)


@configclass
class G1PickupCarryEnvCfg_PLAY(G1PickupCarryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.reset_box.params["x_range"] = (0.7, 0.7)
        self.events.reset_box.params["y_range"] = (0.0, 0.0)
