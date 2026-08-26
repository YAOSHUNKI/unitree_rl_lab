"""Isaac Lab environment: G1-29dof squats in front of a box.

Scene contents
--------------
* Ground plane (flat, keeps training simple for the vision-based task).
* Unitree G1 (29 dof).
* One rigid Cuboid ("box") placed in front of the robot; pose randomized on
  reset via a custom event term.
* A head-mounted ``TiledCamera`` producing depth (RealSense D435-like FOV
  and clip range) that feeds directly into the policy observation.

Observation groups
------------------
* ``policy``  : proprio (5-step history) + flattened depth image.
* ``critic``  : proprio + privileged box pose (asymmetric actor-critic).

Reward
------
See ``mdp/rewards.py`` for the shaping. Summary:
  approach_box + face_box + squat_when_near_box + hold_still_when_squatting
  + standard locomotion regularizers.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.manipulation import mdp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# RealSense D435 (approx.): H-FOV 69deg, V-FOV 42deg, min depth ~0.1 m.
CAMERA_WIDTH = 48
CAMERA_HEIGHT = 48
CAMERA_HFOV_DEG = 69.0
CAMERA_CLIP = (0.1, 3.0)

# Box geometry
BOX_SIZE = (0.30, 0.30, 0.30)
BOX_MASS = 5.0

# Target squat
TARGET_STAND_OFF = 0.5  # m from base to box centre (xy)
TARGET_SQUAT_HEIGHT = 0.55  # G1 stand height ~0.78 m; squat ~0.55 m


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class SquatBeforeBoxSceneCfg(InteractiveSceneCfg):
    """Robot + box + head camera + flat ground."""

    # Flat ground — vision task is hard enough; add rough terrain later.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # Robot
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Target box (rigid, static-ish mass so the robot can lean on it if trained
    # to; increase mass for a truly immovable target).
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.2), roughness=0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, BOX_SIZE[2] / 2.0)),
    )

    # Head-mounted RealSense-like camera.
    # We attach it to torso_link with a forward-and-up offset to approximate
    # a head/chest-mounted D435 on G1.
    head_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.10, 0.0, 0.55),
            rot=(0.5, -0.5, 0.5, -0.5),  # x-forward, z-up world -> camera optical frame
            convention="ros",
        ),
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=CAMERA_CLIP,
        ),
        data_types=["depth"],
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        update_period=0.05,  # 20 Hz — realistic RealSense-ish rate
    )

    # Foot contact (used for gait-style regularization if desired)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    # Lighting
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ---------------------------------------------------------------------------
# MDP components
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Proprio (identical to velocity task minus velocity commands)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        # Depth image from RealSense-like camera (flattened, normalized to [0,1])
        depth_image = ObsTerm(
            func=mdp.depth_image_flat,
            params={"sensor_cfg": SceneEntityCfg("head_camera"), "clip": CAMERA_CLIP},
        )

        def __post_init__(self):
            # Proprio + depth image; do NOT stack camera frames (obs blows up).
            self.history_length = 1
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        # Full proprio incl. linear velocity (privileged)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        # Privileged: exact box pose relative to the base
        box_rel = ObsTerm(func=mdp.box_position_in_base_frame)
        box_scalar = ObsTerm(func=mdp.box_distance_and_heading)

        def __post_init__(self):
            self.history_length = 1
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    # Startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # Reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.5, 0.5)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-0.5, 0.5)},
    )
    reset_box = EventTerm(
        func=mdp.reset_box_pose_uniform,
        mode="reset",
        params={
            "x_range": (0.8, 1.8),
            "y_range": (-0.5, 0.5),
            "z": BOX_SIZE[2] / 2.0,
            "yaw_range": (-3.14, 3.14),
            "object_cfg": SceneEntityCfg("box"),
        },
    )

    # Interval — occasional push to keep policies robust
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    # ---- Task ----
    approach = RewTerm(
        func=mdp.approach_box,
        weight=1.5,
        params={"target_distance": TARGET_STAND_OFF, "std": 0.4},
    )
    facing = RewTerm(func=mdp.face_box, weight=0.5)
    squat = RewTerm(
        func=mdp.squat_when_near_box,
        weight=3.0,
        params={
            "target_height": TARGET_SQUAT_HEIGHT,
            "stand_height": 0.78,
            "near_threshold": TARGET_STAND_OFF + 0.2,
            "std": 0.08,
        },
    )
    still_when_close = RewTerm(
        func=mdp.hold_still_when_squatting,
        weight=0.5,
        params={"near_threshold": TARGET_STAND_OFF + 0.2},
    )
    no_collision = RewTerm(
        func=mdp.box_collision_penalty, weight=5.0, params={"min_distance": 0.32}
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # ---- Regularization (mirrors velocity task) ----
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    flat_orient = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Only kill on truly falling — 0.2 m is basically prone. Squatting to 0.55
    # is safe.
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.25})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


# ---------------------------------------------------------------------------
# Env cfg
# ---------------------------------------------------------------------------


@configclass
class SquatBeforeBoxEnvCfg(ManagerBasedRLEnvCfg):
    # Camera envs cost a lot of GPU memory — start moderate.
    scene: SquatBeforeBoxSceneCfg = SquatBeforeBoxSceneCfg(num_envs=2048, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 15.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Sensor tick rates
        self.scene.contact_forces.update_period = self.sim.dt
        # Camera runs at 20 Hz regardless of decimation; already set in cfg.

        # Consistency sanity check
        assert self.scene.head_camera.height == CAMERA_HEIGHT
        assert self.scene.head_camera.width == CAMERA_WIDTH
        # (kept explicit so an accidental resolution bump is caught early)
        _ = math.radians(CAMERA_HFOV_DEG)


@configclass
class SquatBeforeBoxPlayEnvCfg(SquatBeforeBoxEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        # Tighter box spawn so the play scene is easier to eyeball.
        self.events.reset_box.params["x_range"] = (1.0, 1.2)
        self.events.reset_box.params["y_range"] = (-0.2, 0.2)
