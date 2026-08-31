"""G1 29DOF スクワットタスクの基底設定 (シーン / 行動 / 観測 / イベント / 終了)。

もとは `pickup_carry_env_cfg.py` の `G1PickupCarryEnvCfg` にあったものを、
箱タスクの削除にあたって切り出した。**箱に関わる要素は一切含まない。**

報酬は各タスク側 (`squat_only_env_cfg.py`) が丸ごと差し替える。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unitree_rl_lab.tasks.squat_only.mdp as mdp
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG

# --- ボディ名 (USD に合わせて調整) -----------------------------------------
HAND_BODY_REGEX = ".*_wrist_yaw_link"
PELVIS_BODY_REGEX = ["pelvis", "torso_link"]
FOOT_BODY_REGEX = ".*_ankle_roll_link"


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


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
# Actions — 全身 29 関節
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    # scale は各タスクの __post_init__ で関節ごとに上書きする (落とし穴 16)。
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
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # 手先位置・接触は箱タスク用の観測だったので削除。
        # 肩・肘の角度は joint_pos に含まれるので、手先位置は方策側で復元できる。

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
class G1SquatBaseEnvCfg(ManagerBasedRLEnvCfg):
    """報酬を持たない基底。派生側で rewards を必ず定義すること。"""

    scene: SquatSceneCfg = SquatSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.5, 2.5, 1.5)
