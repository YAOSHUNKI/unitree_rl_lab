from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.pickup_carry.mdp as mdp
from .pickup_carry_env_cfg import G1PickupCarryEnvCfg, FOOT_BODY_REGEX

SQUAT_PERIOD = 6.0   # 秒。ゆったり: 3秒下降 + 3秒上昇


@configclass
class PeriodicSquatRewardsCfg:
    """周期スクワット(立ち↔しゃがみ). 立ち姿勢からスタート、周期に必ず追従。

    要点:
      - 立ちからスタートなので寝転がりへの経路がまず遠い
      - 周期目標があるので立ちっぱなしは報酬半減
      - upright 必須で寝転がりは報酬ゼロ + 早期終了
    """
    # === 周期主報酬(全部位相ゲート付き = 寝転がりで報酬0) ===
    period_height = RewTerm(
        func=mdp.periodic_height_target, weight=15.0,
        params=dict(period=SQUAT_PERIOD, stand_height=0.75, squat_height=0.50, std=0.10),
    )
    period_knee = RewTerm(
        func=mdp.periodic_knee_target, weight=10.0,
        params=dict(period=SQUAT_PERIOD, stand_knee=0.15, squat_knee=1.3),
    )
    period_hip = RewTerm(
        func=mdp.periodic_hip_pitch_target, weight=6.0,
        params=dict(period=SQUAT_PERIOD, stand_hip=-0.1, squat_hip=-0.7),
    )

    # === 立位維持と転倒罰(寝転がり対策) ===
    upright = RewTerm(func=mdp.upright_bonus, weight=10.0)      # 立ってれば +10/step
    fallen  = RewTerm(func=mdp.fallen_penalty, weight=25.0,
                      params=dict(tilt_threshold=-0.3, height_threshold=0.30))

    # === 開脚抑制 ===
    hip_abduct   = RewTerm(func=mdp.hip_abduction_penalty, weight=2.0)
    hip_roll_mag = RewTerm(func=mdp.hip_roll_magnitude_penalty, weight=2.0)
    feet_width = RewTerm(
        func=mdp.feet_lateral_distance_penalty, weight=5.0,
        params=dict(max_stance_width=0.30, foot_body_names=[FOOT_BODY_REGEX]),
    )
    leg_sym = RewTerm(func=mdp.leg_symmetry_penalty, weight=0.3)

    # === 正則化(最小限) ===
    ang_vel_xy   = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02)
    action_rate  = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_acc    = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    dof_pos_lim  = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)


@configclass
class G1PeriodicSquatEnvCfg(G1PickupCarryEnvCfg):
    rewards: PeriodicSquatRewardsCfg = PeriodicSquatRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 速度指令ゼロ固定
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0

        # 2周期(12秒)分のエピソード
        self.episode_length_s = 12.0

        # 位相観測(policy が今どの位相にいるか知る必要がある)
        self.observations.policy.squat_phase = ObsTerm(
            func=mdp.squat_phase_obs,
            params=dict(period=SQUAT_PERIOD),
        )

        # ==== warm start は default(立ち姿勢)のまま。深しゃがみから始めない ====
        # → 寝転がりへの経路が遠くなる

        # ==== 終了条件 ====
        # 偶発的な pelvis 接触では終了しない(深しゃがみで pelvis がふとした瞬間触れても続行)
        self.terminations.base_contact = None
        # 明らかに転倒したら終了
        self.terminations.fell_over = DoneTerm(
            func=mdp.bad_orientation,
            params=dict(limit_angle=1.2),   # 約 69°
        )

        print(">>> PeriodicSquat: period={}s, warm-start=default(standing)".format(SQUAT_PERIOD))
        print(">>> terminations: base_contact=disabled, fell_over=bad_orientation(1.2)")


@configclass
class G1PeriodicSquatEnvCfg_PLAY(G1PeriodicSquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
