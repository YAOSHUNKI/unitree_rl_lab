"""SquatStandLift 用の PPO 設定。学習ハイパーパラメータは共有する。"""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.squat_only.agents.rsl_rl_ppo_cfg import (
    G1PickupCarryPPORunnerCfg,
)


@configclass
class G1SquatBoxLiftPPORunnerCfg(G1PickupCarryPPORunnerCfg):
    """箱把持・立ち上がりタスク専用のログ出力設定。"""

    experiment_name = "squat_box_lift"
