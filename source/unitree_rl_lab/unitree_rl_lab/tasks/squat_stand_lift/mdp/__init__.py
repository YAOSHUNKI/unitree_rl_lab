"""SquatStandLift task の MDP は squat_only.mdp を共有する。

この __init__.py は薄い re-export のみ。実体は
``unitree_rl_lab.tasks.squat_only.mdp`` にある。
"""

from unitree_rl_lab.tasks.squat_only.mdp import *  # noqa: F401, F403
