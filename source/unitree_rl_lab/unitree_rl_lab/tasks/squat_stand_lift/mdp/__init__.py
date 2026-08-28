"""SquatStandLift task の MDP は pickup_carry.mdp を共有する。

この __init__.py は薄い re-export のみ。実体は
``unitree_rl_lab.tasks.pickup_carry.mdp`` にある。
"""

from unitree_rl_lab.tasks.pickup_carry.mdp import *  # noqa: F401, F403
