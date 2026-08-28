"""SquatStandLift task (G1 29DOF).

しゃがみ状態から箱を掴んで立ち上がる単発動作を学習するタスク。
PeriodicSquat の姿勢追従を半周期だけ使ってしゃがみ->立ちの単調追従に転用し、
squat_only の箱把持・持ち上げ報酬を後半フェーズにゲート付きで足す。

MDP・agents は squat_only を共有 (案 A)。
"""
