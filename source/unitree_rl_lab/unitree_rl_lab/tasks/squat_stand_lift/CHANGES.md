# SquatStandLift タスクの追加変更点

Isaac Lab / unitree_rl_lab の `locomotion` タスクを雛形として、
`tasks/squat_stand_lift/` を新規追加。以下、置いた/書いたファイルとその要素。

## 新規タスクフォルダ

`source/unitree_rl_lab/unitree_rl_lab/tasks/squat_stand_lift/`

| ファイル | 内容 |
|---|---|
| `__init__.py` | パッケージ docstring。自己完結タスクである旨のみ |
| `README.md` | タスク概要・報酬一覧・学習コマンド・監視指標 |
| `mdp/__init__.py` | Isaac Lab 標準 mdp + locomotion mdp + 自パッケージの events/observations/rewards を re-export |
| `mdp/events.py` | `reset_box_pose_uniform` (箱を毎エピソード x/y/yaw ランダムで再配置) |
| `mdp/observations.py` | `squat_phase_obs` (位相の sin/cos)、`box_position_in_base_frame`、`box_distance_and_heading`、`hand_positions_in_base_frame`、`box_in_hand_frame`、`hand_contact_flags` |
| `mdp/rewards.py` | 姿勢追従群 (`squat_pose_tracking` 等)、箱把持系 (`hands_near_box`, `hands_contact_box`, `grasp_bonus`, `lift_box`, `stand_up_when_lifting`, `drop_box_penalty`)、定位置保持 (`drift_penalty`, `feet_slip_penalty`, `heading_penalty`, `base_speed_penalty`)、胴姿勢 (`waist_pitch_penalty`, `torso_roll_penalty`, `wrist_neutral_penalty`)、上体・接地 (`upright_bonus`, `feet_grounded`)、位相ヘルパ (`_squat_phase`, `_squat_depth` — `phase_offset` 引数付き) |
| `agents/rsl_rl_ppo_cfg.py` | `G1SquatStandLiftPPORunnerCfg` (experiment_name, network, PPO ハイパー) |
| `agents/__init__.py` | 空 |
| `robots/__init__.py`, `robots/g1/__init__.py` | 空 (パッケージ化) |
| `robots/g1/29dof/__init__.py` | `gym.register` で `Unitree-G1-29dof-SquatStandLift` と `-Play` を登録 |
| `robots/g1/29dof/squat_stand_lift_env_cfg.py` | Scene (地面・G1・**箱**・3 種コンタクトセンサ)、Commands (速度 0 固定)、Actions (29 関節)、Observations (policy/critic)、Events (`reset_robot` + `reset_box`)、Rewards (`SquatStandLiftRewardsCfg`)、Terminations (`time_out`/`fell_over`/`collapsed`)、`G1SquatStandLiftEnvCfg` (spawn 姿勢・箱位置・位相設定を `__post_init__` で確定)、`G1SquatStandLiftEnvCfg_PLAY` (num_envs=16, 箱固定位置) |

## 既存ファイルの変更

なし。`locomotion` タスクや `squat_only` タスクの下は一切触っていない。
上位パッケージ (`tasks/__init__.py`) の `import_packages` が
`squat_stand_lift/` を自動発見するので登録側の変更も不要。

## locomotion 雛形から変わったポイント (要素粒度)

| 要素 | locomotion (雛形) | SquatStandLift |
|---|---|---|
| Scene | 地面 + G1 + 足コンタクト | 地面 + G1 + **箱** + 足/手/胴 コンタクト |
| Commands | 速度追従 (ランダム目標) | 速度 0 固定 (定位置保持) |
| Actions | 全関節 | 全関節 (同じ) |
| Observations | ロコモ標準 | + 位相 + 箱・手先観測 5 種 |
| Events | reset のみ | + `reset_box_pose_uniform` |
| Rewards | 速度追従・生存・正則化 | 姿勢追従 6 + 箱把持 6 + 定位置保持 4 + 胴姿勢 3 + 上体 2 + 正則化 5 |
| Terminations | 転倒 | 転倒 + 骨盤沈み込み |
| spawn 姿勢 | 立位デフォルト | 完全しゃがみ (関節角 6 種と骨盤高 0.39 m を上書き) |
| エピソード長 | 20 s | 3 s (半周期で切って戻り位相に入らせない) |
| PPO experiment_name | ロコモ名 | `g1_squat_stand_lift` |

## 依存関係

- `unitree_rl_lab.assets.robots.unitree.UNITREE_G1_29DOF_CFG` (G1 の USD 定義) を利用
- `isaaclab.envs.mdp` / `isaaclab_tasks.manager_based.locomotion.velocity.mdp` /
  `unitree_rl_lab.tasks.locomotion.mdp` の共通 mdp 関数を `mdp/__init__.py` 経由で使用
- 他のカスタムタスク (`squat_only`, `pickup_carry`) への import は **なし**
