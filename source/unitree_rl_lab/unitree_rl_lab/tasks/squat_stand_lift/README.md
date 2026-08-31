# SquatStandLift タスク

Unitree G1 29DOF に「しゃがみ姿勢で spawn し、目の前の 20 cm 立方体を
両手で掴んで立ち上がる」動作を学習させる env。

- タスク ID : `Unitree-G1-29dof-SquatStandLift` / `-Play`
- env cfg   : `robots/g1/29dof/squat_stand_lift_env_cfg.py` 

---

## フォルダ構成

```
squat_stand_lift/
├── __init__.py
├── README.md                                
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py                    G1SquatStandLiftPPORunnerCfg
├── mdp/
│   ├── __init__.py                          
│   ├── events.py                            reset_box_pose_uniform
│   ├── observations.py                      squat_phase_obs, box_*, hand_*
│   └── rewards.py                           
└── robots/
    └── g1/29dof/
        ├── __init__.py                      gym.register
        └── squat_stand_lift_env_cfg.py      
```
---

## 動作フェーズ

エピソード長 `T_TASK = 3.0 s`。周期 `SQUAT_PERIOD = 6.0 s` の姿勢追従を
`phase_offset = 0.5` で開始し、位相 `phi(t) = 0.5 + t/6` を `t=0` の完全
しゃがみから `t=3` の立ちまで単調に進める。

```
t=0    phi=0.5  depth=1.0  完全しゃがみ (spawn)
t=1.5  phi=0.75 depth=0.5  中間
t=3.0  phi=1.0  depth=0.0  立ち位相 (エピソード終了)
```

`depth(phi) = 0.5 - 0.5 cos(2π phi)` を目標軌道とし、姿勢追従群 6 種と
箱把持・持ち上げ報酬 6 種の複合で学習する。

---

## 参照姿勢

| 量 | spawn (深) | 目標 (立) | 備考 |
|---|---:|---:|---|
| hip_pitch     | -2.10 | -0.10 | rad |
| knee          |  2.20 |  0.30 | rad, 126 度 |
| ankle_pitch   | -0.75 | -0.20 | soft 限界 -0.803 |
| pelvis_z      |  0.39 |  0.73 | m |
| torso_pitch   |  0.65 |  0.00 | 前傾 37 度、重心を踵より前へ |
| hip_roll 許容  |  0.18 |  0.00 | rad、片側許容 |
| 足幅          |  0.28 |  0.20 | m、片側許容 |
| shoulder_pitch | 0.30 |  —    | rad |
| elbow          | 1.00 |  —    | rad |

---

## 箱

- 20 cm 立方体、質量 1 kg、静止摩擦 1.2 / 動摩擦 1.0
- spawn 位置 `x ∈ [0.30, 0.38] m, y ∈ [-0.03, 0.03] m, z = 0.10 m, yaw ∈ [-0.1, 0.1] rad`
- 手 → 箱 の余裕 : 参照姿勢の運動学から 0.146 m
- Play では固定位置 (0.34, 0.0)

---

## 観測

policy / critic 両方に同じ項:

| 項目 | 内容 |
|---|---|
| base_ang_vel        | 胴の角速度 |
| projected_gravity   | 重力の胴座標成分 |
| velocity_commands   | 常に 0 (定位置保持) |
| joint_pos / joint_vel | 全 29 関節の相対値 |
| actions             | 前ステップの行動 |
| squat_phase         | (sin(2π phi), cos(2π phi)) |
| box_rel             | 箱の骨盤座標位置 |
| box_dist_heading    | 箱までの距離と方位 |
| hand_pos            | 両手の骨盤座標位置 |
| box_in_hands        | 箱の各手座標位置 |
| hand_touch          | 両手の接触フラグ |

policy には Isaac Lab の標準ノイズを付与。critic は同構造だが Play で
ノイズ無効化。

---

## 報酬構成

### 姿勢追従 (半周期の PeriodicSquat 追従)

| 項目 | 関数 | weight | 内容 |
|---|---|---:|---|
| pose_coarse | `squat_pose_tracking` (σ=0.85) | 4.0 | 脚 3 関節群 + 側方群を参照姿勢へ (粗) |
| pose_fine   | `squat_pose_tracking` (σ=0.35) | 8.0 | 同 (精度) |
| height_track | `squat_height_tracking` | 3.0 | 骨盤高さの位相追従 |
| torso_pitch  | `torso_pitch_tracking`  | 3.0 | 胴前傾。重心の前後位置を直接支配 |
| hip_abduction_pen | `hip_abduction_tracking` | 6.0 | 開脚抑制、片側許容 |
| stance_pen   | `stance_width_penalty_phased` | 5.0 | 足幅抑制、片側許容 |

### 箱把持・持ち上げ

| 項目 | 関数 | weight | 内容 |
|---|---|---:|---|
| hands_near   | `hands_near_box`    | 1.5 | 両手 → 箱中心の距離 |
| hands_touch  | `hands_contact_box` | 1.5 | 両手接触フラグ |
| grasp        | `grasp_bonus`       | 3.0 | 両手接触かつ手が箱中心近く |
| lift         | `lift_box`          | 4.0 | 箱の z 増分 (掴んでいる時のみ) |
| stand_up     | `stand_up_when_lifting` | 3.0 | 掴んで立ち上がった時のみ加点 |
| drop_pen     | `drop_box_penalty`  | 1.0 | 箱が閾値高より低ければ罰 |

### 定位置保持 (すべて有界ペナルティ、静止で 0)

| 項目 | 関数 | weight | 内容 |
|---|---|---:|---|
| drift_pen    | `drift_penalty` | 1.5 | spawn 位置からの水平距離 |
| slip_pen     | `feet_slip_penalty` | 1.5 | 接地している足の水平速度 |
| heading_pen  | `heading_penalty` | 1.0 | 初期ヨー向きからのずれ |
| speed_pen    | `base_speed_penalty` | 0.5 | 胴の水平速度 (z は無視) |

### 胴姿勢

| 項目 | 関数 | weight | 内容 |
|---|---|---:|---|
| waist_pitch_pen | `waist_pitch_penalty` | 4.0 | 胴の反り抑制 |
| torso_roll_pen  | `torso_roll_penalty`  | 1.0 | 胴の左右傾き抑制 |
| wrist_pen       | `wrist_neutral_penalty` | 3.0 | 手首を中立に |

### 上体・接地 (合計を正に保つ床)

| 項目 | 関数 | weight | 内容 |
|---|---|---:|---|
| upright  | `upright_bonus` | 0.5 | `(1 - g_z) / 2`、立位 1.0 / 逆さま 0 |
| grounded | `feet_grounded` | 0.5 | 接地している足の割合 |

### 正則化

| 項目 | 関数 | weight | 内容 |
|---|---|---:|---|
| ang_vel_xy   | `ang_vel_xy_l2`    | -0.02   | 角速度 |
| action_rate  | `action_rate_l2`   | -0.005  | 行動変化率 |
| joint_acc    | `joint_acc_l2`     | -2.5e-7 | 関節加速度 |
| joint_torque | `joint_torques_l2` | -1.0e-6 | 関節トルク |
| dof_pos_lim  | `joint_pos_limits` | -1.0    | 関節限界超過 (異常検知) |

---

## 終了条件

| 名前 | 関数 | 設定 | 意図 |
|---|---|---|---|
| time_out  | `time_out`                    | `episode_length_s = 3.0` | 通常の終了 |
| fell_over | `bad_orientation`             | limit_angle=1.2 | 約 69 度傾いたら終了 |
| collapsed | `root_height_below_minimum`   | 0.20 m | 骨盤沈み込みで終了 |


---

## PPO 設定 (`agents/rsl_rl_ppo_cfg.py`)

`G1SquatStandLiftPPORunnerCfg`:

- experiment_name : `g1_squat_stand_lift`
- num_steps_per_env : 24
- max_iterations : 40000 
- save_interval : 200
- policy : hidden [512, 256, 128], activation elu, init_noise_std 0.35
- algorithm : clip 0.2, entropy_coef 0.005, epochs 5, minibatches 4,
  lr 3e-4 (adaptive, desired_kl 0.01), gamma 0.99, lam 0.95

---

## 学習コマンド

```bash
conda activate env_isaaclab
cd ~/unitree_rl_lab
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-SquatStandLift \
    --num_envs 64 --max_iterations 3000
```

再生:

```bash
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-SquatStandLift \
    --checkpoint logs/rsl_rl/g1_squat_stand_lift/<run>/model_<n>.pt
```
