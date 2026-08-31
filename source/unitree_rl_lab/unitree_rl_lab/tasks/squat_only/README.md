# PeriodicSquat タスク

Unitree G1 29DOF に「立ち姿勢で spawn し、その場で立ち ↔ 完全しゃがみを
周期的に繰り返す」動作を学習させる env。腕は肩の高さで前へ伸ばした姿勢に固定する。

- タスク ID : `Unitree-G1-29dof-PeriodicSquat` / `-Play`
- env cfg   : `robots/g1/29dof/squat_only_env_cfg.py` 

---

## フォルダ構成

```
squat_only/
├── __init__.py
├── README.md                                
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py                    G1PeriodicSquatPPORunnerCfg
├── mdp/
│   ├── __init__.py                          
│   ├── events.py                            独自イベントなし
│   ├── observations.py                      squat_phase_obs
│   └── rewards.py                           
└── robots/
    └── g1/29dof/
        ├── __init__.py                      gym.register
        └── squat_only_env_cfg.py            
```
---

## 動作フェーズ

エピソード長 `episode_length_s = 12.0 s`。周期 `SQUAT_PERIOD = 6.0 s` の姿勢追従を
`phase_offset = 0.0` で開始し、位相 `phi(t) = (t mod 6) / 6` を `t=0` の立ちから
`t=3` の完全しゃがみまで進めて `t=6` で立ちへ戻す。これを 2 周期繰り返す。

```
t=0    phi=0.0  depth=0.0  立ち (spawn)
t=1.5  phi=0.25 depth=0.5  中間
t=3.0  phi=0.5  depth=1.0  完全しゃがみ
t=6.0  phi=1.0  depth=0.0  立ちへ戻る
```

`depth(phi) = 0.5 - 0.5 cos(2π phi)` を目標軌道とし、姿勢追従群 7 種と
定位置保持・胴姿勢の有界ペナルティ群の複合で学習する。追従項は
`_relative_track` で「何もしない状態」を 0 点に正規化してある。

---

## 参照姿勢

| 量 | spawn (立) | 目標 (深) | 備考 |
|---|---:|---:|---|
| hip_pitch     | -0.10 | -2.10 | rad |
| knee          |  0.30 |  2.20 | rad, 126 度 |
| ankle_pitch   | -0.20 | -0.75 | soft 限界 -0.803 |
| pelvis_z      |  0.73 |  0.39 | m |
| torso_pitch   |  0.00 |  0.65 | 前傾 37 度、重心を踵より前へ |
| hip_roll 許容  |  0.00 |  0.18 | rad、片側許容 |
| 足幅          |  0.20 |  0.28 | m、片側許容 |
| shoulder_pitch | -1.6236 | —  | rad、固定 |
| elbow          |  1.4668 | —  | rad、固定 |

重心 (運動学の検算) : COM_x +0.066 m、踵まで 0.126 m、つま先まで 0.084 m

---

## 腕

- 学習対象外。`use_default_offset=True` なので初期姿勢が action 0 の姿勢になる
- shoulder_pitch = -1.6236 (幾何 -1.4911 − 自重たわみ 0.1325)
- elbow = 1.4668 (幾何 +1.4368 + 自重たわみ 0.0300)
- shoulder_roll / shoulder_yaw / wrist_* = 0.0 (元の ±0.25 / ±0.15 を 0 に)
- 実際に落ち着く位置 : 手先が肩と同じ高さ、前方 0.382 m
- 質量 7.04 kg (全体の 20%)。真下 → 水平前方で重心が前へ 2.0 cm 移動する

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

policy には Isaac Lab の標準ノイズを付与。critic は同構造だが Play で
ノイズ無効化。行動は全 29 関節の目標角で、`scale` は hip_pitch / knee が 0.8、
ankle_pitch が 0.5、その他 25 関節が 0.25。

---

## 報酬構成

### 姿勢追従 (周期の位相追従)

| 項目 | 関数 | weight | 内容 |
|---|---|---:|---|
| pose_coarse | `squat_pose_tracking` (σ=1.80) | 5.0 | 脚 3 関節群 + 側方群を参照姿勢へ (粗) |
| pose_fine   | `squat_pose_tracking` (σ=0.35) | 8.0 | 同 (精度) |
| height_track | `squat_height_tracking` | 3.0 | 骨盤高さの位相追従 |
| torso_pitch  | `torso_pitch_tracking`  | 3.0 | 胴前傾。重心の前後位置を直接支配 |
| squat_shortfall_pen | `squat_depth_shortfall_penalty` | 3.0 | 深さ不足を罰する。depth ゲート |
| hip_abduction_pen | `hip_abduction_tracking` | 3.0 | 開脚抑制、片側許容 |
| stance_pen   | `stance_width_penalty_phased` | 2.5 | 足幅抑制、片側許容 |

### 腕の固定

| 項目 | 関数 | weight | 内容 |
|---|---|---:|---|
| knee_clear_pen | `hands_knee_clearance_penalty` | 5.0 | 手が膝にめり込むのを防ぐ |
| arm_hold_pen   | `joint_default_deviation_penalty` | 4.0 | 腕 14 関節を初期姿勢に。margin 0.25 で自重たわみを無罰 |

### 定位置保持 (すべて有界ペナルティ、静止で 0)

| 項目 | 関数 | weight | 内容 |
|---|---|---:|---|
| drift_pen    | `drift_penalty` | 3.0 | spawn 位置からの水平距離 |
| heading_pen  | `heading_penalty` | 2.0 | 初期ヨー向きからのずれ |
| yaw_rate_pen | `yaw_rate_penalty` | 2.0 | ヨー角速度。`ang_vel_xy_l2` は z を見ない |
| slip_pen     | `feet_slip_penalty` | 1.0 | 接地している足の水平速度 |
| speed_pen    | `base_speed_penalty` | 0.5 | 胴の水平速度 (z は無視) |

### 胴姿勢

| 項目 | 関数 | weight | 内容 |
|---|---|---:|---|
| waist_pitch_pen | `waist_pitch_penalty` | 4.0 | 腰から上の反り抑制 |
| backlean_pen    | `torso_backlean_penalty` | 3.0 | 体全体を後ろに反らすのを罰する |
| torso_roll_pen  | `torso_roll_penalty`  | 3.0 | 胴の左右傾き抑制 |
| knee_lateral_pen | `lateral_offset_penalty` | 3.0 | 両膝が同じ側に寄るのを罰する |
| feet_lateral_pen | `lateral_offset_penalty` | 2.0 | 両足が同じ側に寄るのを罰する |
| ankle_roll_pen  | `joint_default_deviation_penalty` | 2.0 | 足首の左右傾き |

### 上体・接地 (合計を正に保つ床)

| 項目 | 関数 | weight | 内容 |
|---|---|---:|---|
| upright  | `upright_bonus` | 3.0 | `(1 - g_z) / 2`、立位 1.0 / 逆さま 0 |
| grounded | `feet_grounded` | 3.0 | 接地している足の割合 |

### 正則化

| 項目 | 関数 | weight | 内容 |
|---|---|---:|---|
| ang_vel_xy   | `ang_vel_xy_l2`    | -0.02   | 角速度 |
| action_rate  | `action_rate_l2`   | -0.015  | 行動変化率 |
| joint_acc    | `joint_acc_l2`     | -2.5e-7 | 関節加速度 |
| joint_torque | `joint_torques_l2` | -1.0e-6 | 関節トルク |
| dof_pos_lim  | `joint_pos_limits` | -1.0    | 関節限界超過 (異常検知) |

---

## 終了条件

| 名前 | 関数 | 設定 | 意図 |
|---|---|---|---|
| time_out  | `time_out`                    | `episode_length_s = 12.0` | 通常の終了 (2 周期) |
| fell_over | `bad_orientation`             | limit_angle=1.2 | 約 69 度傾いたら終了 |
| collapsed | `root_height_below_minimum`   | 0.20 m | 骨盤沈み込みで終了 |

`base_contact` は無効化。深いしゃがみでの偶発的な骨盤接触を許すため。

---

## PPO 設定 (`agents/rsl_rl_ppo_cfg.py`)

`G1PeriodicSquatPPORunnerCfg`:

- experiment_name : `squat_only`
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
    --task Unitree-G1-29dof-PeriodicSquat \
    --num_envs 4096 --headless
```

再生:

```bash
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-PeriodicSquat \
    --checkpoint logs/rsl_rl/squat_only/<run>/model_<n>.pt
```
