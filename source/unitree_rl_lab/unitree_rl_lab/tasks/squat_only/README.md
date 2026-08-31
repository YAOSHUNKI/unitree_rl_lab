# PeriodicSquat タスク

Unitree G1 29DOF に「その場で立ち ↔ 完全しゃがみを周期的に繰り返す」動作を
学習させる env。
- タスク ID : `Unitree-G1-29dof-PeriodicSquat` / `-Play`
- env cfg   : `robots/g1/29dof/squat_only_env_cfg.py`




---

## フォルダ構成と変更内容

```
squat_only/
├── __init__.py                              
├── README.md                                
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py                  G1SquatBasePPORunnerCfg
│                                            G1PeriodicSquatPPORunnerCfg
├── mdp/
│   ├── __init__.py                       Isaac Lab / locomotion の mdp を再エクスポート
│   ├── events.py                    
│   ├── observations.py                
│   └── rewards.py                    
└── robots/
    └── g1/29dof/
        ├── __init__.py                   gym.register（
        └── squat_only_env_cfg.py    
```

## 動作

エピソード長 `episode_length_s = 12.0 s`（周期 6.0 s × 2）。
位相 `phi(t) = (t mod 6) / 6`、深さ `depth = 0.5 - 0.5 cos(2π phi)`。

```
phi=0.0  depth=0.0  立ち     (spawn)
phi=0.5  depth=1.0  完全しゃがみ
phi=1.0  depth=0.0  立ちに戻る
```

すべての姿勢目標は `stand + (squat - stand) * depth` で補間する。

---

## 参照姿勢

| 量 | 立ち | 完全しゃがみ | soft 限界 | 備考 |
|---|---:|---:|---|---|
| hip_pitch   | -0.10 | -2.10 | -2.26 | rad |
| knee        |  0.30 |  2.20 |  2.73 | rad、126 度 |
| ankle_pitch | -0.20 | -0.75 | -0.803 | 余裕 0.053 |
| pelvis_z    |  0.73 |  0.39 | — | m |
| torso_pitch |  0.00 |  0.65 | — | 前傾 37 度、重心を踵より前へ |
| hip_roll 許容 | 0.00 | 0.18 | — | rad、片側許容 |
| 足幅        |  0.20 |  0.28 | — | m、片側許容 |


### 腕（固定・学習対象外）

| 関節 | 値 | 備考 |
|---|---:|---|
| shoulder_pitch | **-1.6236** | 幾何 -1.4911 − 自重たわみ 0.1325 |
| elbow          | **1.4668**  | 幾何 +1.4368 + 自重たわみ 0.0300 |
| shoulder_roll / yaw, wrist_* | 0.0 | 元の ±0.25 / ±0.15 を 0 に |

---

## 行動

`JointPositionActionCfg`（29 関節）。目標関節角 = `default_joint_pos + scale * action`。

| 関節 | scale | 目標到達に必要な action |
|---|---:|---:|
| hip_pitch      | 0.8  | -2.5 |
| knee           | 0.8  | +2.4 |
| ankle_pitch    | 0.5  | -1.1 |
| その他 25 関節 | 0.25 | 0 付近を維持 |


---

## 観測

policy / critic 両方に同じ項。policy には Isaac Lab の標準ノイズ、Play では無効化。

| 項目 | 内容 |
|---|---|
| base_ang_vel        | 胴の角速度 |
| projected_gravity   | 重力の胴座標成分 |
| velocity_commands   | 常に 0（定位置保持）|
| joint_pos / joint_vel | 全 29 関節の相対値 |
| actions             | 前ステップの行動 |
| squat_phase         | `(sin 2π phi, cos 2π phi)` |


---

## 報酬構成

正報酬 最大 25.0 / ペナルティ 最小 -29.5。棒立ち 1 step 4.15 に対し完璧 27.97。

追従項（`pose_*` / `height_track` / `torso_pitch`）は `_relative_track` で
**「何もしない状態」を 0 点に正規化**してある。棒立ちでは 0 点。

### 姿勢追従

| 項目 | 関数 | weight | σ | 内容 |
|---|---|---:|---:|---|
| pose_coarse  | `squat_pose_tracking`   | 5.0 | 1.80 | 脚 3 関節群 + 側方群（粗）|
| pose_fine    | `squat_pose_tracking`   | 8.0 | 0.35 | 同（精度）|
| height_track | `squat_height_tracking` | 3.0 | 0.24 | 骨盤高さの位相追従 |
| torso_pitch  | `torso_pitch_tracking`  | 3.0 | 0.40 | 胴前傾。重心の前後位置を支配 |

### しゃがみ・胴姿勢のペナルティ

| 項目 | 関数 | weight | σ | 内容 |
|---|---|---:|---:|---|
| squat_shortfall_pen | `squat_depth_shortfall_penalty` | 3.0 | 0.90 | 深さ不足（depth ゲート）|
| waist_pitch_pen | `waist_pitch_penalty`    | 4.0 | 0.12 | 腰から上の反り |
| backlean_pen    | `torso_backlean_penalty` | 3.0 | 0.15 | 体全体を後ろに反らす |
| torso_roll_pen  | `torso_roll_penalty`     | 3.0 | 0.25 | 胴の左右傾き |

### 開脚・左右非対称の抑制（すべて片側・対称な姿勢は無罰）

| 項目 | 関数 | weight | σ | 内容 |
|---|---|---:|---:|---|
| hip_abduction_pen | `hip_abduction_tracking`      | 3.0 | 0.12 | 開脚（hip_roll）|
| knee_lateral_pen  | `lateral_offset_penalty`      | 3.0 | 0.06 | 両膝が同じ側に寄る |
| stance_pen        | `stance_width_penalty_phased` | 2.5 | 0.08 | 足幅の広がり |
| feet_lateral_pen  | `lateral_offset_penalty`      | 2.0 | 0.08 | 両足が同じ側に寄る |
| ankle_roll_pen    | `joint_default_deviation_penalty` | 2.0 | 0.20 | 足首の左右傾き |

### 腕（固定姿勢の維持）

| 項目 | 関数 | weight | σ | 内容 |
|---|---|---:|---:|---|
| arm_hold_pen   | `joint_default_deviation_penalty` | 4.0 | 0.35 | 腕 14 関節を初期姿勢に。margin 0.25 で自重たわみを無罰 |
| knee_clear_pen | `hands_knee_clearance_penalty`    | 5.0 | 0.08 | 手が膝にめり込む（保険）|

### 定位置保持（静止で 0）

| 項目 | 関数 | weight | σ | 内容 |
|---|---|---:|---:|---|
| drift_pen    | `drift_penalty`      | 3.0 | 0.60 | spawn からの水平距離 |
| heading_pen  | `heading_penalty`    | 2.0 | —    | 初期ヨー向きからのずれ |
| yaw_rate_pen | `yaw_rate_penalty`   | 2.0 | 0.50 | ヨー角速度（`ang_vel_xy_l2` は z を見ない）|
| slip_pen     | `feet_slip_penalty`  | 1.0 | 0.30 | 接地足の水平速度 |
| speed_pen    | `base_speed_penalty` | 0.5 | 0.40 | 胴の水平速度（z は無視）|

### 上体・接地（合計を正に保つ床）

| 項目 | 関数 | weight | 内容 |
|---|---|---:|---|
| upright  | `upright_bonus` | 3.0 | `(1 - g_z)/2`、立位 1.0 / 逆さま 0 |
| grounded | `feet_grounded` | 3.0 | 接地している足の割合 |

### 正則化

| 項目 | 関数 | weight | 内容 |
|---|---|---:|---|
| dof_pos_lim  | `joint_pos_limits` | -1.0    | 関節限界超過（異常検知に有用）|
| ang_vel_xy   | `ang_vel_xy_l2`    | -0.02   | 角速度 |
| action_rate  | `action_rate_l2`   | -0.015  | 行動変化率（振動抑制）|
| joint_acc    | `joint_acc_l2`     | -2.5e-7 | 関節加速度 |
| joint_torque | `joint_torques_l2` | -1.0e-6 | 関節トルク |

---

## 関節の役割


| 役割 | 関節 | 担当 |
|---|---|---|
| 参照姿勢を追う | hip_pitch / knee / ankle_pitch | `squat_pose_tracking` |
| 0 に固定 | hip_yaw / waist_yaw / waist_roll | `LATERAL_CFG` |
| デフォルト維持 | ankle_roll | `joint_default_deviation_penalty` |
| 固定（学習対象外）| 腕 14 関節 | `arm_hold_pen` |
| 専用項 | hip_roll（開脚）/ waist_pitch（反り）| `hip_abduction_tracking` / `waist_pitch_penalty` |

---

## 終了条件

| 名前 | 関数 | 設定 | 意図 |
|---|---|---|---|
| time_out  | `time_out`                  | `episode_length_s = 12.0` | 通常の終了（2 周期）|
| fell_over | `bad_orientation`           | limit_angle = 1.2 | 約 69 度傾いたら終了 |
| collapsed | `root_height_below_minimum` | 0.20 m | 骨盤沈み込みで終了 |

---

## PPO 設定 (`agents/rsl_rl_ppo_cfg.py`)

`G1PeriodicSquatPPORunnerCfg`:

- experiment_name : `squat_only`
- num_steps_per_env : 24 / max_iterations : 40000 / save_interval : 200
- empirical_normalization : True
- policy : hidden [512, 256, 128]、activation elu、**init_noise_std 0.35**
- algorithm : clip 0.2、entropy_coef 0.005、epochs 5、minibatches 4、
  lr 3e-4（adaptive、desired_kl 0.01）、gamma 0.99、lam 0.95

`init_noise_std` はアクションスケールと連動する。関節目標のノイズは
`scale × init_noise_std`（膝で 0.28 rad = 16 度）。

---

## 学習コマンド

```bash
conda activate env_isaaclab
cd ~/unitree_rl_lab
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-PeriodicSquat \
    --num_envs 4096 --headless
```

続きから:

```bash
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-PeriodicSquat \
    --num_envs 4096 --headless \
    agent.resume=true agent.load_run=<run名>
```

再生:

```bash
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-PeriodicSquat-Play --num_envs 16
```

---
