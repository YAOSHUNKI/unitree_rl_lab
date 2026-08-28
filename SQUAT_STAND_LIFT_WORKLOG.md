# SquatStandLift タスク 作業ログ

Unitree G1 29DOF に「腰を下ろした状態から箱を掴んで立ち上がる」動作を、
`PeriodicSquat` とは独立したポリシーとして学習させる。

- タスク ID: `Unitree-G1-29dof-SquatStandLift`
- タスクフォルダ: `source/unitree_rl_lab/unitree_rl_lab/tasks/squat_stand_lift/`
- 学習ログ: `logs/rsl_rl/g1_squat_stand_lift/`
- 基盤: `G1PickupCarryEnvCfg` を継承 (箱・センサ・観測を流用)
- **フォルダ方針: 案 A** — 新タスクだけ独立、mdp/agents は `squat_only` を参照

---

## 現在の状態 (2026-08-28)

| # | 内容 | 状態 |
|---|---|---|
| 0 | `tasks/squat_stand_lift/` フォルダを scaffold | **DONE** |
| 1 | `squat_only/mdp/rewards.py` の `_squat_phase` / `_squat_depth` / 全 8 報酬関数に `phase_offset` 追加 | **DONE** |
| 2 | `squat_only/mdp/observations.py` の `squat_phase_obs` に `phase_offset` 追加 | **DONE** |
| 3 | SQUAT 底の腕関節角 (shoulder_pitch, elbow) を実測 | TODO (暫定値で先行) |
| 4 | `squat_stand_lift_env_cfg.py` を新規作成 | **DONE (暫定 spawn 値)** |
| 5 | `tasks/squat_stand_lift/robots/g1/29dof/__init__.py` に gym.register 追加 | **DONE** |
| 6 | 上位パッケージから import_packages で自動発見される (確認) | **DONE (仕組み確認済)** |
| 7 | `list_envs.py` で新タスクが列挙されるか確認 | TODO |
| 8 | `play.py` で spawn 姿勢と箱位置の視認 | TODO |
| 9 | 学習開始 (500 iter で `Loss/learning_rate` を確認) | TODO |
| 10 | Sim2Sim (Mujoco) で検証 | TODO |

---

## 実際に作ったフォルダ構成

```
tasks/
├── squat_only/                       ← 既存 + mdp に phase_offset 追加のみ
│   ├── mdp/
│   │   ├── rewards.py                  ← _squat_phase / _squat_depth / 8 関数に phase_offset
│   │   ├── rewards.py.pre_phaseoffset  ← 変更前バックアップ
│   │   ├── observations.py             ← squat_phase_obs に phase_offset
│   │   └── observations.py.pre_phaseoffset
│   ├── agents/rsl_rl_ppo_cfg.py
│   └── robots/g1/29dof/
│       ├── pickup_carry_env_cfg.py
│       └── squat_only_env_cfg.py       (PeriodicSquat タスク; 破壊的変更なし)
└── squat_stand_lift/                   ← 新規 (scaffold は元々複製されていたが再構築)
    ├── __init__.py                     ← タスク説明の docstring のみ
    ├── mdp/                            ← re-export シム
    │   ├── __init__.py                 ← from unitree_rl_lab.tasks.squat_only.mdp import *
    │   ├── events.py                   ← placeholder (import されない)
    │   ├── observations.py             ← placeholder
    │   └── rewards.py                  ← placeholder
    ├── agents/
    │   ├── __init__.py                 ← 空
    │   └── rsl_rl_ppo_cfg.py           ← squat_only の G1PickupCarryPPORunnerCfg を re-export
    ├── robots/
    │   ├── __init__.py, g1/__init__.py ← 空 (パッケージ化)
    │   └── g1/29dof/
    │       ├── __init__.py             ← gym.register(SquatStandLift, -Play)
    │       └── squat_stand_lift_env_cfg.py    ← G1SquatStandLiftEnvCfg 本体
    └── _to_delete/                     ← 元 scaffold の duplicate を退避
        ├── pickup_carry_env_cfg.py
        └── squat_only_env_cfg.py
```

**注**: `_to_delete/` は `__init__.py` を置いていないので `import_packages`
の自動発見に引っかからない。手動で削除して OK。

---

## 全体方針 (再掲)

`PeriodicSquat` の姿勢追従群を **半周期だけ** 使ってしゃがみ→立ちの単調追従に転用し、
`squat_only` にある箱把持・持ち上げ報酬を後半フェーズにゲート付きで足す。
周期関数は破壊せず、位相オフセット 1 引数だけ追加する。

### 設計原則 (`g1-squat-reward-reference.md` より)

1. 正報酬 = タスク達成のみ。定位置保持はペナルティで。
2. ペナルティは有界 `exp(-x^2/sigma^2) - 1 in [-1, 0]`。
3. 転倒は罰ではなく終了で扱う。
4. 参照姿勢は静的に安定でなければならない。

---

## 実装済みの変更点

### 修正 1 : `squat_only/mdp/rewards.py`

- `_squat_phase(env, period, phase_offset: float = 0.0)`
  返り値: `((env.episode_length_buf * env.step_dt) % period) / period + phase_offset) % 1.0`
- `_squat_depth(env, period, phase_offset: float = 0.0)` phase_offset をパススルー
- 以下 8 個の報酬関数に `phase_offset: float = 0.0` 引数を追加、内部の
  `_squat_depth(env, period)` 呼び出しを `_squat_depth(env, period, phase_offset)` に:
  - `squat_pose_tracking`
  - `squat_height_tracking`
  - `torso_pitch_tracking`
  - `hip_abduction_tracking`
  - `stance_width_penalty_phased`
  - `arm_forward_direction`
  - `arm_extension_penalty`
  - `arm_forward_shortfall_penalty`

デフォルト 0.0 なので `PeriodicSquat` の既存呼び出しは無変更で動く (**後方互換**)。

### 修正 2 : `squat_only/mdp/observations.py`

- `squat_phase_obs(env, period=3.0, phase_offset: float = 0.0)`

### 修正 3 : `squat_stand_lift_env_cfg.py` (新規、暫定 spawn 値)

主要定数:
```
T_TASK       = 3.0       # しゃがみ→立ちにかける時間 [s]
SQUAT_PERIOD = 6.0       # 2 * T_TASK
PHASE_OFFSET = 0.5       # phi(t=0)=0.5 → depth=1 から開始

# spawn = 完全しゃがみ
hip_pitch = -2.10, knee = 2.20, ankle = -0.75
pelvis_z  = 0.39
shoulder_pitch = 0.30  ← 暫定 (TODO: 実測)
elbow          = 1.00  ← 暫定
```

報酬構成:
- 姿勢追従群 (`pose_coarse`, `pose_fine`, `height_track`, `torso_pitch`,
  `hip_abduction_pen`, `stance_pen`): `phase_offset=0.5` で PeriodicSquat から流用
- 箱系 (`hands_near`, `hands_touch`, `grasp`, `lift`, `stand_up`, `drop_pen`):
  squat_only から流用 (現状ゲートなし; 挙動を見て後で追加)
- 定位置保持 (`drift_pen`, `slip_pen`, `heading_pen`, `speed_pen`)
- 姿勢ペナルティ (`waist_pitch_pen`, `torso_roll_pen`, `wrist_pen`)
- 正則化 (`ang_vel_xy`, `action_rate`, `joint_acc`, `joint_torque`, `dof_pos_lim`)
- 削除: `arm_forward_*`, `hands_width`, `arm_ext_pen`, `arm_shortfall_pen`, `knee_clear_pen`

`__post_init__`:
- 移動コマンドを 0 に固定 (rel_standing_envs=1.0)
- `episode_length_s = T_TASK` で戻り位相に入らせない
- 位相観測を追加 (`period=6.0, phase_offset=0.5`)
- 初期姿勢 (脚 + 骨盤高) を SQUAT 底に、腕を暫定値に上書き
- `reset_box` の x_range を (0.30, 0.38) に変更 (しゃがみでの手の届く範囲)
- 終了: `base_contact=None`, `fell_over` (limit_angle=1.2), `collapsed` (0.20 m)

### 修正 4 : `squat_stand_lift/robots/g1/29dof/__init__.py`

`Unitree-G1-29dof-SquatStandLift` と `-Play` を gym.register。
env_cfg entry_point は
`unitree_rl_lab.tasks.squat_stand_lift.robots.g1.29dof.squat_stand_lift_env_cfg:G1SquatStandLiftEnvCfg`。
`rsl_rl_cfg_entry_point` は `squat_only` の `G1PickupCarryPPORunnerCfg` を流用。

### 修正 5 : import chain 対策 (importlib)

フォルダ名 `29dof` は先頭数字で Python 識別子として無効なので、
`squat_stand_lift_env_cfg.py` 側では `importlib.import_module` を使って
`G1PickupCarryEnvCfg` を読み込む。これは既存 gym.register の entry_point
文字列でも同じフォルダを参照しているので実績のあるパターン。

---

## 次にやること

### 優先 1: `list_envs.py` で列挙確認

```bash
conda activate env_isaaclab
cd ~/unitree_rl_lab
python scripts/list_envs.py 2>&1 | grep SquatStandLift
```

`Unitree-G1-29dof-SquatStandLift` と `Unitree-G1-29dof-SquatStandLift-Play`
の 2 つが出れば `gym.register` は成功。出ない場合は import chain の失敗が
考えられるので、Python トレースバックを追う。

### 優先 2: `play.py` で spawn 姿勢の視認

```bash
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-SquatStandLift-Play \
    --num_envs 16
```

学習前 (ランダム policy) でも spawn 状態は見れる。以下を確認:
- しゃがみ姿勢で spawn されるか (棒立ちで始まるなら init_state.pos の反映失敗)
- 箱が手の前 (34 cm) にあるか
- 腕の暫定値 (shoulder_pitch=0.30, elbow=1.00) で腕が箱に近い位置に来ているか

### 優先 3: 腕関節角の実測 (タスク #3)

`PeriodicSquat-Play` を回し、phi=0.5 (完全しゃがみ) のときの
shoulder_pitch と elbow の関節角を print。値を得たら
`squat_stand_lift_env_cfg.py` の `SQUAT_SHOULDER_PITCH`, `SQUAT_ELBOW`
を書き換える。

### 優先 4: 学習開始

```bash
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-SquatStandLift \
    --num_envs 64 --max_iterations 3000
```

500 iter 時点で **`Loss/learning_rate` が 1e-4 以上** を確認。下限
(1e-5) に張り付いていたら即停止して報酬を見直す (落とし穴 8)。

---

## 学習時に見る指標

| 指標 | 健全 | 異常時の意味 |
|---|---|---|
| `Loss/learning_rate` | 1e-4 以上 | 下限張り付き → 即停止 (落とし穴 8) |
| `Loss/entropy` | 単調減少 | 横ばい → 何も学習していない |
| `Episode_Reward/lift` | 上昇 | 上がらない → 掴めていない |
| `Episode_Reward/stand_up` | `lift` と連動 | 独立に上がる → 箱を離して立ってる |
| `Episode_Termination/fell_over` | 減少 | 4 割超 → 参照姿勢か spawn が不安定 |
| `Episode_Termination/collapsed` | 0 に近い | 発火 → spawn 直後の沈み込み。閾値 0.15 に |

---

## 履歴

- 2026-08-28 初版作成。修正 1-6 の設計を確定。腕関節角の実測がブロッキング。
- 2026-08-28 フォルダ構成を案 A に確定。`tasks/squat_stand_lift/` を新規、
  mdp/agents は `squat_only` を参照する形に。タスク一覧を再編。
- 2026-08-28 **実装完了**: mdp の phase_offset 追加、squat_stand_lift フォルダの
  scaffold 再構築 (元は squat_only の duplicate だったので shim 化)、
  env_cfg / gym.register 作成。腕 spawn は暫定値。全 Python ファイルの
  構文チェック通過。次は `list_envs.py` での列挙確認。
