# G1-29dof velocity ポリシー sim2sim 実行手順 (unitree_mujoco + g1_ctrl)

学習した velocity ポリシーを **公式ルート(unitree_mujoco + C++ `g1_ctrl`)** で MuJoCo 上で動かすための手順です。
**ゲームパッド無し(キーボード操作)** に対応済みです。

---

## 0. このセットアップで済ませたこと

### (a) 学習ポリシーの配置
学習run `logs/rsl_rl/unitree_g1_29dof_velocity/2026-08-20_14-04-44/` の
`exported/`(policy.onnx / policy.pt)と `params/deploy.yaml` を、
デプロイ側に **非破壊で** 配置しました:

```
deploy/robots/g1_29dof/config/policy/velocity/
├── v0/            ← リポジトリ同梱の参照ポリシー(そのまま保持)
└── v1/            ← ★今回学習したポリシー(新規配置・自動選択される)
    ├── exported/policy.onnx   (学習runとmd5一致)
    ├── exported/policy.pt
    └── params/deploy.yaml
```

`config.yaml` の `Velocity.policy_dir: config/policy/velocity` に対し、C++ が
サブフォルダを名前順で最後(= v1)を自動選択するため、**設定変更なしで v1 が使われます**。
(起動ログ `Policy directory: .../velocity/v1` で確認可能)

### (b) キーボード操作対応(ゲームパッド無し用)★今回の変更
`g1_ctrl` の FSM 状態遷移は本来 **無線コントローラ(ジョイスティック)入力のみ**で評価されます。
ジョイスティックは `rt/lowstate` の `wireless_remote` に埋め込まれて届くため、
外部から DDS で仮想コントローラを流しても g1_ctrl は受け取りません。
そこで **g1_ctrl 側にキーボード遷移を追加**しました。変更ファイルは以下4つ:

| ファイル | 変更内容 |
|---|---|
| `deploy/include/FSM/FSMState.h` | config の `transitions_key` を読み、端末キーで状態遷移を発火(ジョイスティックと併用可) |
| `deploy/robots/g1_29dof/src/State_RLBase.cpp` | `keyboard_velocity_commands` を deploy.yaml の速度レンジにクランプするよう改善 |
| `deploy/robots/g1_29dof/config/config.yaml` | Passive/FixStand/Velocity に `transitions_key`(1/2/3)を追加 |
| `.../config/policy/velocity/v1/params/deploy.yaml` | 観測 `velocity_commands` → `keyboard_velocity_commands` に変更 |

> ⚠️ **C++を変更したので再ビルドが必要です**(§2)。

---

## 1. 動作の整合性(確認済み)

| 項目 | 値 |
|---|---|
| 観測次元 | 480 (= 96 × history 5) |
| 観測項目 | base_ang_vel, projected_gravity, keyboard_velocity_commands, joint_pos_rel, joint_vel_rel, last_action |
| 行動次元 | 29 (関節位置, scale 0.25, offset = default_joint_pos) |
| 制御周期 | 50 Hz (step_dt 0.02) |
| ONNX I/O | input `obs`[1,480] → output `actions`[1,29] |

---

## 2. 再ビルド(必須)

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
cmake .. && make -j
# うまくいかない時はクリーンビルド:
# cd unitree_rl_lab/deploy/robots/g1_29dof && rm -rf build && mkdir build && cd build && cmake .. && make -j
```

- `unitree_sdk2` はインストール済みが前提。
- onnxruntime は RUNPATH で解決されます。万一 `libonnxruntime.so.1 not found` が出たら:
  ```bash
  export LD_LIBRARY_PATH=$PWD/../../../thirdparty/onnxruntime-linux-x64-1.22.0/lib:$LD_LIBRARY_PATH
  ```

---

## 3. unitree_mujoco 側の設定(クラッシュ対策)

`unitree_mujoco/simulate/config.yaml`:

```yaml
robot: "g1"
domain_id: 0
enable_elastic_hand: 1
use_joystick: 0     # ★ 0 にする(ゲームパッドが無いと 1 では起動時に強制終了する)
```

> `use_joystick: 0` にしても lowstate は publish され続けるので g1_ctrl は動作します。
> ジョイスティック値はニュートラル(0)になり、代わりにキーボードで操作します。

---

## 4. 実行手順(キーボード操作)

### 4-1. シミュレータ起動
```bash
cd unitree_mujoco/simulate/build
./unitree_mujoco
```

### 4-2. コントローラ起動(別ターミナル)
```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
```
> キーボード入力は **この g1_ctrl のターミナルにフォーカスがある時**に効きます。
> エラスティックバンド(7/8/9)は **MuJoCo ウィンドウにフォーカス**がある時に効きます。
> 2つのウィンドウを切り替えながら操作します。

### 4-3. 操作シーケンス

| 手順 | 操作 | フォーカス |
|---|---|---|
| 1. 直立へ | キー **`2`**(FixStand) | g1_ctrl 端末 |
| 2. 足を接地 | キー **`8`**(バンドを下げる) | MuJoCo 窓 |
| 3. ポリシー開始 | キー **`3`**(Velocity) | g1_ctrl 端末 |
| 4. バンド解除 | キー **`9`** | MuJoCo 窓 |
| 5. 歩行操作 | 下表(w/a/s/d/q/e) | g1_ctrl 端末 |
| 停止 | キー **`1`**(Passive) | g1_ctrl 端末 |

### 4-4. 速度指令(キーボード)

| キー | 指令 | 値(deploy.yaml のレンジ) |
|---|---|---|
| `w` / `s` | 前進 / 後退 | +1.0 / -0.5 m/s |
| `a` / `d` | 左 / 右 | +0.3 / -0.3 m/s |
| `q` / `e` | 左旋回 / 右旋回 | +0.2 / -0.2 rad/s |
| (無入力) | 停止 | 0 |

> キーは押している間だけ有効(端末のキーリピートに依存)。連続前進はキーを押し続けます。

---

## 5. ゲームパッドを使う場合(参考)

物理ゲームパッドを接続して `unitree_mujoco` の `use_joystick: 1` に戻せば、
従来どおりジョイスティックでも操作できます(キーボード遷移と併用可)。その場合は
v1 の deploy.yaml の観測名を `keyboard_velocity_commands` → `velocity_commands` に戻すと
スティックで連続的な速度指令が使えます。ボタン操作:
`L2+Up`→FixStand、`R1+X`→Velocity、`L2+B`→Passive。

---

## 6. トラブルシュート

| 症状 | 対処 |
|---|---|
| unitree_mujoco が「joystick が無い」で落ちる | `use_joystick: 0` にする(§3) |
| キー `2/3/1` を押しても遷移しない | g1_ctrl のターミナルにフォーカスがあるか確認。再ビルド済みか確認(§2) |
| `The other process is using the lowcmd channel` | 別の g1_ctrl / 実機制御が動作中。停止して再実行 |
| `Unmatched robot type` | unitree_mujoco の robot が g1(29dof)か確認 |
| すぐ倒れる | 手順順序(2→8→3→9)を確認。接地(8)前にポリシー開始しない |
| `libonnxruntime.so.1 not found` | §2 の `LD_LIBRARY_PATH` |
| 別ポリシーに変えたい | `config/policy/velocity/v2/` に置けば自動で最新が選ばれる |

---

## 参考:squat / squat_only を sim2sim する場合
velocity と違い観測に box / hand / squat_phase 等のカスタム項(計118次元)を含みます。
公式 C++ ルートでは (1) `play.py` で export、(2) `observations.h` に対応する C++ 観測関数を追加、
(3) MuJoCo シーンに box を追加、が必要です。まず velocity で経路を確立してから着手するのがおすすめです。
