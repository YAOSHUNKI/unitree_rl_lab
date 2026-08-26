# G1-29dof sim2sim 手順ガイド(unitree_rl_lab)

Isaac Lab で学習した方策を MuJoCo で検証(sim2sim)し、最終的に実機(sim2real)へ持っていくまでの
全体像と具体手順を、**今回のセッションで実際にやったことを含めて**まとめます。

---

## 0. sim2sim とは / 全体の流れ

```
[Isaac Lab で学習]
      │  scripts/rsl_rl/train.py
      ▼
[方策を export]  ─ play.py が exported/policy.onnx(.pt) と params/deploy.yaml を生成
      │
      ├─────────────► ルートA: 公式(unitree_mujoco + g1_ctrl, DDS)
      │                         標準観測の方策(velocity等)向け・実機と同一経路
      │
      └─────────────► ルートB: Python単体MuJoCo(DDS不要)   ★今回成功
                                カスタム観測の方策(squat/pickup_carry)向け・軽量
      ▼
[実機 sim2real]  ─ g1_ctrl --network eth0
```

ポイントは **「学習した方策 = ニューラルネット」+「観測の作り方」+「行動の変換」** の3点を、
学習時とまったく同じに MuJoCo 側で再現することです。ここがズレると挙動が崩れます。

---

## 1. 共通:学習 → export

学習PC(Isaac Lab が動く環境)で行います。

```bash
# 学習
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-PeriodicSquat

# export(推論の再生 + ファイル書き出し)
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-PeriodicSquat --num_envs 16
```

`play.py` を実行すると、その run フォルダ `logs/rsl_rl/<experiment>/<日時>/` に次が生成されます。

| 生成物 | 中身 | sim2simでの役割 |
|---|---|---|
| `exported/policy.onnx` | 正規化を内包した推論用ネット | **方策本体**(生の観測を入れるだけ) |
| `exported/policy.pt` | 同上(TorchScript版) | 代替 |
| `params/deploy.yaml` | 観測構成・`joint_ids_map`・PDゲイン・default姿勢・action scale | **観測と行動の設計図** |

> ⚠️ 学習しただけの run には `model_*.pt` しか無く、`exported/` はまだありません。
> **sim2sim には必ず一度 `play.py` で export が必要**です(今回まさにここが最初の詰まり)。

### deploy.yaml が持つ重要情報
- `joint_ids_map`: **IsaacLab関節順 ↔ SDK(=MuJoCo)関節順** の対応表
- `stiffness` / `damping`: PDゲイン(SDK順)
- `default_joint_pos`: 基準姿勢(IsaacLab順)
- `actions.*.scale`: 行動スケール(0.25)。行動 = `default + scale × action`
- `observations`: 観測項目とその順序・スケール

観測次元の例:
- **velocity**: `(base_ang_vel3 + projected_gravity3 + velocity_commands3 + joint_pos29 + joint_vel29 + last_action29) × history5 = 480`
- **squat**: 上記(history1)96 + `box_rel3 + box_dist_heading3 + hand_pos6 + box_in_hands6 + hand_touch2 + squat_phase2 = 118`

---

## 2. ルートA:公式(unitree_mujoco + g1_ctrl / DDS)

実機 deploy と同じ C++ 経路。**標準観測だけの方策(velocity など)向け**。

### 2-1. 仕組み
```
unitree_mujoco (シミュレータ)  ──rt/lowstate──►  g1_ctrl (C++)
        ▲                       (IMU/関節/wireless_remote)   │
        └────────rt/lowcmd──────(関節指令 q/kp/kd)────────────┘
```
- ジョイスティックは `rt/lowstate` の `wireless_remote`(40バイト)に埋め込まれて届く
  (`g1_sub.h` の `joystick.extract(...)`)。**別トピックではない**ため、外部からの
  仮想コントローラ注入は効かない。
- g1_ctrl は **domain_id 固定=0**(`main.cpp` の `Init(0, ...)`)。

### 2-2. 必要な準備(今回やったこと)
1. **学習した velocity 方策を配置**(非破壊):
   `logs/.../unitree_g1_29dof_velocity/<日時>/exported` と `params/deploy.yaml` を
   `deploy/robots/g1_29dof/config/policy/velocity/v1/` にコピー。
   C++ はサブフォルダ名を昇順ソートし最後(=v1)を自動選択するので設定変更不要。
2. **キーボード操作対応を追加**(ゲームパッド無し対策・今回改修):
   - `deploy/include/FSM/FSMState.h`: config の `transitions_key` を読み、端末キーで状態遷移
   - `config.yaml`: Passive/FixStand/Velocity に `transitions_key`(1/2/3)を追加
   - `State_RLBase.cpp`: `keyboard_velocity_commands` を速度レンジにクランプ
   - v1 の `deploy.yaml`: 観測 `velocity_commands → keyboard_velocity_commands`
   - → **C++変更後は再ビルド必須**
3. ビルド:
   ```bash
   cd deploy/robots/g1_29dof && rm -rf build && mkdir build && cd build && cmake .. && make -j
   ```

### 2-3. 実行
```bash
# unitree_mujoco/simulate/config.yaml:  robot:"g1"  domain_id:0  interface:"lo"  use_joystick:0
cd unitree_mujoco/simulate/build && ./unitree_mujoco       # 先に起動
cd unitree_rl_lab/deploy/robots/g1_29dof/build && ./g1_ctrl -n lo   # ★domain/interface を揃える
```
操作(キーボード):端末で `2`(FixStand)→ MuJoCo窓で `8`(接地)→ 端末で `3`(Velocity)→
MuJoCo窓で `9`(バンド解除)→ `w/a/s/d/q/e` で歩行 → `1` で停止。

### 2-4. 今回ハマったところ(教訓)
| 症状 | 原因 | 対処 |
|---|---|---|
| joystick が無いと落ちる | `use_joystick:1` で物理パッド必須 | `use_joystick:0` + キーボード対応 |
| `waiting for lowstate` で停止 | g1_ctrl(domain0/lo)と unitree_mujoco の DDS 経路不一致 | `domain_id:0`・`interface:"lo"` を揃え、`./g1_ctrl -n lo` |
| squat 方策が動かせない | box/hand/squat_phase の C++ 観測が未実装 | ルートB を使う(下記) |

> **ルートAの限界**: 対応する観測は `observations.h` に実装済みのもの(base_ang_vel,
> projected_gravity, velocity_commands, joint_pos_rel, joint_vel_rel, last_action, gait_phase)
> のみ。**squat/pickup_carry のカスタム観測は C++ 未実装**なので、そのままでは動きません。

---

## 3. ルートB:Python単体MuJoCo(DDS不要)★スクワットはこちらで成功

`unitree_mujoco` も `g1_ctrl` も DDS もジョイスティックも使わず、`policy.onnx` を
MuJoCo(python)で直接回し、観測を Python で再現します。**カスタム観測の方策に最適**。

### 3-1. 配置(今回作成)
```
deploy/mujoco_py/
├── deploy_mujoco_squat.py     # ビューア本体(観測118次元を再現・PD制御・box配置)
├── g1_model/                  # unitree_mujoco の g1 モデルをコピー
│   ├── g1_29dof.xml
│   ├── scene_squat.xml        # robot+floor+box(自動生成)
│   └── meshes/
└── policies/squat/
    ├── policy.onnx            # 学習PCから持ってきた方策(export済み)
    └── deploy.yaml            # 観測118次元の設計図
```
> 学習PCから持ってくるのは **`policy.onnx` だけで十分**(deploy.yaml は同一タスクなら
> こちら側のものが使える。次元不一致ならスクリプトが起動時にエラーで知らせる)。

### 3-2. スクリプトがやっていること
1. `deploy.yaml` を読み、`joint_ids_map`/ゲイン/default姿勢/観測順を取得
2. MuJoCo(SDK順)⇄ IsaacLab順を相互変換
3. 毎ステップ観測118次元を Python で構築
   (IMUのgyro→base_ang_vel、base quat→projected_gravity、box/手の位置→box_rel等、
    経過時間→squat_phaseのsin/cos)
4. `policy.onnx` に生観測を入力(正規化は onnx 内包)→ 行動29
5. 行動 = `default + 0.25×action` を目標に **PD位置制御 50Hz**
   (sim 0.002s × decimation 10)で `mj_step`

### 3-3. 実行
```bash
pip install mujoco onnxruntime numpy pyyaml
cd ~/unitree_rl_lab/deploy/mujoco_py
python deploy_mujoco_squat.py
```
MuJoCoビューアが開き、約6秒周期で立ち↔しゃがみを再生。

### 3-4. 検証結果(今回クラウドで実行)
- 観測118次元が policy 入力[1,118]と一致・全項目サイズ正常
- 骨盤高さ 0.80→0.41→0.78 を約6秒周期で往復(スクワット成立)
- projected-gravity z ≈ -1.0 維持(直立=転倒せず)

---

## 4. どちらのルートを使うか

| 方策 | 観測 | 推奨ルート | 理由 |
|---|---|---|---|
| velocity(歩行) | 標準のみ | A または B | Aは実機同一経路。Bは手軽 |
| squat / pickup_carry | カスタム(box/hand/squat_phase) | **B(Python単体)** | AはC++観測未実装 |
| 実機へ出す最終確認 | ― | A | sim2real と同一経路で確認できる |

squat を最終的に実機へ出すなら、いずれ `observations.h` に box/hand/squat_phase 相当の
C++ 観測関数を追加してルートA化する必要があります(box はシーン追加も要)。
まず Python(ルートB)で方策の妥当性を確認 → 必要になったら C++化、が現実的です。

---

## 5. 別の方策・チェックポイントに差し替える

- **ルートB**: `policies/squat/` の `policy.onnx` と `deploy.yaml` を置き換えるだけ。
- **ルートA**: `config/policy/velocity/v2/` のように新しい番号のフォルダに
  `exported/` と `params/` を置けば、C++ が自動で最新を選択。

---

## 6. チェックリスト(sim2sim を最短で通す)

1. [ ] 学習PCで対象runを `play.py` して `exported/policy.onnx` と `params/deploy.yaml` を生成
2. [ ] 方策が標準観測か、カスタム観測かを確認(=ルートA/Bの判断)
3. [ ] **ルートB**: `policy.onnx`+`deploy.yaml` を `deploy/mujoco_py/policies/<name>/` に置く →
       `python deploy_mujoco_squat.py`
4. [ ] **ルートA**: v1配置 → (必要ならキーボード改修) → `make` → unitree_mujoco(domain0/lo) →
       `./g1_ctrl -n lo`
5. [ ] 挙動確認(倒れない・意図した動作)→ OKなら実機 sim2real へ
