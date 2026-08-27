# DDS不要 MuJoCo ランナー(G1-29dof / マルチタスク)

`unitree_mujoco` も `g1_ctrl` も DDS もジョイスティックも使わず、export した方策
(`policy.onnx`)を MuJoCo で直接動かして可視化します。**タスクごとにフォルダを分け、
同じ `run_mujoco.py` で切り替えて実行**できます。

## 構成
```
deploy/mujoco_py/
├── run_mujoco.py          # 汎用ランナー(--task で切替)
├── g1_model/              # ロボットモデル(全タスク共有)
│   ├── g1_29dof.xml
│   ├── scene_29dof.xml    # box 無しシーン
│   ├── scene_box.xml      # box 有りシーン(box/hand系の観測を使うタスク用)
│   └── meshes/
└── tasks/
    ├── squat/             # ← 立ち↔しゃがみ(検証済み)
    │   ├── policy.onnx
    │   ├── deploy.yaml
    │   └── task.yaml       # 任意設定
    ├── velocity/          # ← 歩行(例)
    │   ├── policy.onnx
    │   ├── deploy.yaml
    │   └── task.yaml
    └── <新タスク>/         # 追加はフォルダを作るだけ
```

## セットアップ
```bash
pip install mujoco onnxruntime numpy pyyaml
```

## 実行
```bash
cd ~/unitree_rl_lab/deploy/mujoco_py
python run_mujoco.py --task squat            # スクワット
python run_mujoco.py --task velocity --vx 0.5  # 歩行(前進0.5m/s)
python run_mujoco.py --task <task名>
```
MuJoCo ビューアが開きます(閉じる/Ctrl-C で終了)。

## 新しいタスクを追加する手順
1. 学習PCで対象タスクを `play.py` して export
   (`exported/policy.onnx` と `params/deploy.yaml` を生成)
2. このPCで:
   ```bash
   mkdir -p tasks/<task名>
   cp /path/exported/policy.onnx  tasks/<task名>/policy.onnx
   cp /path/params/deploy.yaml    tasks/<task名>/deploy.yaml
   ```
3. 実行:`python run_mujoco.py --task <task名>`

これだけです。**box を使う観測(box_rel など)が含まれていれば自動で box 有りシーンを選択**します。
観測次元は起動時に policy.onnx と照合し、一致しなければエラーで知らせます。

## task.yaml(任意・各タスクフォルダに置く)
すべて省略可。CLI 引数が優先されます。
```yaml
base_height: 0.80          # 初期の骨盤高さ[m]
period: 6.0                # 周期動作の周期[s](squatはdeploy.yamlが優先)
box_x: 0.7                 # box初期位置
box_y: 0.0
vx: 0.5                    # 速度指令(velocity系のみ)
vy: 0.0
wz: 0.0
scene: scene_box.xml       # シーンを明示指定したい場合
history_order: oldest_first # 履歴積み観測の順序(velocityは5フレーム積み)
```

## 対応している観測項目
`base_ang_vel, projected_gravity, velocity_commands / keyboard_velocity_commands,
joint_pos(_rel), joint_vel(_rel), actions / last_action, gait_phase / squat_phase,
box_rel, box_dist_heading, hand_pos, box_in_hands, hand_touch`
— history_length(フレーム積み)にも対応。

## 動作の仕組み(要点)
- `deploy.yaml` から観測を再現し、MuJoCo(SDK順)⇄IsaacLab順を `joint_ids_map` で変換
- 行動 = default姿勢 + scale(0.25)×action を PD位置制御(50Hz)
- 正規化は onnx 内包なので生観測を入力

## 検証済みの挙動(こちらで実行)
- **squat**: box有シーン自動選択・観測118次元・約6秒周期で立ち↔しゃがみ・直立維持 ✓
- **velocity**: box無シーン自動選択・観測480次元(5フレーム積み)を正しく構築 ✓
  ※同梱の velocity 方策は学習途中(model_9)のため歩行は安定しません。
    よく学習した velocity チェックポイントを export して差し替えてください。

## トラブルシュート
| 症状 | 対処 |
|---|---|
| `built observation = N but policy expects M` | deploy.yaml と policy.onnx の組が不一致。対応する deploy.yaml を置く |
| すぐ倒れる | 方策の学習不足の可能性。よく学習したcheckpointをexportし直す。`--base-height` 微調整 |
| box系タスクで box が無い | `g1_model/scene_box.xml` があるか確認(box系観測で自動選択) |
| history積み方策が変 | `--history-order newest_first` を試す |
| 画面が出ない | GUI環境で実行(SSHはX転送/VNC) |
