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
│   ├── scene_box.xml      # box 有りシーン
│   └── meshes/
└── tasks/
    ├── squat/             # ← 立ち↔しゃがみ
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

## 新しいタスクを追加する手順
1. 対象タスクを `play.py` して export
   (`exported/policy.onnx` と `params/deploy.yaml` を生成)
2. 
   ```bash
   mkdir -p tasks/<task名>
   cp /path/exported/policy.onnx  tasks/<task名>/policy.onnx
   cp /path/params/deploy.yaml    tasks/<task名>/deploy.yaml
   ```
3. 実行:`python run_mujoco.py --task <task名>`


## task.yaml(任意・各タスクフォルダに置く)
すべて省略可。
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

