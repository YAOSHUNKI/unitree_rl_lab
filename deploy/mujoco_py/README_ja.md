# DDS不要 MuJoCo ビューア (G1-29dof / スクワット方策)

`unitree_mujoco` も `g1_ctrl` も DDS もジョイスティックも使わずに、
export した方策(`policy.onnx`)を MuJoCo で直接動かして可視化します。

## 構成
```
deploy/mujoco_py/
├── deploy_mujoco_squat.py     # 本体(このスクリプト)
├── g1_model/                  # unitree_mujoco の g1 モデルをコピーしたもの
│   ├── g1_29dof.xml
│   ├── scene_squat.xml        # ★robot+floor+box のシーン(自動生成済み)
│   └── meshes/
└── policies/
    └── squat/
        ├── policy.onnx        # ★別PCから持ってきた学習済み方策
        └── deploy.yaml        # 観測118次元・joint_ids_map・PDゲイン・default姿勢
```

## セットアップ
```bash
pip install mujoco onnxruntime numpy pyyaml
```

## 実行
```bash
cd ~/unitree_rl_lab/deploy/mujoco_py
python deploy_mujoco_squat.py \
    --policy policies/squat/policy.onnx \
    --deploy policies/squat/deploy.yaml \
    --scene  g1_model/scene_squat.xml
```
MuJoCo ビューアが開き、**約6秒周期で立ち↔しゃがみ**を繰り返します。
ウィンドウを閉じる(または Ctrl-C)で終了。

## 動作の仕組み(要点)
- `deploy.yaml` から観測118次元を Python で再現
  (base_ang_vel / projected_gravity / velocity_commands / joint_pos / joint_vel /
   actions / box_rel / box_dist_heading / hand_pos / box_in_hands / hand_touch / squat_phase)
- 関節順は MuJoCo(SDK順) ↔ IsaacLab順 を `joint_ids_map` で相互変換
- 行動 = default姿勢 + 0.25×action を PD位置制御(50Hz, gains は deploy.yaml)
- `squat_phase` は経過時間から sin/cos を生成(period=6.0s)
- 正規化は onnx に内包済みなので生の観測をそのまま入力

## よく使うオプション
| オプション | 意味 | 既定 |
|---|---|---|
| `--period` | スクワット周期[s](deploy.yamlのsquat_phaseで上書き) | 6.0 |
| `--base-height` | 初期の骨盤高さ[m] | 0.80 |
| `--box-x`, `--box-y` | boxの初期位置 | 0.7, 0.0 |
| `--vx --vy --wz` | 速度指令(velocity方策を読ませた場合のみ有効) | 0 |

## 別の方策に差し替える
`policies/squat/` の `policy.onnx` と `deploy.yaml` を差し替えるだけです
(観測次元がスクリプトの想定と違う場合は起動時にエラーで知らせます)。
velocity方策(`--vx`等で操作)も同じスクリプトで動きます。

## トラブルシュート
| 症状 | 対処 |
|---|---|
| `observation size N != policy input M` | deploy.yaml と policy.onnx の組が不一致。対応する deploy.yaml を置く |
| mesh が見つからない | `g1_model/` に `meshes/` ごとコピーされているか確認 |
| すぐ倒れる/暴れる | `--base-height` を調整(0.78〜0.80)。deploy.yaml のゲインが学習と一致しているか確認 |
| 画面が出ない(ヘッドレス) | GUI環境で実行。SSHなら X 転送 or VNC が必要 |
