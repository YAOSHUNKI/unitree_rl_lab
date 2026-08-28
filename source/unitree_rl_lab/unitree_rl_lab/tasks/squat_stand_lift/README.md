# squat_only — 報酬関数リファレンス

Unitree G1 29DOF に「その場で周期的にスクワットしながら両手を前へ伸ばす」動作を
学習させるための報酬関数群。

- 実装: `mdp/rewards.py`
- 設定: `robots/g1/29dof/squat_only_env_cfg.py`
- タスク ID: `Unitree-G1-29dof-PeriodicSquat`
- 周期: 6.0 秒 / 正報酬 最大 26.0 / ペナルティ 最小 -20.5

---

## 設計原則

失敗から得た 4 つ。個々の関数より優先度が高い。新しい項を足すときは必ず照合する。

### 1. 正報酬はタスク達成のみ。「何もしない」で稼げる項を作らない

静止・直立・接地といった「動かなければ満点」の項を正報酬に置くと、棒立ちが
高得点になり学習が止まる。定位置保持や姿勢維持は**ペナルティ側**に置き、
静止で 0・崩れた分だけマイナスとする。

> 実測: 棒立ちの得点が 12.15 → 4.81 に低下し、スクワットとの比が 1.31 → 2.05 倍に。

### 2. ペナルティは必ず有界にする

1 ステップの合計が負に振れると、エージェントは**早期終了で return を最大化**
しようとし、わざと転倒する。全ペナルティを `exp(-x²/σ²) - 1` の形にして
値域を `[-1, 0]` に収める。

### 3. 転倒は罰ではなく終了条件で扱う

転倒に大きなペナルティを与えると原則 2 に抵触する。`bad_orientation` と
`root_height_below_minimum` でエピソードを打ち切れば、残りステップの正報酬を
失うことが自然な罰になる。

### 4. 参照姿勢は静的に安定でなければならない

目標姿勢の重心が支持基底から外れていると、「追従すること」と「転ばないこと」が
両立せず学習が停止する。報酬をどう調整しても解けない。

> **G1 固有の制約**: 股関節が足首の真上に来る条件は `knee = 2 × |ankle|`。
> 足首背屈の soft 限界が 0.803 rad なので、踵接地のまま股関節を足の上に
> 保てるのは `knee <= 1.61` まで。それより深い姿勢は前傾で重心を戻す必要がある。

---

## 位相と深さ

周期系の全報酬がこの 2 つの内部関数を共有する。

```
phi   = (episode_length_buf * step_dt mod period) / period
depth = 0.5 - 0.5 * cos(2*pi*phi)          # 0=立ち, 1=しゃがみ切り

target = stand_value + (squat_value - stand_value) * depth
```

位相は observation にも `squat_phase_obs` として渡す (sin/cos の 2 次元)。
これが無いと方策は「今しゃがむ番か」を知れない。

---

## 現行の配点

### 正報酬 — タスク達成 (最大 26.0)

| cfg 名 | 関数 | weight | 値域 | 役割 |
|---|---|---:|---|---|
| `pose_coarse` | `squat_pose_tracking` | 4.0 | [0,1] σ=0.85 | 脚の参照姿勢へ粗く誘導 |
| `pose_fine` | `squat_pose_tracking` | 8.0 | [0,1] σ=0.35 | 同じ姿勢を高精度で要求 |
| `height_track` | `squat_height_tracking` | 3.0 | [0,1] | 骨盤高さの追従 |
| `torso_pitch` | `torso_pitch_tracking` | 3.0 | [0,1] | 胴の前傾 = 重心の前後位置 |
| `arm_forward` | `arm_forward_direction` | 5.0 | [0,1] | 上腕を前方へ振る |
| `hands_width` | `hands_width_match` | 2.0 | [0,1] | 両手の間隔を膝幅に |
| `upright` | `upright_bonus` | 0.5 | [0,1] | 合計を正に保つ床 |
| `grounded` | `feet_grounded` | 0.5 | [0,1] | 同上 |

### ペナルティ — 崩れた分だけマイナス (最小 -20.5)

| cfg 名 | 関数 | weight | 値域 | 役割 |
|---|---|---:|---|---|
| `hip_abduction_pen` | `hip_abduction_tracking` | 6.0 | [-1,0] 片側 | 開脚 (hip_roll) |
| `stance_pen` | `stance_width_penalty_phased` | 5.0 | [-1,0] 片側 | 足幅の広がり |
| `arm_ext_pen` | `arm_extension_penalty` | 3.0 | [-1,0] 片側 | 肘の曲がり |
| `drift_pen` | `drift_penalty` | 1.5 | [-1,0] | 水平ドリフト |
| `slip_pen` | `feet_slip_penalty` | 1.5 | [-1,0] | 足の滑り |
| `hands_sym_pen` | `hands_symmetry_penalty` | 1.0 | [-1,0] | 手の左右非対称 |
| `torso_roll_pen` | `torso_roll_penalty` | 1.0 | [-1,0] | 胴の左右傾き |
| `heading_pen` | `heading_penalty` | 1.0 | [-1,0] | ヨー方向のずれ |
| `speed_pen` | `base_speed_penalty` | 0.5 | [-1,0] | 水平速度 |

### 正則化

| cfg 名 | 関数 | weight | 備考 |
|---|---|---:|---|
| `dof_pos_lim` | `joint_pos_limits` | -1.0 | 参照姿勢が関節限界に食い込むと発火。異常検知に有用 |
| `wrist_default` | `joint_deviation_l1` | -0.30 | 手首のみ。肩は前へ伸ばすので対象外 |
| `ang_vel_xy` | `ang_vel_xy_l2` | -0.02 | |
| `action_rate` | `action_rate_l2` | -0.005 | |
| `joint_torque` | `joint_torques_l2` | -1.0e-6 | |
| `joint_acc` | `joint_acc_l2` | -2.5e-7 | |

---

## 関数リファレンス

### 姿勢追従

#### `squat_pose_tracking` -> [0,1]

hip_pitch / knee / ankle_pitch の 3 グループを位相に応じた目標へ追従させ、
lateral グループ (hip_yaw, waist_yaw, waist_roll) を 0 に固定する。

```
err = sum( mean((q - target)^2) ) over {hip_pitch, knee, ankle}
    + mean(q_lateral^2)
reward = exp(-err / std^2)
```

**1 項にまとめる理由**: 関節ごとに別々の報酬を置くと互いに矛盾する解が生まれる。
1 本の参照姿勢なら「膝を曲げる」「腰を落とす」「脚をひねらない」が同時に
成立する組み合わせしか高得点にならない。

**σ を変えて 2 回登録している** (coarse 0.85 / fine 0.35)。目標が深くなると
追従失敗時の誤差が大きくなり、単一の狭い σ では底で勾配が消える。粗い方が
遠方からの誘導を、細かい方が精度を担当する。棒立ちとの識別比が 2.5 → 3.5 倍に改善。

#### `squat_height_tracking` -> [0,1]

骨盤の高さを位相の目標に追従させる。関節角だけ合っていて体が浮く・傾く解を潰す。

#### `torso_pitch_tracking` -> [0,1]

`projected_gravity_b[:, 0] = sin(前傾角)` を目標に追従させる。深いスクワットでは
前傾しないと重心が踵より後ろに抜けて後方転倒する。

脚の関節追従でも前傾は間接的に決まるが、あれは個々の関節誤差の和なので前傾角
そのものがずれても部分点が入る。重心の前後位置を直接支配する量なので独立させている。

### 腕

腕の姿勢は「向き・伸展・間隔」の 3 つで完全に決まる。いずれもスケールフリーなので、
腕の長さや肩の高さを知る必要がない。手の絶対位置を別途指定すると腕長の推定値に
依存し、互いに引っ張り合う。

#### `arm_forward_direction` -> [0,1]

**上腕** (肩 -> 肘) の単位ベクトルの前方 x 成分を位相の目標に追従させる。
0 = 真下、0.55 = 鉛直から 33 度前、1.0 = 水平前方。

当初は「肩 -> 手」で測っていたが、それだと**肘を曲げるだけで前方成分を稼げてしまい**、
肩を回さずに満点が取れた。上腕で測れば肩関節を回す以外に達成手段がない。

位置目標ではなく向きにしている理由: 向きは必ず到達可能で、全域で単調な勾配が出る。
位置目標は腕の長さの外にあると勾配が薄くなり、腕を垂らしたままでも部分点が入る。

#### `arm_extension_penalty` -> [-1,0]

肩・肘・手の 3 点の幾何で腕の伸び具合を測る。深さでゲートしているので立ち位相では 0。

```
straightness = ||肩->手|| / (||肩->肘|| + ||肘->手||)

3点が一直線 -> 1.0    肘の屈曲角 f に対し厳密に cos(f/2)
屈曲 28度   -> 0.970  (min_straightness の既定値。ここまで無罰)
屈曲 56度   -> 0.885  (G1 のデフォルト姿勢)
屈曲 86度   -> 0.732  (以降 -1 に飽和)
```

**関節角で判定しない理由**: G1 の elbow は可動域が -1.05〜2.09 と非対称で、
0 が「真っ直ぐ」かどうか USD を見ないと判断できない。幾何ならリンク長も
符号規約も知らずに済む。

#### `hands_width_match` -> [0,1]

両手の左右間隔を、実測した膝の間隔 (× width_scale、下限 min_width) に一致させる。

これが無いと両手が中央で重なる。`hands_symmetry_penalty` は「左右対称」しか
要求しないので、**両手が y=0 で重なっていても満点**になってしまう。

#### `hands_symmetry_penalty` -> [-1,0]

骨盤ヨー座標系で、前後 x は左右一致・左右 y は符号反転 (和が 0)・上下 z は左右一致。
すべて差の二乗で見るので `find_bodies` が返す左右の並び順に依存しない。

### 開脚抑制

深いスクワットは narrow stance では踏ん張りにくいため、開脚が「安い抜け道」になる。
両方とも**片側ペナルティ** (閉じている分は罰しない) なので、重みを上げても
正しい姿勢のコストは 0。

#### `hip_abduction_tracking` -> [-1,0]

`|hip_roll|` の平均が「深さ相応の許容量」を**超えた分だけ**罰する。
立ち位相では 0、完全しゃがみでは 0.18 rad (約 10 度) まで許容。

完全に 0 を要求するのは非現実的 — 大腿が水平近くまで来る深さでは、人間でも
胴を入れるスペースのために脚をやや開く。

#### `stance_width_penalty_phased` -> [-1,0]

足の左右間隔が深さ相応の許容幅 (0.20 -> 0.28 m) を超えた分だけ罰する。

### 定位置保持

すべてペナルティ形式。静止していれば 0 なので、原則 1 の「タダ取り」が発生しない。

| 関数 | 内容 |
|---|---|
| `drift_penalty` | スポーン位置 (`env.scene.env_origins`) からの水平距離。25cm で -0.63、50cm で -0.98 |
| `feet_slip_penalty` | **接地している足だけ**の水平速度。遊脚は見ないので踏み替え自体は妨げない |
| `heading_penalty` | 初期ヨー向きからのずれ。`cos(yaw) = 2w^2 - 1` を使い、ラップアラウンド問題を回避 |
| `base_speed_penalty` | 胴体の水平速度。上下 (z) は見ないので沈み込み・立ち上がりを阻害しない |

### 転倒対策

正報酬側は「合計を正に保つ床」として小さく置くだけ。実際の抑止は終了条件が担当する。

| 関数 | 内容 |
|---|---|
| `upright_bonus` | `(1 - g_z) / 2`。立位 1.0 / 横倒し 0.5 / 逆さま 0 |
| `feet_grounded` | 接地している足の割合。正報酬なので跳ねる解を潰す |
| `torso_roll_penalty` | `projected_gravity_b[:, 1]` のみ。**前傾を一切罰さずにロールだけ潰す** |

終了条件:

| cfg 名 | 関数 | 設定 | 意図 |
|---|---|---|---|
| `fell_over` | `bad_orientation` | limit_angle=1.2 | 約 69 度傾いたら終了 |
| `collapsed` | `root_height_below_minimum` | 0.20 m | 骨盤が沈み込んだら終了 |
| `base_contact` | — | None (無効化) | 深いしゃがみでの偶発的な骨盤接触を許す |

### 内部ヘルパー

| 関数 | 戻り値 | 役割 |
|---|---|---|
| `_squat_phase` | (N,) | [0,1) の周期位相 |
| `_squat_depth` | (N,) | [0,1] のしゃがみ深さ (余弦) |
| `_bodies_in_yaw_frame` | (N,K,3) | 骨盤原点・ヨーのみ揃えた座標系。「前」が胴の前傾によらず水平前方を指す |
| `_sorted_by_lateral` | (N,2,3) | y 座標で並べ替え、左右の対応付けを保証。肩・肘・手を別々の正規表現で引くと `find_bodies` の順序が一致する保証がないため必要 |

---

## 参照姿勢の定数

`squat_only_env_cfg.py` 冒頭。運動学で重心を検算した上で決めている (原則 4)。

| 定数 | 立ち | 完全しゃがみ | soft 限界 | 備考 |
|---|---:|---:|---|---|
| `HIP_PITCH` | -0.10 | -2.10 | -2.26 | 余裕 0.16 |
| `KNEE` | 0.30 | 2.20 | 2.73 | 126 度 |
| `ANKLE` | -0.20 | -0.75 | -0.803 | 余裕 0.053 |
| `HEIGHT` | 0.73 | 0.39 | — | 骨盤高 [m] |
| `TORSO_PITCH` | 0.00 | 0.65 | — | 前傾 37 度 |
| `ARM_FWD` | 0.00 | 0.55 | — | 上腕の前方成分 |
| `ABDUCTION` | 0.00 | 0.18 | — | \|hip_roll\| 許容量 |
| `WIDTH` | 0.20 | 0.28 | — | 足の左右間隔 [m] |

この姿勢の重心:

| 項目 | 値 | 判定 |
|---|---:|---|
| COM_x | +0.046 m | 足首より前 |
| 踵からの余裕 | 0.106 m | 後方転倒への余裕 |
| つま先までの余裕 | 0.104 m | 前方転倒への余裕 (ほぼ均衡) |
| 手から箱まで | 0.146 m | 箱を中心 (0.35, 0.10) と仮定 |

> **寸法は推定値**: 運動学の計算に使った大腿・下腿 0.30 m、胴 0.32 m、腕 0.36 m は
> G1 の公称身長から較正したもの。`knee = 2*|ankle|` という関係自体は比率で決まるので
> 寸法誤差に強いが、骨盤高 0.39 m などの絶対値には数 cm の誤差があり得る。
> `height_track` が 2.0/3.0 を超えない場合は実測が必要。

---

## 落とし穴

いずれも「エラーは出ないが学習が壊れる」種類。症状から原因に辿り着きにくい。

### 1. SceneEntityCfg をデフォルト引数に置くと解決されない

Isaac Lab の `ManagerBase._resolve_common_term_cfg` は **`term_cfg.params` の中にある
SceneEntityCfg だけ**を `resolve(scene)` する。関数のデフォルト引数は一切見ない。

結果として `joint_ids` が `slice(None)` のまま残り、意図した 2 関節ではなく
**全 29 関節**を指す。`hip_roll` の絶対値和のつもりが「どの関節も動かすな」という
命令になり、ロボットが硬直した。

**対策**: SceneEntityCfg は必ず `RewTerm(params=...)` に明示的に書く。
モジュール定数にまとめて共有すると書き忘れにくい。

### 2. 報酬の合計が負だとエージェントは自殺する

1 ステップの合計が負に振れると、エピソードを早く終わらせた方が return が大きくなる。
方策は**意図的に転倒する**ようになる。「寝転がって動かない」症状はこれ。

**対策**: 全ペナルティを有界形にし、正報酬の床が常にペナルティ合計を上回るよう配点する。

### 3. 両側の追従目標は「やらないこと」まで罰する

開脚の許容量を `exp(-(abd - target)^2/std^2)` の追従目標にすると、
**脚を閉じた棒立ちが「目標より閉じすぎ」で減点される**。実測で -1.97 の不当な減点。

**対策**: 上限として扱うものは `excess = (value - target).clamp(min=0)` の片側にする。
片側なら重みを上げても正しい姿勢のコストが 0 なので遠慮なく強められる。

### 4. 「対称」は「離れている」を意味しない

`hands_symmetry_penalty` は前後 x の一致・左右 y の和が 0・上下 z の一致を要求する。
**両手が y=0 で重なっていてもすべて満たす**。間隔を要求する項が別途必要。

### 5. グループ平均は追加した関節でシグナルが薄まる

`squat_pose_tracking` の lateral 項はグループ内の平均。「胴を正面に向ける」対応で
waist を 2 関節追加したところ、hip_roll の誤差寄与が 0.125 -> 0.083 に**33% 薄まった**。

**対策**: 抑止力を効かせたい関節はグループに混ぜず、専用項として切り出す。

### 6. 端点の位置で測ると途中の関節で達成される

`arm_forward_direction` を「肩 -> 手」で測っていたため、**肩を回さず肘を曲げるだけで
前方成分を稼げた**。arm_ext_pen (3.0) より arm_forward (5.0) が強く、折れ曲がる方が得。

**対策**: 動かしたい関節の直後のリンクで測る (肩 -> 肘)。

### 7. 位置目標と向き目標は幾何的に競合する

腕の長さは固定なので、**腕を前に振れば手は必然的に上がる**。「手を膝の高さに」という
位置目標と「腕を前方へ」という向き目標が正面から逆方向に引っ張り合っていた。

**対策**: 腕の姿勢は向き・伸展・間隔の 3 つで完全に決まる。手の絶対位置は指定しない。

### 8. 学習率が下限に張り付いたら報酬設計を疑う

rsl_rl の adaptive スケジューラは KL が目標を超え続けると学習率を割り続け、
下限 (1e-5) で止まる。**報酬が荒いと方策更新が暴れて KL が跳ね、これが起きる**。
エントロピーが初期値のまま横ばいなら方策は何も学習していない。

**対策**: 500 iteration 以内に `Loss/learning_rate` を確認する。下限に張り付いて
いたらそのまま回しても改善しないので停止して報酬を見直す。

---

## 未使用の関数

`rewards.py` に残っているが現在の `PeriodicSquatRewardsCfg` では登録していないもの。

### 箱の pick & carry タスク用 — 削除禁止

`pickup_carry_env_cfg.py` (本タスク) が参照している。

| 関数 | フェーズ |
|---|---|
| `approach_box` / `face_box` | 1-2 箱へ接近し正対する |
| `squat_when_near_box` / `hold_still_when_squatting` / `knee_flexion_when_squatting` | 3 箱の近くでしゃがむ |
| `hands_near_box` / `hands_contact_box` | 4 両手を箱へ |
| `is_grasped` / `grasp_bonus` | 5 掴む |
| `lift_box` / `stand_up_when_lifting` | 6 持ち上げて立つ |
| `carry_box_velocity` | 7 運ぶ |
| `drop_box_penalty` / `box_collision_penalty` | 安全 |

### スクワット実験の旧版 — 参照が無ければ削除可

| 旧関数 | 置き換え先 | 理由 |
|---|---|---|
| `knee_bent_reward` / `hip_pitch_bent_reward` / `height_low_gated_by_knee` | `squat_pose_tracking` | 関節ごとの個別項は互いに矛盾する |
| `periodic_height_target` / `periodic_knee_target` / `periodic_hip_pitch_target` | `squat_pose_tracking` | 1 本の参照姿勢に統合 |
| `hip_abduction_penalty` / `hip_roll_magnitude_penalty` / `feet_lateral_distance_penalty` / `leg_symmetry_penalty` | `hip_abduction_tracking`, `stance_width_penalty_phased` | 非有界かつ両側。落とし穴 2・3 |
| `stay_in_place` / `low_base_speed` / `heading_hold` / `feet_no_slip` / `feet_stance_width` | `*_penalty` 版 | 正報酬だと棒立ちで満点。原則 1 |
| `stance_width_penalty` | `stance_width_penalty_phased` | 固定目標では深さに応じた許容ができない |
| `hands_forward_tracking` / `hands_at_knee_front` | `arm_forward_direction` | 位置目標が向き目標と競合。落とし穴 7 |
| `fallen_penalty` | `bad_orientation` (終了条件) | 大きな負の報酬は自殺を招く。原則 3 |
| `freeze_penalty` / `feet_air_time_penalty` | — | 周期目標が立ち止まりを直接潰すため不要 |

---

## 学習時に最初に見る指標

| 指標 | 健全 | 異常時の意味 |
|---|---|---|
| `Loss/learning_rate` | 1e-4 以上 | **下限張り付き -> 即停止**。落とし穴 8 |
| `Loss/entropy` | 単調減少 | 横ばい -> 何も学習していない |
| `Episode_Reward/dof_pos_lim` | -0.1 以上 | -0.5 以下 -> 参照姿勢が関節限界に食い込んでいる |
| `Episode_Reward/torso_pitch` | 2.4 以上 / 3.0 | 低い -> 前傾できず重心が後ろのまま |
| `Episode_Reward/pose_fine` | 上昇し続ける | 頭打ち -> 参照姿勢がまだ不安定 |
| `Episode_Termination/fell_over` | 減少傾向 | 4 割超 -> 深さが物理的に無理 |
