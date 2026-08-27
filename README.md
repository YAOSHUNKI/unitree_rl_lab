# Unitree RL Lab

## 参照

[unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) \
[IsaacLab2.3.0](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html) \
[unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation)

## 目標タスク

原子炉建屋内での瓦礫撤去
Unitree G1の強化学習による開発
- 瓦礫までの移動
- 瓦礫をピックアップするために腰を下ろす動作
- 瓦礫を掴む
- 立ち上がる

## 環境構築

### Isaaclab
既存のIsaac Labのインストール手順では依存関係で問題が発生するため修正した手順を記す. 
- Install Isaac Lab 2.3.0 + Isaac Sim 5.1
  
  - conda環境の作成
    ```bash
    conda create -n env_isaaclab python=3.11 -y
    conda activate env_isaaclab
    pip install --upgrade pip
    ```

  - pkg_resources問題を回避するためにsetuptoolsをpln
    ```bash
    pip install "setuptools<81" wheel
    ```

  - build isolation を回避するためflatdict を先に手動 installする
    ```bash
    pip install flatdict==4.0.1 --no-build-isolation
    ```
     
  - Isaac Sim 5.1をinstall
    ```bash
    PIP_CONSTRAINT=<(echo "setuptools<81")
    pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
    ```
     
   - PyTorch (cu128) を install
     ```bash
     pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 
     --index-url https://download.pytorch.org/whl/cu128
     ```
     
   - Isaac Sim 動作確認
     ```bash
     isaacsim
     # 初回起動は時間がかかる
     ```
     
   - Isaac Lab clone & checkout
     ```bash
     cd ~
     git clone https://github.com/isaac-sim/IsaacLab.git
     cd IsaacLab
     git checkout v2.3.0  
     sudo apt install -y cmake build-essential
     ```

   - Isaac Lab を editable install
     ```bash
     ./isaaclab.sh --install
     ```
     
   - torch と Isaac Sim pin を復元
     ```bash
     # torch を 2.7.0 に戻す 
     pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
     --index-url https://download.pytorch.org/whl/cu128

     # isaacsim-kernel の pin 要件を復元
     pip install "click==8.1.7" "typing_extensions==4.12.2" "psutil==5.9.8"

     # stable-baselines3 を torch 2.7 対応版に 
     pip install "stable-baselines3==2.6.0"
     ```
     
   - 動作確認
     ```bash
     ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
     ```
### 改良版Unitree RL Labのinstall

- Uniree RL lab環境のinstall 
  - git clone this repository
    ```bash
    git clone https://github.com/unitreerobotics/unitree_rl_lab.git
    ```
    ```bash
    conda activate env_isaaclab
    cd ~/uniree_rl_lab/
    ./unitree_rl_lab.sh -i
    # restart your shell to activate the environment changes.
    ```
- Unitreeロボットのdescriptionファイルをインストールする
  ```bash
  cd ~
  git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
  ```
     
  ```bash
  cd ~
  git clone https://github.com/unitreerobotics/unitree_ros.git
  ```
    
  このあと公式のリポジトリでは以下のようなPATHの書き換え作業があるがこのリポジトリでは自動反映するようにしているため必要ない
  ```bash
  UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
  UNITREE_ROS_DIR = "</home/user/projects/unitree_ros/unitree_ros>"
  ```

### Deploy

モデルの学習が完了したら, Mujocoで学習済みのpolicyに対してsim2simを実行し,モデルの性能を検証する必要がある. 
その後、sim2realを展開する. 

```bash
# Install dependencies
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
# Install unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # Install on the /usr/local directory
sudo make install
# Compile the robot_controller
cd unitree_rl_lab/deploy/robots/g1_29dof # or other robots
mkdir build && cd build
cmake .. && make
```

### Sim2Sim

- unitree mujocoのinstall
  - 依存関係
    ```bash
    sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev \
                      libspdlog-dev libfmt-dev \
                      libglfw3-dev libxinerama-dev libxcursor-dev libxi-dev
    ```
    
  - unitree_sdk2のビルド・インストール
    ```bash
    git clone https://github.com/unitreerobotics/unitree_sdk2.git
    cd unitree_sdk2
    mkdir build && cd build
    cmake .. -DBUILD_EXAMPLES=OFF
    sudo make install sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev \
                          libspdlog-dev libfmt-dev \
                          libglfw3-dev libxinerama-dev libxcursor-dev libxi-dev
    ```
   
  - mujocoのバイナリ
    ```bash
    mkdir -p ~/.mujoco && cd ~/.mujoco
    wget https://github.com/google-deepmind/mujoco/releases/download/3.3.6/mujoco-3.3.6-linux-x86_64.tar.gz
    tar -xzf mujoco-3.3.6-linux-x86_64.tar.gz
    ``` 

  - uniree mujocoのビルド
    ```bash
    git clone https://github.com/unitreerobotics/unitree_mujoco.git
    cd unitree_mujoco/simulate
    ln -s ~/.mujoco/mujoco-3.3.6 mujoco

    mkdir build && cd build
    cmake ..
    make -j4
    ```
       
  - uniree_rl_labのrobotcontollerのビルド
    ```bash
    cd ~/unitree_rl_lab/deploy/robots/g1_29dof
    mkdir build && cd build
    cmake .. && make
    ```

- 設定
  - Set the `robot` at `/simulate/config.yaml` to g1
  - Set `domain_id` to 0
  - Set `enable_elastic_hand` to 1
  - Set `use_joystck` to 0.




<div align="center">

| <div align="center"> Isaac Lab </div> | <div align="center">  Mujoco </div> |  
|--- | --- |
| [<img src="Materials/Isaaclab.gif" width="240px">](g1_sim.gif) | [<img src="Materials/mujoco.gif" width="240px">](g1_mujoco.gif) | 

</div>

## 学習
- train
  ```bash
  python scripts/rsl_rl/train.py --task Unitree-G1-29dof-<task_name> --num_envs <Number of Parallel Processes> --max_iterations <Number of train> 
  ```
  例：PeriodicSquatタスクを64並列処理で3000回学習する場合
  ```bash
  python scripts/rsl_rl/train.py --task Unitree-G1-29dof-PeriodicSquat --num_envs 64 --max_iterations 3000 
  ```

- play
  ```bash
  python scripts/rsl_rl/play.py --task Unitree-G1-29dof-<run_name> --checkpoint logs/rsl_rl/g1_pickup_carry/<run_dir_name>/<model>.pt
  ```
  例：PeriodicSquatタスクの2026-8-24_16-54-51のログファイルの中にあるmodel_600.ptを再生する場合
  ```bash
  python scripts/rsl_rl/play.py --task Unitree-G1-29dof-PeriodicSquat --checkpoint logs/rsl_rl/g1_pickup_carry/2026-08-24_16-54-51/model_600.pt
  ```
  
- restart train 
  ```bash
  python scripts/rsl_rl/train.py --task Unitree-G1-29dof-PeriodicSquat \
  --resume True --load_run <run_dir_name> --checkpoint model_1500.pt \
  --max_iterations 3000
  ```

## Sim2Sim Mujoco
- 学習済みのポリシーをplayする
  ```bash
  python scripts/rsl_rl/play.py --task Unitree-G1-29dof-PeriodicSquat --checkpoint logs/rsl_rl/g1_pickup_carry/2026-08-24_16-54-51/model_600.pt
  ```
- taskを配置する
  `unitree_rl_lab/deploy/mujoco_py/task/`にフォルダーを作成する. \
  このフォルダーに先程playしたポリシーのログファイルの中にある`params/deploy.yaml`と`exported/policy.onnx`を配置し, `task.yaml`を作成・記述し配置する. 記述方法は`deploy/mujoco_py`にあるREADME.mdを参照

- mujocoで再生
  ```bash
  cd ~/unitree_rl_lab/deploy/mujoco_py
  python run_mujoco.py --task <task_name> 
  ```
  例：タスク名がsquatだった場合
  ```bash
  cd ~/unitree_rl_lab/deploy/mujoco_py
  python run_mujoco.py --task squat 
  ```

  
  
