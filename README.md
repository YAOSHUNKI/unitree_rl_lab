# Unitree RL Lab

## 参照

[unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) \
[IsaacLab2.3.0](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html)

## 目標タスク

原子炉建屋内での瓦礫撤去
Unitree G1の強化学習による開発
- 瓦礫までの移動
- 瓦礫をピックアップするために腰を下ろす動作
- 瓦礫を掴む
- 立ち上がる

## 作業内容

### 開発環境の構築
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


<div align="center">

| <div align="center"> Isaac Lab </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="Materials/screencast-from-2026-08-26-10-37-56_RVfoMS3u.gif" width="240px">](g1_sim.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](g1_mujoco.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](g1_real.gif) |

</div>




## Deploy

After the model training is completed, we need to perform sim2sim on the trained strategy in Mujoco to test the performance of the model.
Then deploy sim2real.

### Setup

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

Installing the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation).

- Set the `robot` at `/simulate/config.yaml` to g1
- Set `domain_id` to 0
- Set `enable_elastic_hand` to 1
- Set `use_joystck` to 1.

```bash
# start simulation
cd unitree_mujoco/simulate/build
./unitree_mujoco
# ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # alternative
```

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
# 1. press [L2 + Up] to set the robot to stand up
# 2. Click the mujoco window, and then press 8 to make the robot feet touch the ground.
# 3. Press [R1 + X] to run the policy.
# 4. Click the mujoco window, and then press 9 to disable the elastic band.
```

### Sim2Real

You can use this program to control the robot directly, but make sure the on-borad control program has been closed.

```bash
./g1_ctrl --network eth0 # eth0 is the network interface name.
```

## Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.
- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [robot_lab](https://github.com/fan-ziqi/robot_lab): Referenced for project structure and parts of the implementation.
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): Versatile humanoid control framework for motion tracking.
