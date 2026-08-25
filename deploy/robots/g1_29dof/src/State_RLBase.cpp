#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands
// To use it, change the "velocity_commands" observation name in the policy
// deploy.yaml to "keyboard_velocity_commands".
// Keys (press in the g1_ctrl terminal window):
//   w / s : forward / backward   (lin_vel_x)
//   a / d : left / right         (lin_vel_y)
//   q / e : turn left / right    (ang_vel_z)
// The commanded value is taken from the range bounds in deploy.yaml so it
// stays inside the policy's training distribution.
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard ? FSMState::keyboard->key() : std::string();
    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    const float vx_min = cfg["lin_vel_x"][0].as<float>();
    const float vx_max = cfg["lin_vel_x"][1].as<float>();
    const float vy_min = cfg["lin_vel_y"][0].as<float>();
    const float vy_max = cfg["lin_vel_y"][1].as<float>();
    const float wz_min = cfg["ang_vel_z"][0].as<float>();
    const float wz_max = cfg["ang_vel_z"][1].as<float>();

    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if(key == "w")      cmd = {vx_max, 0.0f, 0.0f};
    else if(key == "s") cmd = {vx_min, 0.0f, 0.0f};
    else if(key == "a") cmd = {0.0f, vy_max, 0.0f};
    else if(key == "d") cmd = {0.0f, vy_min, 0.0f};
    else if(key == "q") cmd = {0.0f, 0.0f, wz_max};
    else if(key == "e") cmd = {0.0f, 0.0f, wz_min};

    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
