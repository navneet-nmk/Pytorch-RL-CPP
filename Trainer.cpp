//
// Created by Navneet Madhu Kumar on 2019-07-10.
//

#include "ExperienceReplay.h"
#include "dqn.h"
#include <torch/torch.h>

class Trainer{

    private: ExperienceReplay buffer;
    private: DQN network, target_network;
    private: torch::optim::Adam dqn_optimizer;


    Trainer(int64_t input_channels, int64_t num_actions, int64_t capacity):
        buffer(capacity),
        network(input_channels, num_actions),
        target_network(input_channels, num_actions),
        dqn_optimizer(
            network.parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5)){}

    torch::Tensor compute_td_loss(int64_t batch_size, float gamma){
        std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> batch =
                buffer.sample_queue(batch_size);

        std::vector<torch::Tensor> states;
        std::vector<torch::Tensor> new_states;
        std::vector<torch::Tensor> actions;
        std::vector<torch::Tensor> rewards;
        std::vector<torch::Tensor> dones;

        for (auto i : batch){
            states.push_back(std::get<0>(i));
            new_states.push_back(std::get<1>(i));
            actions.push_back(std::get<2>(i));
            rewards.push_back(std::get<3>(i));
            dones.push_back(std::get<4>(i));
        }

        // Serialize and load
        std::stringstream stream_states, stream_new_states, stream_actions, stream_rewards, stream_dones;
        torch::save(states, stream_states);
        torch::save(new_states, stream_new_states);
        torch::save(actions, stream_actions);
        torch::save(rewards, stream_rewards);
        torch::save(dones, stream_dones);


        torch::Tensor states_tensor;
        torch::Tensor new_states_tensor;
        torch::Tensor actions_tensor;
        torch::Tensor rewards_tensor;
        torch::Tensor dones_tensor;

        torch::load(states_tensor, stream_states);
        torch::load(new_states_tensor, stream_new_states);
        torch::load(actions_tensor, stream_actions);
        torch::load(rewards_tensor, stream_rewards);
        torch::load(dones_tensor, stream_dones);


        torch::Tensor q_values = network.forward(states_tensor);
        torch::Tensor next_target_q_values = target_network.forward(new_states_tensor);
        torch::Tensor next_q_values = network.forward(new_states_tensor);

        torch::Tensor q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1);
        torch::Tensor maximum = std::get<1>(next_q_values.max(1));
        torch::Tensor next_q_value = next_target_q_values.gather(1, maximum.unsqueeze(1)).squeeze(1);
        torch::Tensor expected_q_value = rewards_tensor + gamma*next_q_value*(1-dones_tensor);
        torch::Tensor loss = torch::mse_loss(q_value, expected_q_value);

        dqn_optimizer.zero_grad();
        loss.backward();
        dqn_optimizer.step();

        return loss;

    }


};

