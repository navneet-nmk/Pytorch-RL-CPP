//
// Created by Navneet Madhu Kumar on 2019-07-10.
//

#include "ExperienceReplay.h"
#include "dqn.h"
#include <torch/torch.h>
#include "/Users/navneetmadhukumar/Downloads/Arcade-Learning-Environment-master/src/ale_interface.hpp"
#include <math.h>

class Trainer{

    private: ExperienceReplay buffer;
    private: DQN network, target_network;
    private: torch::optim::Adam dqn_optimizer;
    private: ALEInterface ale;
    private: double epsilon_start = 1.0;
    private: double epsilon_final = 0.01;
    private: int64_t epsilon_decay = 500;
    private: int64_t batch_size = 32;
    private: float gamma = 0.99;

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


        torch::Tensor states_tensor;
        torch::Tensor new_states_tensor;
        torch::Tensor actions_tensor;
        torch::Tensor rewards_tensor;
        torch::Tensor dones_tensor;

        states_tensor = torch::cat(states, 0);
        new_states_tensor = torch::cat(new_states, 0);
        actions_tensor = torch::cat(actions, 0);
        rewards_tensor = torch::cat(rewards, 0);
        dones_tensor = torch::cat(dones, 0);


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

    void load_enviroment(int64_t random_seed, std::string rom_path){
        ale.setInt("random_seed", random_seed);
        ale.setBool("display_screen", true);
        ale.loadROM(rom_path);

    }

    double epsilon_by_frame(int64_t frame_id){
        return epsilon_final + (epsilon_start - epsilon_final) * exp(-1. * frame_id / epsilon_decay);
    }

    torch::Tensor get_tensor_observation(std::vector<unsigned char> state){
        torch::Tensor state_tensor = torch::from_blob(std::data(state), {3, 210, 160});
        return state_tensor;
    }

    void update_target_model(){
        torch::save(network, "network-checkpoint.pt");
        torch::load(target_network, "network-checkpoint.pt");
    }

    void train(int64_t random_seed, std::string rom_path, int64_t num_epochs){
        load_enviroment(random_seed, rom_path);
        ActionVect legal_actions = ale.getLegalActionSet();
        ale.reset_game();
        std::vector<unsigned char> state;
        ale.getScreenRGB(state);
        float episode_reward = 0.0;
        std::vector<float> all_rewards;
        std::vector<torch::Tensor> losses;

        for(int i=0; i<num_epochs; i++){
            double epsilon = epsilon_by_frame(i);
            auto r = ((double) rand() / (RAND_MAX));
            torch::Tensor state_tensor = get_tensor_observation(state);
            Action a;
            if (r <= epsilon){
                a = legal_actions[rand() % legal_actions.size()];
            }
            else{
                torch::Tensor action_tensor = network.act(state_tensor);
                int index = action_tensor[0].item<int>();
                a = legal_actions[index];
            }

            float reward = ale.act(a);
            episode_reward += reward;
            std::vector<unsigned char> new_state;
            ale.getScreenRGB(new_state);
            torch::Tensor new_state_tensor = get_tensor_observation(new_state);
            bool done = ale.game_over();

            torch::Tensor reward_tensor = torch::tensor(reward);
            torch::Tensor done_tensor = torch::tensor(done);
            torch::Tensor action_tensor_new = torch::tensor(a);

            buffer.push(state_tensor, new_state_tensor, action_tensor_new, done_tensor, reward_tensor);

            state = new_state;

            if (done){
                ale.reset_game();
                std::vector<unsigned char> state;
                ale.getScreenRGB(state);
                all_rewards.push_back(episode_reward);
                episode_reward = 0.0;
            }

            if (buffer.size_buffer() > batch_size){
                torch::Tensor loss = compute_td_loss(batch_size, gamma);
                losses.push_back(loss);
            }

            if (i%100==0){
                update_target_model();
            }

        }


    }


};

