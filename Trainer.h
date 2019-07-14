//
// Created by Navneet Madhu Kumar on 2019-07-12.
//
#pragma once

#include <torch/torch.h>

#include "ExperienceReplay.h"
#include "dqn.h"
#include "/Users/navneetmadhukumar/Downloads/Arcade-Learning-Environment-master/src/ale_interface.hpp"



class Trainer{

    private: ExperienceReplay buffer;
    private: DQN network, target_network;
    private: torch::optim::Adam dqn_optimizer;
    private: ALEInterface ale;
    private: double epsilon_start = 1.0;
    private: double epsilon_final = 0.01;
    private: int64_t epsilon_decay = 30000;
    private: int64_t batch_size = 32;
    private: float gamma = 0.99;

    public:
        Trainer(int64_t input_channels, int64_t num_actions, int64_t capacity);
        torch::Tensor compute_td_loss(int64_t batch_size, float gamma);
        void load_enviroment(int64_t random_seed, std::string rom_path);
        double epsilon_by_frame(int64_t frame_id);
        torch::Tensor get_tensor_observation(std::vector<unsigned char> state);
        void loadstatedict(torch::nn::Module& model,
                           torch::nn::Module& target_model);
        void train(int64_t random_seed, std::string rom_path, int64_t num_epochs);





};
