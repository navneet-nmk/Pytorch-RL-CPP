//
// Created by Navneet Madhu Kumar on 2019-07-10.
//

#include "ExperienceReplay.h"
#include "dqn.h"
#include <torch/torch.h>

class Trainer{

    private: ExperienceReplay buffer;
    private: DQN network, target_network;


    Trainer(int64_t input_channels, int64_t num_actions, int64_t capacity):
        buffer(capacity),
        network(input_channels, num_actions),
        target_network(input_channels, num_actions){}

    torch::Tensor compute_td_loss(int64_t batch_size){
        std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> batch =
                buffer.sample_queue(batch_size);

    }


};

