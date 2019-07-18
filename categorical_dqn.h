//
// Created by Navneet Madhu Kumar on 2019-07-18.
//


#pragma once


#include <torch/torch.h>
#include "noisy.h"

struct CategoricalDQN : torch::nn::Module{
    CategoricalDQN(int64_t num_inputs, int64_t num_actions, int64_t num_atoms, int64_t Vmin, int64_t Vmax)
            :
            num_atoms(num_atoms),
            num_actions(num_actions),
            Vmin(Vmin),
            Vmax(Vmax),
            linear1(torch::nn::Linear(num_inputs, 128)),
            linear2(torch::nn::Linear(128, 128)),
            noisy1(NoisyLinear(128, 512, 0.4)),
            noisy2(NoisyLinear(512, num_actions*num_atoms, 0.4)){

    }

    torch::Tensor forward(torch::Tensor input) {
        // Flatten the output
        input = torch::relu(linear1(input));
        input = torch::relu(linear2(input));
        input = torch::relu(noisy1.forward(input));
        input = noisy2.forward(input);
        input = torch::softmax(input.view({-1, num_atoms}), 1).view({-1, num_actions, num_atoms});
        return input;
    }

    torch::Tensor act(torch::Tensor state){
        torch::Tensor dist = forward(state);
        dist = dist*torch::linspace(Vmin, Vmax, num_atoms);
        torch::Tensor action = std::get<1>(dist.sum(2).max(1));
        return action;
    }

    int64_t num_atoms;
    int64_t num_actions;
    int64_t Vmin;
    int64_t Vmax;
    torch::nn::Linear linear1, linear2;
    NoisyLinear noisy1, noisy2;


};
