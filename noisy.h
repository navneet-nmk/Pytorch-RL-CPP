//
// Created by Navneet Madhu Kumar on 2019-07-18.
//

#pragma once


#include <torch/torch.h>
#include <math.h>

struct NoisyLinear : torch::nn::Module{
    NoisyLinear(int64_t in_features, int64_t out_features, float_t std_init){
        W_mu = register_parameter("W_mu", torch::randn({out_features, in_features}));
        W_sigma = register_parameter("W_sigma", torch::randn({out_features, in_features}));
        b_mu = register_parameter("b_mu", torch::randn(out_features));
        b_sigma = register_parameter("b_sigma", torch::randn(out_features));
        W_epsilon = register_buffer("W_epsilon", torch::randn({out_features, in_features}));
        b_epsilon  = register_buffer("b_epsilon", torch::randn(out_features));

        reset_params(std_init);
        reset_noise(in_features, out_features);

    }

    torch::Tensor forward(torch::Tensor input) {
        torch::Tensor weight;
        torch::Tensor bias;
        if (is_training()){
            weight = W_mu + W_sigma.mul(W_epsilon);
            bias = b_mu + b_sigma.mul(b_epsilon);

        }else{
            weight = W_mu;
            bias = b_mu;
        }

        return torch::addmm(bias, input, weight);
    }

    void reset_params(float_t std_init){
        float_t mu_range = 1 / sqrt(W_mu.size(1));
        W_mu.uniform_(-mu_range, mu_range);
        W_sigma.fill_(std_init / sqrt(W_sigma.size(1)));

        b_mu.uniform_(-mu_range, mu_range);
        b_sigma.fill_(std_init / sqrt(b_sigma.size(1)));

    }
    void reset_noise(int64_t in_features, int64_t out_features){
        torch::Tensor epsilon_in = scale_noise(in_features);
        torch::Tensor epsilon_out = scale_noise(out_features);

        W_epsilon.copy_(epsilon_out.ger(epsilon_in));
        b_epsilon.copy_(scale_noise(out_features));

    }

    torch::Tensor scale_noise(int64_t size){
        torch::Tensor x = torch::randn(size);
        x = x.sign().mul(x.abs().sqrt());
        return x;
    }

    torch::Tensor W_mu, W_sigma, b_mu, b_sigma, W_epsilon, b_epsilon;
};



