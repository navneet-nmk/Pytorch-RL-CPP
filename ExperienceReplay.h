//
// Created by Navneet Madhu Kumar on 2019-07-08.
//
#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <iostream>


#include <algorithm>
#include <iterator>
#include <random>

class ExperienceReplay{

    private: int64_t capacity;
    public: std::deque<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> buffer;

    public:
        ExperienceReplay (int64_t capacity);
        void push(torch::Tensor state,torch::Tensor new_state,torch::
        Tensor action,torch::Tensor done,torch::Tensor reward);
        int64_t size_buffer();
        std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> sample_queue(int64_t batch_size);


};