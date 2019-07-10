//
// Created by Navneet Madhu Kumar on 2019-07-08.
//

#ifndef REINFORCEMENT_CPP_STORAGE_H
#define REINFORCEMENT_CPP_STORAGE_H

#endif //REINFORCEMENT_CPP_STORAGE_H


#include <memory>
#include <vector>
#include <iostream>

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>
#include <algorithm>
#include <iterator>
#include <random>

class ExperienceReplay{

    private:torch::Tensor state, new_state, action, done, reward;
    private: int64_t capacity;
    public: std::deque<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> buffer;

    public:
        ExperienceReplay (int64_t capacity);
        void push(torch::Tensor state,torch::Tensor new_state,torch::
        Tensor action,torch::Tensor done,torch::Tensor reward);
        std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> sample_queue(int64_t batch_size);


};