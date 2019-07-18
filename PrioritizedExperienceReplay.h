//
// Created by Navneet Madhu Kumar on 2019-07-18.
//


#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <iostream>


#include <algorithm>
#include <iterator>
#include <random>

struct element{
    float_t td_error;
    int64_t index;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> transition;

    element(float_t error, int64_t ind,
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> trans):
            td_error(error),
            index(ind),
            transition(trans)
            {}

    bool operator<(const struct element& other) const{
        if (td_error != other.td_error){
            return td_error < other.td_error;
        }else{
            return index < other.index;
        }
    }
};

class PrioritizedExperienceReplay{

private: int64_t capacity;

public: std::priority_queue<element> buffer;

public:
    PrioritizedExperienceReplay (int64_t capacity);
    void push(torch::Tensor state,torch::Tensor new_state,torch::
    Tensor action,torch::Tensor done,torch::Tensor reward, float_t td_error, int64_t index);
    int64_t size_buffer();
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> sample_queue(int64_t batch_size);


};