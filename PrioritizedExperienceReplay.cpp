//
// Created by Navneet Madhu Kumar on 2019-07-18.
//



//
// Created by Navneet Madhu Kumar on 2019-07-08.
//
#include "PrioritizedExperienceReplay.h"
#include <memory>
#include <vector>
#include <iostream>

#include <torch/torch.h>
#include <c10/util/ArrayRef.h>


#include <algorithm>
#include <iterator>
#include <random>


PrioritizedExperienceReplay::PrioritizedExperienceReplay(int64_t size) {
    capacity = size;
}

void PrioritizedExperienceReplay::push(torch::Tensor state,torch::Tensor new_state,torch::
Tensor action,torch::Tensor done,torch::Tensor reward, float_t td_error, int64_t ind){
    float_t error(td_error);
    int64_t index(ind);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample (state, new_state, action, reward, done);
    element sample_struct(error, index, sample);

    if (buffer.size() < capacity){
        buffer.push(sample_struct);
    }
    else {
        while (buffer.size() >= capacity) {
            buffer.pop();
        }
        buffer.push(sample_struct);
    }
}

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
PrioritizedExperienceReplay::sample_queue(
        int64_t batch_size){
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> b(batch_size);
    while (batch_size > 0 and buffer.size() > 0){
        element s = buffer.top();
        buffer.pop();
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample = s.transition;
        b.push_back(sample);
    }
    return b;
}

int64_t PrioritizedExperienceReplay::size_buffer(){

    return buffer.size();
}
