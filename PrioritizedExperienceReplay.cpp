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


PrioritizedExperienceReplay::PrioritizedExperienceReplay(int64_t size, float_t prob_alpha) {

    capacity = size;
}

void PrioritizedExperienceReplay::push(torch::Tensor state,torch::Tensor new_state,torch::
Tensor action,torch::Tensor done,torch::Tensor reward){

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample (state, new_state, action, reward, done);
    if (buffer.size() < capacity){
        buffer.push_back(sample);
    }
    else {
        while (buffer.size() >= capacity) {
            buffer.pop_front();
        }
        buffer.push_back(sample);
    }
}

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
PrioritizedExperienceReplay::sample_queue(
        int64_t batch_size){
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> b(batch_size);
    return b;
}

int64_t PrioritizedExperienceReplay::size_buffer(){

    return buffer.size();
}
