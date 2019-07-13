//
// Created by Navneet Madhu Kumar on 2019-07-08.
//

#include "ExperienceReplay.h"
#include <memory>
#include <vector>
#include <iostream>

#include <torch/torch.h>
#include <c10/util/ArrayRef.h>


#include <algorithm>
#include <iterator>
#include <random>


ExperienceReplay::ExperienceReplay(int64_t size) {

    capacity = size;
}

void ExperienceReplay::push(torch::Tensor state,torch::Tensor new_state,torch::
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

    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> ExperienceReplay::sample_queue(
            int64_t batch_size){
        std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> b(batch_size);
        std::sample(buffer.begin(), buffer.end(),
                    b.begin(), b.size(),
                    std::mt19937{std::random_device{}()});
        return b;
    }

    int64_t ExperienceReplay::size_buffer(){

        return buffer.size();
}
