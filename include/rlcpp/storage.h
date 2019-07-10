//
// Created by Navneet Madhu Kumar on 2019-07-05.
//


#include <memory>
#include <c10/util/ArrayRef.h>
#include <torch/torch.h>


namespace rlcpp
{

    class ExperienceReplay{

    private:
        torch::Tensor observations, next_states, actions, rewards, dones;
        torch::Device device;
        int64_t num_steps;
        int64_t step;

    public:
        ExperienceReplay(int64_t num_steps,
        int64_t num_processes,
                c10::ArrayRef<int64_t> obs_shape,
        ActionSpace action_space,
                int64_t hidden_state_size,
        torch::Device device);
    };




}

