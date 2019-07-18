cmake_minimum_required(VERSION 3.14)
project(Reinforcement_CPP
        LANGUAGES CXX
        VERSION 1.0.0
        DESCRIPTION "Reinforcement learning in C++ using PyTorch"
        )

set(CMAKE_CXX_STANDARD 17)

set(Torch_DIR /Users/navneetmadhukumar/Downloads/libtorch/share/cmake/Torch)


set(CPPRL_INCLUDE_DIRS
        include
        src
        )

find_package(Torch REQUIRED)

add_executable(Reinforcement_CPP main.cpp ExperienceReplay.cpp ExperienceReplay.h dqn.cpp Trainer.cpp dqn.h Trainer.h PrioritizedExperienceReplay.cpp PrioritizedExperienceReplay.h)


include_directories(${CPPRL_INCLUDE_DIRS})

find_library(LIBALE libale.so /Users/navneetmadhukumar/Downloads/Arcade-Learning-Environment-master/)
target_link_libraries(Reinforcement_CPP  ${LIBALE} ${TORCH_LIBRARIES} )
