# Pytorch-RL-CPP
<img width="160px" height="22px" href="https://github.com/pytorch/pytorch" src="https://pp.userapi.com/c847120/v847120960/82b4/xGBK9pXAkw8.jpg"> 

A Repository with C++ implementations of Reinforcement Learning Algorithms (Pytorch)
 

<img src="/assets/pong_dqn.gif?raw=true" width="200">

**RlCpp is a reinforcement learning framework, written using the [PyTorch C++ frontend](https://pytorch.org/cppdocs/frontend.html).**

RlCpp aims to be an extensible, reasonably optimized, production-ready framework for using reinforcement learning in projects where Python isn't viable. It should be ready to use in desktop applications on 
user's computers with minimal setup required on the user's side.

The Environment used is the C++ Port of [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) 

# Currently Supported Models
1. Double DQN
* Plans to support more models and more sophisticated methods in the future.

# Results for Pong using Double DQN
<img src="/assets/dqn_pong_results.png" width="600">


# Installing the dependencies

# Arcade Learning Environment

Install main dependences:
```
sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
```

Compilation:

```
$ mkdir build && cd build
$ cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
$ make -j 4
```

To install python module:

```
$ pip install .
or
$ pip install --user .
```

Getting the ALE to work on Visual Studio requires a bit of extra wrangling. You may wish to use IslandMan93's [Visual Studio port of the ALE.](https://github.com/Islandman93/Arcade-Learning-Environment)

To ask questions and discuss, please join the [ALE-users group](https://groups.google.com/forum/#!forum/arcade-learning-environment).

# Libtorch

## Building
CMake is used for the build system. 
Most dependencies are included as submodules (run `git submodule update --init --recursive` to get them).
Libtorch has to be [installed seperately](https://pytorch.org/cppdocs/installing.html).

```bash
cd Reinforcement_CPP
cd build
cmake ..
make -j4
```

Before running, make sure to add `libtorch/lib` to your `PATH` environment variable.

## Changes to cmake file

The CMake file requires some changes for things to run smoothly.
1. After building ALE, link the libale.so
2. Set torch dir, after building libtorch.
Refer to the current CMakeLists.txt and make the relevant changes.





