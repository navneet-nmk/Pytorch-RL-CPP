# Pytorch-RL-CPP
<img width="160px" height="22px" href="https://github.com/pytorch/pytorch" src="https://pp.userapi.com/c847120/v847120960/82b4/xGBK9pXAkw8.jpg"> 

A Repository with C++ implementations of Reinforcement Learning Algorithms (Pytorch)
 

<img src="/assets/pong_dqn.gif?raw=true" width="200">

**RlCpp is a reinforcement learning framework, written using the [PyTorch C++ frontend](https://pytorch.org/cppdocs/frontend.html).**

RlCpp aims to be an extensible, reasonably optimized, production-ready framework for using reinforcement learning in projects where Python isn't viable. It should be ready to use in desktop applications on 
user's computers with minimal setup required on the user's side.

The Environment used is the C++ Port of [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) 

# Currently Supported Models
The deep reinforcement learning community has made several independent improvements to the DQN algorithm. This repository presents latest extensions to the DQN algorithm: 

  1. Playing Atari with Deep Reinforcement Learning [[arxiv]](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
  2. Deep Reinforcement Learning with Double Q-learning [[arxiv]](https://arxiv.org/abs/1509.06461) 
  3. Dueling Network Architectures for Deep Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1511.06581) 
  4. Prioritized Experience Replay [[arxiv]](https://arxiv.org/abs/1511.05952) 
  5. Noisy Networks for Exploration [[arxiv]](https://arxiv.org/abs/1706.10295) 
  6. A Distributional Perspective on Reinforcement Learning [[arxiv]](https://arxiv.org/pdf/1707.06887.pdf) 
  7. Rainbow: Combining Improvements in Deep Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1710.02298)
  8. Distributional Reinforcement Learning with Quantile Regression [[arxiv]](https://arxiv.org/pdf/1710.10044.pdf) 
  9. Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation  [[arxiv]](https://arxiv.org/abs/1604.06057)
  10. Neural Episodic Control [[arxiv]](https://arxiv.org/pdf/1703.01988.pdf) 

# Results for Pong using Double DQN
<img src="/assets/dqn_pong_results.png" width="600">


# Environments (All Atari Environments)
1. Breakout 
2. Pong
3. Montezuma's Revenge (Current Research)
4. Pitfall
5. Gravitar
6. CarRacing


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

# Future Plans
Plans to support:
1. Runtime differences between C++ and Python.
2. Python bindings for the Trainer module.
3. More models and methods.
4. Support for mujoco environment.

Stay tuned !



