#include "Trainer.h"
#include <iostream>
// Move torch imports before ale because ale uses namespace std which interferes with the torch imports.
#include "/Users/navneetmadhukumar/Downloads/Arcade-Learning-Environment-master/src/ale_interface.hpp"


int main() {

    Trainer trainer(3, 18, 10000);
    trainer.train(123, "/Users/navneetmadhukumar/CLionProjects/Reinforcement_CPP/atari_roms/pong.bin", 1000);

}