#include <iostream>
// Move torch imports before ale because ale uses namespace std which interferes with the torch imports.
#include "/Users/navneetmadhukumar/Downloads/Arcade-Learning-Environment-master/src/ale_interface.hpp"

int main() {

    ALEInterface ale;
    ale.setInt("random_seed", 123);
    ale.loadROM("/Users/navneetmadhukumar/CLionProjects/Reinforcement_CPP/montezuma_revenge.bin");
    ActionVect legal_actions = ale.getLegalActionSet();

    float total_reward = 0.0;
    while (!ale.game_over()){
        Action a = legal_actions[rand() % legal_actions.size()];
        float reward = ale.act(a);
        total_reward += reward;
        std::cout<< "The episode ended with score " << total_reward << std::endl;
    }

    return 0;

}