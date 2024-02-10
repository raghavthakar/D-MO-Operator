#include "environment.h"
#include "team.h"

int main() {
    Environment env;
    env.loadConfig("../config/config.yaml"); // Load configuration from YAML file
    env.printInfo(); // Print information about loaded POIs

    Team team("../config/config.yaml", 0);
    team.printInfo();

    // for (auto i=0; i<5; i++) {
    //     team.agents[2].move(5, 5, env);
    //     team.printInfo();

    // }
    team.simulate("../config/config.yaml", env);
    team.printInfo();

    return 0;
}
