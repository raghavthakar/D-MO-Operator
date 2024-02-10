#include "evolutionary.h"
#include "environment.h"
#include "policy.h"
#include "team.h"
#include <vector>
#include <yaml-cpp/yaml.h>

Individual::Individual(const std::string& filename, int id) : 
    team(filename, id){}

void Individual::evaluate(const std::string& filename, Environment environment) {
    std::vector<std::vector<int>> ep_rew = team.simulate(filename, environment);

    for (auto e : ep_rew)
        std::cout<<e<<std::endl;
}

Evolutionary::Evolutionary(const std::string& filename) {
    YAML::Node config = YAML::LoadFile(filename);
    const YAML::Node& evolutionary_config = config["evolutionary"];

    numberOfGenerations = evolutionary_config["numberOfGenerations"].as<int>();
    numberOfEpisodes = evolutionary_config["numberOfEpisodes"].as<int>();
    populationSize = evolutionary_config["populationSize"].as<int>();

    for (int i=0; i < populationSize; i++) {
        population.push_back(Individual(filename, i)); // Create a population of individuals with id
    }
}

// DECIDE and Generate as many environment configurations as numberOfEpisodes
std::vector<Environment> Evolutionary::generateTestEnvironments
    (const std::string& filename) {
    YAML::Node config = YAML::LoadFile(filename);

    int numberOfEnvironments = numberOfEpisodes;
    bool randomPOIs = config["environment"]["randomPOIs"].as<bool>();

    std::vector<Environment> testEnvironments;
    
    Environment env;
    env.loadConfig(filename);
    for(int i = 0; i < numberOfEnvironments; i++) {
        if (randomPOIs) env.loadConfig(filename);
        testEnvironments.push_back(env);
    }

    return testEnvironments;
}

// Actually run the simulation across teams and evolve them
void Evolutionary::evolve(const std::string& filename) {
    std::vector<Environment> envs = generateTestEnvironments(filename);

    for(auto env : envs) {
        for (auto& ind : population) {
            ind.evaluate(filename, env);
        }
    }
}
