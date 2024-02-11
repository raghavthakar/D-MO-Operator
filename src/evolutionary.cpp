#include "evolutionary.h"
#include "environment.h"
#include "policy.h"
#include "team.h"
#include <vector>
#include <yaml-cpp/yaml.h>

Individual::Individual(const std::string& filename, int id) : 
    team(filename, id), id(id){}

std::vector<int> Individual::evaluate(const std::string& filename, std::vector<Environment> environments) {
    std::vector<std::vector<int>> stepwiseEpisodeReward; // Reward vector from each step of an episode
    std::vector<int> cumulativeEpisodeReward; // Sum of stewise rewards of an episode
    std::vector<std::vector<int>> cumulativeRewardsFromEachEpisode; // List of the cumulative episode rewards
    std::vector<int> combinedCumulativeRewards; // Sum of the cumulative rewards
    
    for (Environment env : environments) {
        stepwiseEpisodeReward = team.simulate(filename, env);
        cumulativeEpisodeReward = std::vector<int>(stepwiseEpisodeReward[0].size(), 0);

        for (auto& episodeReward : stepwiseEpisodeReward) {
            for (size_t i=0; i< episodeReward.size(); i++) {
                cumulativeEpisodeReward[i] += episodeReward[i];
            }
        }
        // tag the episode reward at the end of lsit
        cumulativeRewardsFromEachEpisode.push_back(cumulativeEpisodeReward);
    }

    combinedCumulativeRewards = std::vector<int>(cumulativeRewardsFromEachEpisode[0].size(), 0);
    for (auto& cumulativeReward : cumulativeRewardsFromEachEpisode) {
        for (size_t i=0; i< cumulativeReward.size(); i++) {
            combinedCumulativeRewards[i] += cumulativeReward[i];
        }
    }

    return combinedCumulativeRewards;
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
        if (randomPOIs) {
            env.reset();
            env.loadConfig(filename);
        }
        testEnvironments.push_back(env);
    }

    std::cout<<"''''''''''''''\nGenerated "<<numberOfEnvironments<<" environments.";
    for (auto env : testEnvironments) {
        std::cout<<"\nEnv:\n";
        env.printInfo();
    }
    std::cout<<"'''''''''''''\n";

    return testEnvironments;
}

// Actually run the simulation across teams and evolve them
void Evolutionary::evolve(const std::string& filename) {
    std::vector<Environment> envs = generateTestEnvironments(filename);

    // Now we have a lsit of environments, one environment for each episode
    // Each individual needs to be simualted in each environment (AKA each episode)
    for (auto individual : population) {
        std::cout<<"Individual "<<individual.id<<"'s reward: "<<individual.evaluate(filename, envs)<<std::endl; 
    }
}
