#include "evolutionary.h"
#include "environment.h"
#include "policy.h"
#include "team.h"
#include <vector>
#include <yaml-cpp/yaml.h>
#include <limits>

const int NONE = std::numeric_limits<int>::min();

Individual::Individual(const std::string& filename, int id) : team(filename, id), id(id) {
    YAML::Node config = YAML::LoadFile(filename);

    // Initialise the fitness of the individual as NONE
    int numberOfObjectives = config["environment"]["numberOfClassIds"].as<int>();
    for(int i = 0; i < numberOfObjectives; i++) {
        fitness.push_back(NONE);
    }
}

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

    // set the fitness of the individual
    fitness = combinedCumulativeRewards;
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
    bool differentEnvs = config["environment"]["differentEnvs"].as<bool>();

    std::vector<Environment> testEnvironments;
    
    Environment env;
    env.loadConfig(filename);
    for(int i = 0; i < numberOfEnvironments; i++) {
        // Load up a new env configuration if env should be different for each episode
        if (differentEnvs) {
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

// Compute the hypervolume contained by the given pareto front
double Evolutionary::getHypervolume(std::vector<Individual> individuals, int hypervolumeReference) {
    // get the hypervolume computation reference point from the origin
    std::vector<int> referencePoint(individuals[0].fitness.size(), hypervolumeReference);

    // Manual sorting of individuals based on fitness of 0th objective
    for (size_t i = 0; i < individuals.size(); ++i) {
        for (size_t j = i + 1; j < individuals.size(); ++j) {
            // Compare fitness values of adjacent individuals
            if (individuals[j].fitness[0] < individuals[i].fitness[0]) {
                // Swap individuals if necessary
                std::swap(individuals[i], individuals[j]);
            }
        }
    }

    double hypervolume = 0.0;

    // Calculate hypervolume contribution for each individual
    for (size_t i = 0; i < individuals.size(); ++i) {
        double volume = 1.0;
        for (size_t j = 0; j < individuals[i].fitness.size(); ++j) {
            volume *= referencePoint[j] - individuals[i].fitness[j];
        }
        hypervolume += volume;
    }

    return hypervolume;
}

// finds if the individual a dominates individual b
bool Evolutionary::dominates(Individual a, Individual b) {
    if (a.fitness.size() != b.fitness.size()) {
        std::cout<<"Cannot find dominating solution. Imbalanced fitnesses";
        exit(1);
    }
    else if (a.fitness[0] == NONE || b.fitness[0] == NONE) {
        std::cout<<"Cannot find dominating solution. NONE fitnesses";
        exit(1);
    }

    for (int i = 0; i < a.fitness.size(); i++) {
        if (a.fitness[i] < b.fitness[i])
            return false;
    }

    return true;
}

// Find and return the pareto front of the given population
std::vector<Individual> Evolutionary::findParetoFront(const std::vector<Individual>& population) {
    std::vector<Individual> paretoFront;

    for (const Individual& individual : population) {
        bool isNonDominated = true;

        // Check if the individual is non-dominated by comparing its fitness with others
        for (const Individual& other : population) {
            if (&individual != &other) { // Skip self-comparison
            // if other dominates individual, then individual should not be on pareto front
                if (dominates(other, individual)) {
                    isNonDominated = false;
                    break;
                }
            }
        }

        // If the individual is non-dominated, add it to the Pareto front
        if (isNonDominated) {
            paretoFront.push_back(individual);
        }
    }

    return paretoFront;
}

// Actually run the simulation across teams and evolve them
void Evolutionary::evolve(const std::string& filename) {
    std::vector<Environment> envs = generateTestEnvironments(filename);

    // Compute the origin for the hypervolume computation
    YAML::Node config = YAML::LoadFile(filename);
    const int hypervolumeReference = config["team"]["numberOfAgents"].as<int>()
                                    * config["episode"]["length"].as<int>()
                                    * config["environment"]["penalty"].as<int>();
    
    std::cout<<"Hypervolume origin is: "<<hypervolumeReference<<std::endl;

    // Now we have a lsit of environments, one environment for each episode
    // Each individual needs to be simualted in each environment (AKA each episode)
    for (auto& individual : population) {
        individual.evaluate(filename, envs);
        std::cout<<"Individual "<<individual.id<<"'s fitness: "<<individual.fitness<<std::endl; 
    }

    std::cout<<"The hypervolume is: "<<getHypervolume(population, hypervolumeReference)<<std::endl;

    std::vector<Individual> pf = findParetoFront(population);

    std::cout<<"PF:\n";
    for(auto cc: pf)
        std::cout<<cc.id<<std::endl;
}
