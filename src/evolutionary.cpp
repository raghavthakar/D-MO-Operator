#include "evolutionary.h"
#include "evolutionary_utils.h"
#include "environment.h"
#include "policy.h"
#include "team.h"
#include <vector>
#include <random>
#include <yaml-cpp/yaml.h>
#include <pagmo/utils/hypervolume.hpp>
#include <pagmo/types.hpp>
#include <limits>
#include <iostream>
#include <algorithm>
#include <execution>
#include <unordered_set>
#include <thread>
#include <fstream>

const int NONE = std::numeric_limits<int>::min();

Individual::Individual(const std::string& filename, int id) : team(filename, id), id(id) {
    YAML::Node config = YAML::LoadFile(filename);

    // Initialise the fitness of the individual as NONE
    int numberOfObjectives = config["environment"]["numberOfClassIds"].as<int>();
    for(int i = 0; i < numberOfObjectives; i++) {
        fitness.push_back(NONE);
    }
}

Individual::Individual(const std::string& filename, int id, std::vector<Agent> agents) : team(filename, agents, id), id(id) {
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
        env.reset();
        env.loadConfig(filename);
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
    this->fitness = combinedCumulativeRewards;
    return combinedCumulativeRewards;
}

// compute and update the difference evaluations member variable
void Individual::differenceEvaluate(const std::string& filename, std::vector<Environment> environments, std::vector<Individual> paretoFront, int paretoIndex, double hypervolume, double lowerBound) {
    YAML::Node config = YAML::LoadFile(filename);
    const YAML::Node& evolutionary_config = config["evolutionary"];
    std::string counterfactualType = evolutionary_config["counterfactualType"].as<std::string>();
    
    EvolutionaryUtils evoHelper;
    
    // 1. get and sum the replay rewards for each environment in environments
    std::vector<std::vector<int>> cumulativeReplayRewards;
    for (auto environ : environments) {
        if (cumulativeReplayRewards.size() == 0) {
            cumulativeReplayRewards = team.replayWithCounterfactual(filename, environ, counterfactualType);
        } else {
            std::vector<std::vector<int>> replayRewards = team.replayWithCounterfactual(filename, environ, counterfactualType);
            for (int i = 0; i < cumulativeReplayRewards.size(); i++) { // for all agents in the team
                for (int rewNumber = 0; rewNumber < cumulativeReplayRewards[i].size(); rewNumber++) { // add up the counterfactual rewards
                    cumulativeReplayRewards[i][rewNumber] += replayRewards[i][rewNumber];
                }
            }
        }
    }

    // 2. Get pareto front hypervolume with these rewards swapped in for the original agent rewards
    std::vector<std::vector<int>> paretoFitnesses; // temporary front to deal with new hypervolume computations for each agent
    for (int i=0; i<paretoFront.size(); i++) { // populate working pareto front with all but this individual
        if (i == paretoIndex) continue;
        else {
            paretoFitnesses.push_back(paretoFront[i].fitness);
        }
    }

    // if (team.agents.size() != 10) {
    //     std::cout<<"Invalid team size";
    //     exit(1);
    // }

    for (int i=0; i<this->team.agents.size(); i++) { // add each counterfactual fitness to the working pareto front
        paretoFitnesses.push_back(cumulativeReplayRewards[i]);
        double counterfactualHypervolume = evoHelper.getHypervolume(paretoFitnesses, lowerBound); // get the hypervolume with this counterfactual fitness inserted
        double differenceImpact = hypervolume - counterfactualHypervolume; // find the difference with actual pareto hypervolume
        this->differenceEvaluations.push_back (differenceImpact); // assign the difference impact to the agent
        paretoFitnesses.pop_back();// delete the last (counterfactual) fitness from the pareto fitnesses
    }
    // std::cout<<team.agents.size()<<std::endl;
}

// mutate the agents' policies with some noise
void Individual::mutate() {
    this->team.mutate();
}

// Return the team's agents
std::vector<Agent> Individual::getAgents() {
    return this->team.agents;
}

// return the individual's team trajectory
std::string Individual::getTeamTrajectoryAsString() {
    std::stringstream output;
    
    // return the trajectory transpose (so each row is one agent's trajectory now)
    for (int i=0; i<this->team.teamTrajectory[0].size(); i++) {
        for (int j=0; j < team.teamTrajectory.size(); j++) {
            output << team.teamTrajectory[j][i].first << "," << team.teamTrajectory[j][i].second << " ";
        }
        output << std::endl;
    }

    return output.str();
}

Evolutionary::Evolutionary(const std::string& filename) {
    YAML::Node config = YAML::LoadFile(filename);
    const YAML::Node& evolutionary_config = config["evolutionary"];

    numberOfGenerations = evolutionary_config["numberOfGenerations"].as<int>();
    numberOfEpisodes = evolutionary_config["numberOfEpisodes"].as<int>();
    populationSize = evolutionary_config["populationSize"].as<int>();
    teamIDCounter = 0;

    for (int i=0; i < populationSize; i++) {
        population.push_back(Individual(filename, teamIDCounter++)); // Create a population of individuals with id
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

    // std::cout<<"''''''''''''''\nGenerated "<<numberOfEnvironments<<" environments.";
    // for (auto env : testEnvironments) {
    //     std::cout<<"\nEnv:\n";
    //     env.printInfo();
    // }
    // std::cout<<"'''''''''''''\n";

    return testEnvironments;
}


// Actually run the simulation across teams and evolve them
void Evolutionary::evolve(const std::string& filename, const std::string& data_fileneme) {
    std::vector<Environment> envs = generateTestEnvironments(filename);

    EvolutionaryUtils evoHelper;

    // Compute the origin for the hypervolume computation
    YAML::Node config = YAML::LoadFile(filename);
    const int lowerBound = config["team"]["numberOfAgents"].as<int>()
                                    * config["episode"]["length"].as<int>()
                                    * config["environment"]["penalty"].as<int>()
                                    * config["evolutionary"]["numberOfEpisodes"].as<int>() - 1;
    
    // How many offsprings does the generation create?
    const int numberOfOffsprings = config["evolutionary"]["numberOfOffsprings"].as<int>();
    
    // How many parents are selected to make these offsprings
    const int numberOfParents = config["evolutionary"]["numberOfParents"].as<int>();

    // how many generations to do this for?
    const int numberOfGenerations = config["evolutionary"]["numberOfGenerations"].as<int>();

    // log info about the algo every how many gens?
    const int genLogInterval = config["experiment"]["genLogInterval"].as<int>();
    
    // std::cout<<"Hypervolume origin is: "<<lowerBound<<std::endl;

    for (int gen = 0; gen < numberOfGenerations; gen++) {
        // parallelised this

        // std::cout<<"Generation: "<<gen<<std::endl;
        std::for_each(std::execution::par, population.begin(), population.end(), [&](Individual& ind) {
            ind.evaluate(filename, envs);
            // for (auto f:ind.fitness) {
            //     std::cout<<f<<",";
            // }
            // std::cout<<std::endl;
        });

        // for (auto &ind : population) {
        //     ind.evaluate(filename, envs);
        // }
        // std::cout<<"Evaluation complete"<<std::endl;

        std::vector<std::vector<Individual>> paretoFronts; // Better PFs first

        // 1. Get at least 'numberOfOffsprings' solutions from the population into as many pareto fronts as needed
        std::vector<Individual> workingPopulation = this->population; // temporary population variable to generate the pareto fronts

        while (workingPopulation.size() > this->population.size() - numberOfParents) {
            std::vector<Individual> innerPF = evoHelper.findParetoFront(workingPopulation);
            
            int numParetoInds = 0;
            for (auto pf : paretoFronts) {
                numParetoInds += pf.size();
            }

            if (numParetoInds + innerPF.size() > numberOfParents) {
                // cull the inner PF to only as many individuals as possible to maintain population size
                innerPF = evoHelper.cull(innerPF, numberOfParents - numParetoInds); // cull by numberofparents - numparetoinds
            }
            paretoFronts.push_back(innerPF);
            workingPopulation = evoHelper.without(workingPopulation, innerPF); // remove the newest pareto front from working population
        }
        // std::cout<<"working population set"<<std::endl;

        // remove the non-pareto solutions from the population
        this->population = evoHelper.without(this->population, workingPopulation);
        
        // std::cout<<"this population set: "<<this->population.size()<<std::endl;

        // 2. Update agent-level difference impact/reward for each solution on the above pareto fronts
        // #pragma omp parallel for
        for (int i = 0; i < paretoFronts.size(); ++i) {
            double paretoHypervolume = evoHelper.getHypervolume(paretoFronts[i], lowerBound);
            for (int j = 0; j < paretoFronts[i].size(); ++j) {
                paretoFronts[i][j].differenceEvaluate(filename, envs, paretoFronts[i], j, paretoHypervolume, lowerBound);
            }
        }
        // std::cout<<"difference evaluations complete"<<std::endl;

        // 3. Each individual on each pareto front now has an updated difference evaluation
        // Assemble new joint policies from these individuals

        // create a matrix of difference-evaluations (columns) vs individuals
        std::vector<std::vector<double>> differenceImpactsMatrix;
        for (int i=0; i<paretoFronts.size(); i++) {
            for (int j=0; j<paretoFronts[i].size(); j++) {
                differenceImpactsMatrix.push_back(paretoFronts[i][j].differenceEvaluations);
            }
        }
        // std::cout<<"difference impact matrix formed"<<std::endl;
        
        // required number of new agents are added
        for (int newIndividualNum = 0; newIndividualNum < numberOfOffsprings; newIndividualNum++) {
            std::vector<Agent> offSpringsAgents;

            for (int agentIndex=0; agentIndex<differenceImpactsMatrix[0].size(); agentIndex++) {
                std::vector<double> selectionProbabilities = evoHelper.getColumn(differenceImpactsMatrix, agentIndex);
                int selectedIndIndex = evoHelper.softmaxSelection(selectionProbabilities); // get an index of the selected individual for thatagent's policy
                // std::cout<<"Selected index is: "<<selectedIndIndex<<std::endl;
                // find this individual on the pareto front
                int indexCounter = 0;
                for (int i=0; i<paretoFronts.size(); i++) {
                    for (int ii=0; ii<paretoFronts[i].size(); ii++) {
                        if (indexCounter == selectedIndIndex) {
                            auto selectedAgent = paretoFronts[i][ii].getAgents()[agentIndex];
                            offSpringsAgents.push_back(selectedAgent); // add this agent to the offspring agents
                        }
                        indexCounter++;
                    }
                }
            }
            
            // TODO: MUTATION IS AFFECTING PARENT INDIVIDUAL'S AGENTS!!!!! NOT GOOD!!!!
            // 4. Create a team from these assembled joint policies and add it to the populatino
            Individual offspring = Individual(filename, this->teamIDCounter++, offSpringsAgents);
            offspring.mutate();
            this->population.push_back(offspring);
        }
        
        std::fstream dataFile;
        dataFile.open(data_fileneme, std::ios::app);
        
        dataFile << "Generation: " << gen << std::endl;
        for (auto pf : paretoFronts) {
            for (auto ind : pf) {
                dataFile << "Individual " <<ind.id <<" fitness: ";
                for (auto f : ind.fitness) {
                    dataFile << f <<",";
                }
                dataFile << std::endl;

                dataFile << "Individual " << ind.id << " difference impacts: ";
                for (double diffImpact : ind.differenceEvaluations) {
                    dataFile << diffImpact << ",";
                }
                dataFile << std::endl;

                if (gen % genLogInterval == 0) {
                    dataFile << "Individual's " << ind.id << " team trajectory: "<<std::endl;
                    dataFile << ind.getTeamTrajectoryAsString();
                }
            }
        }
        dataFile << std::endl;
    }
}

