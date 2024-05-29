#include "MOD.h"
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


MOD::MOD(const std::string& filename) {
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

// Actually run the simulation across teams and evolve them
void MOD::evolve(const std::string& filename) {
    EvolutionaryUtils evoHelper;

    std::vector<Environment> envs = evoHelper.generateTestEnvironments(filename);

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
    
    // initialise an empty data dict with just the keys (used for logging data)
    DataArranger dataHelper;

    for (int gen = 0; gen < numberOfGenerations; gen++) {
        // parallelised this

        // std::cout<<"Generation: "<<gen<<std::endl;
        std::for_each(std::execution::par, population.begin(), population.end(), [&](Individual& ind) {
            if (ind.fitness[0] == NONE) {
                ind.evaluate(filename, envs);
            }
        });

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
            
            // 4. Create a team from these assembled joint policies and add it to the populatino
            Individual offspring = Individual(filename, this->teamIDCounter++, offSpringsAgents);
            offspring.mutate();
            this->population.push_back(offspring);
        }

        // --------------------DATA LOGGING------------------------
        if (gen + 1 < numberOfGenerations) { // dont log data for the last generation
        int numinds = population.size();
            for (auto ind : population) {
                dataHelper.clear();
                dataHelper.addData("gen", gen);
                dataHelper.addData("individual_id", ind.id);
                dataHelper.addData("fitness", ind.fitness);
                dataHelper.addData("difference_impacts", ind.differenceEvaluations);
                dataHelper.addData("nondomination_level", ind.nondominationLevel);
                dataHelper.addData("crowding_distance", ind.crowdingDistance);
                dataHelper.addData("trajectories", ind.getTeamTrajectoryAsString());
            }

            auto tempdat = dataHelper.get();
            for (const auto& pair : tempdat) {
                std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
            } 
        }
        // --------------------------------------------------------
    }
}

