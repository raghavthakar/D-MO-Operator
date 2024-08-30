#include "NSGA_II.h"
#include "evolutionary_utils.h"
#include "environment.h"
#include "policy.h"
#include "team.h"
#include <yaml-cpp/yaml.h>
#include <limits>
#include <iostream>
#include <algorithm>
#include <execution>
#include <unordered_set>
#include <thread>
#include <fstream>
#include <cassert>

const int NONE = std::numeric_limits<int>::min();
const int MAX  = std::numeric_limits<int>::max();

NSGA_II::NSGA_II(const std::string& filename) {
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
void NSGA_II::evolve(const std::string& filename, const std::string& data_filename) {
    EvolutionaryUtils evoHelper;

    std::vector<Environment> envs = evoHelper.generateTestEnvironments(filename);

    // Compute the origin for the hypervolume computation
    YAML::Node config = YAML::LoadFile(filename);
    const int lowerBound = config["team"]["numberOfAgents"].as<int>()
                                    * config["episode"]["length"].as<int>()
                                    * config["MOREPDomain"]["penalty"].as<int>()
                                    * config["evolutionary"]["numberOfEpisodes"].as<int>() - 1;

    // How many generations to do this for?
    const int numberOfGenerations = config["evolutionary"]["numberOfGenerations"].as<int>();

    // How many offsprings does the generation create?
    const int numberOfOffsprings = config["evolutionary"]["numberOfOffsprings"].as<int>();

    // Log info about the algo every how many gens?
    const int genLogInterval = config["experiment"]["genLogInterval"].as<int>();

    // -----------NSGA Base Case--------------
    
    for (int gen = 0; gen < numberOfGenerations; gen++) {
        // std::cout<<"Generation: "<<gen<<std::endl;
        std::for_each(std::execution::par, population.begin(), population.end(), [&](Individual& ind) {
            if (ind.fitness[0] == NONE) {
                ind.evaluate(filename, envs);
            }
        });

        std::vector<std::vector<Individual>> paretoFronts; // Better PFs first

        // 1. Transfer all solutions from the population into as many pareto fronts as needed
        while (population.size() > 0) {
            std::vector<Individual> innerPF = evoHelper.findParetoFront(population);
            paretoFronts.push_back(innerPF);
            population = evoHelper.without(population, innerPF); // remove the newest pareto front from population
        }
        // population is now empty, all individuals are in pareto fronts

        // 2. assign the nondomination level to each individual in the population based on its pareto front
        u_int pfNum = 0; 
        for (auto& pf : paretoFronts) {
            pfNum++;
            for (auto& pfInd : pf) {
                pfInd.nondominationLevel = pfNum;
            }
        }

        // 3. assign the crowding distance metric to each individual in the pareto fronts
        for (int p = 0; p < paretoFronts.size(); p++) {
            // i: set crowding distance to 0 for each individual
            for (auto& pfInd : paretoFronts[p]) {
                pfInd.crowdingDistance = 0;
            }
            /*std::cout<<"Crowding Distances: ";
            for (auto x:paretoFronts[p]) {
                std::cout<<x.crowdingDistance<<",";
            }
            std::cout<<std::endl;*/
            
            // ii: for each objective function
            for (int i = 0; i < paretoFronts[p][0].fitness.size(); i++) {
                // print out the individual id's before and after sorting
                // std::cout<<"PF before sorting: ";
                // for (auto x:paretoFronts[p]) {
                //     std::cout<<x.id<<",";
                // } 
                // (a): sort based on objective i (ascending)
                std::sort(paretoFronts[p].begin(), paretoFronts[p].end(), [i](const Individual& ind1, const Individual& ind2) {
                    return ind1.fitness[i] < ind2.fitness[i];
                });

                /*std::cout<<"PF after sorting: ";
                for (auto x:paretoFronts[p]) {
                    std::cout<<x.id<<",";
                }

                std::cout<<"Fitnesses: ";
                for (auto x:paretoFronts[p]) {
                    std::cout<<x.fitness[i]<<",";
                }
                std::cout<<std::endl;*/

                // (b): assign infinite crowding distance to the boundary elements (first and last)
                paretoFronts[p].front().crowdingDistance = MAX;
                paretoFronts[p].back() .crowdingDistance = MAX;

                int maxObjDelta = abs(paretoFronts[p].front().fitness[i] - paretoFronts[p].back().fitness[i]); // the difference in objective values between max and min

                // (c): crowding distance update => dk = dk + (f(k+1) - f(k-1))/maxObjDelta (all absolute values);
                for (int j = 1; j < paretoFronts[p].size() - 1; j++) {
                    if (maxObjDelta > 0)
                        paretoFronts[p][j].crowdingDistance += abs(((double)(paretoFronts[p][j-1].fitness[i]) - (double)(paretoFronts[p][j+1].fitness[i])) / (double) maxObjDelta);
                }
            }

            // iii: sort the pareto front based on crowding distance (descending)
            std::sort(paretoFronts[p].begin(), paretoFronts[p].end(), [](const Individual& ind1, const Individual& ind2) {
                    return ind1.crowdingDistance > ind2.crowdingDistance;
                });

            // std::cout<<"Crowding Distances: ";
            // for (auto x:paretoFronts[p]) {
            //     std::cout<<x.crowdingDistance<<",";
            // }
            // std::cout<<std::endl;
        }

        // 4. transfer the top populationSize agents from the sorted pareto fronts back to the empty population
        int pfSize = 0;
        for (auto pf : paretoFronts) {
            for (auto ind : pf) {
                if (pfSize < populationSize) {
                    population.push_back(ind);
                }
                pfSize++;
            }
        }

        // --------------------DATA LOGGING------------------------
        // initialise an empty data dict with just the keys (used for logging data)
        DataArranger dataHelper(data_filename);
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
            dataHelper.write();
        }
        // --------------------------------------------------------

        // 5. offsprings
        for (int i = 0; i < numberOfOffsprings; i++) { // generate as many offsprings as the current size of the population
            // a) select parents using binary tournament
            Individual parent1 = evoHelper.binaryTournament(paretoFronts, populationSize);
            Individual parent2 = evoHelper.binaryTournament(paretoFronts, populationSize);

            // b) crossover such that half of offspring comes from parent1, the other half from parent2
            // generate a random list of indices = 1/2 of size of offspring and supply those agents from parent1
            // supply the others from parent2
            std::vector<Agent> offspringAgents = evoHelper.crossover(parent1, parent2);
            population.push_back(Individual(filename, teamIDCounter++, offspringAgents));
        }
        
    }
}
