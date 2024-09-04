#include "MOD_team_ablated.h"
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


MODTeamAblated::MODTeamAblated(const std::string& filename) {
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
void MODTeamAblated::evolve(const std::string& filename, const std::string& data_filename) {
    EvolutionaryUtils evoHelper(filename);

    std::vector<Environment> envs = evoHelper.generateTestEnvironments(filename);

    // Compute the origin for the hypervolume computation
    YAML::Node config = YAML::LoadFile(filename);
    const int lowerBound = config["team"]["numberOfAgents"].as<int>()
                                    * config["episode"]["length"].as<int>()
                                    * config["MOREPDomain"]["penalty"].as<int>()
                                    * config["evolutionary"]["numberOfEpisodes"].as<int>() - 1;
    
    // How many offsprings does the generation create?
    const int numberOfOffsprings = config["evolutionary"]["numberOfOffsprings"].as<int>();
    
    // How many parents are selected to make these offsprings
    const int numberOfParents = config["evolutionary"]["numberOfParents"].as<int>();

    // how many generations to do this for?
    const int numberOfGenerations = config["evolutionary"]["numberOfGenerations"].as<int>();

    // log info about the algo every how many gens?
    const int genLogInterval = config["experiment"]["genLogInterval"].as<int>();
    
    for (int gen = 0; gen < numberOfGenerations; gen++) {
        // parallelised this

        // std::cout<<"Generation: "<<gen<<std::endl;
        std::for_each(std::execution::par, population.begin(), population.end(), [&](Individual& ind) {
            if (ind.fitness[0] == NONE) {
                ind.evaluate(filename, envs);
            }
        });
        // std::cout<<"Done evalsuting\n";

        std::vector<std::vector<Individual>> paretoFronts; // Better PFs first

        // 1. Get at least populationsize solutions from the bloated population into as many pareto fronts as needed
        int numParetoInds = 0;
        while (numParetoInds < populationSize) {
            std::vector<Individual> innerPF = evoHelper.findParetoFront(population);

            if (numParetoInds + innerPF.size() > populationSize) {
                // cull the inner PF to only as many individuals as possible to maintain population size
                innerPF = evoHelper.cull(innerPF, populationSize - numParetoInds); // cull by populationSize - numparetoinds
            }
            
            numParetoInds += innerPF.size();

            paretoFronts.push_back(innerPF);
            population = evoHelper.without(population, innerPF); // remove the newest pareto front from working population
        }
        // std::cout<<"Done making pfs\n";

        // remove the non-pareto solutions from the population
        // this->population = evoHelper.without(this->population, workingPopulation);
        this->population.clear(); // empty out the popoulation
        // std::cout<<"Done clearing population\n";
        
        // std::cout<<"this population set: "<<this->population.size()<<std::endl;

        // 2. Update agent-level difference impact/reward for each solution on the above pareto fronts
        // #pragma omp parallel for
        for (int i = 0; i < paretoFronts.size(); ++i) {
            double paretoHypervolume = evoHelper.getHypervolume(paretoFronts[i], lowerBound);

            // copy all the agent fitnesses into a vector
            std::vector<std::vector<int>> paretoFitnesses;
            for (int j = 0; j < paretoFronts[i].size(); ++j) {
                paretoFitnesses.push_back(paretoFronts[i][j].fitness); // push the fitness into fitness vector
            }

            // sequentially set the fitness to origin (lowerbound) and recompute hyopervolume to find contribution
            for (int j = 0; j < paretoFitnesses.size(); j++) {
                auto paretoFitness = paretoFitnesses[j]; // store the fitness
                paretoFitnesses[j] = std::vector<int>(paretoFitnesses[j].size(), lowerBound + 1); // replace the fitness with lowerbound
                auto counterfactualHypervolume = evoHelper.getHypervolume(paretoFitnesses, lowerBound);
                paretoFitnesses[j] = paretoFitness; // swap back the original fitness
                auto differenceImpact = paretoHypervolume - counterfactualHypervolume;
                paretoFronts[i][j].differenceEvaluations = std::vector<double>(paretoFronts[i][j].getAgents().size(), differenceImpact); // update the agent-level difference impact
                population.push_back(paretoFronts[i][j]); // push the difference evaluated into the population
            }
        }
        // std::cout<<"difference evaluations complete"<<std::endl;

        // for (auto ind : population) {
        //     auto de = ind.differenceEvaluations;
        //     for (auto d: de)
        //         std::cout << d << ",";
        //     std::cout<<std::endl;
        //     std::cout<<ind.differenceEvaluations.size()<<std::endl;
        // }

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
            // std::cout<<"\tCreating ind "<<newIndividualNum<<"\n";
            std::vector<Agent> offSpringsAgents;

            for (int agentIndex=0; agentIndex<differenceImpactsMatrix[0].size(); agentIndex++) {
                std::vector<double> selectionProbabilities = evoHelper.getColumn(differenceImpactsMatrix, agentIndex);
                // std::cout<<"\t\tselection probs found\n";
                int selectedIndIndex = evoHelper.epsilonGreedySelection(selectionProbabilities); // get an index of the selected individual for thatagent's policy
                // std::cout<<"\t\tsoftmax found\n";
                // std::cout<<"Selected index is: "<<selectedIndIndex<<std::endl;
                // find this individual on the pareto front
                int indexCounter = 0;
                for (int i=0; i<paretoFronts.size(); i++) {
                    for (int ii=0; ii<paretoFronts[i].size(); ii++) {
                        if (indexCounter == selectedIndIndex) {
                            auto selectedAgent = paretoFronts[i][ii].getAgents()[agentIndex];
                            offSpringsAgents.push_back(selectedAgent); // add this agent to the offspring agents
                            // std::cout<<"\t\tagent pushed\n";
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
        // std::cout<<"Done offpsrings\n";
    }
}

