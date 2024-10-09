#include "DNSGA.h"
#include "evolutionary_utils.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <list>
#include "team.h"
#include <yaml-cpp/yaml.h>
#include <pagmo/utils/multi_objective.hpp>


DNSGA::DNSGA(int d) {
    int x = 1;
}


void DNSGA::evolve(const std::string& filename, const std::string& data_filename) {
    YAML::Node config = YAML::LoadFile(filename);

    // load up the evolutionary parameters
    int numberOfGenerations = config["evolutionary"]["numberOfGenerations"].as<int>();
    int popsize = config["evolutionary"]["populationSize"].as<int>();

    // Generate some environments for testing the agents
    EvolutionaryUtils evoHelper;
    std::vector<Environment> envs = evoHelper.generateTestEnvironments(filename);

    // Initialize the populations
    std::vector<std::vector<Agent>> populations;
    int numagents = 0;
    // create agent populations based on info from beach sections
    std::vector<YAML::Node> sections = config["MOBPDomain"]["Sections"].as<std::vector<YAML::Node>>();

    for (auto section : sections) {
        int numMale = section["maleTourists"].as<int>();
        int numFemale = section["femaleTourists"].as<int>();
        numagents += (numMale + numFemale);
        int section_id = section["id"].as<int>();

        for (int i = 0; i < numMale; i++) {
            std::vector<Agent> pop;
            for (int j = 0; j < popsize; j++) {
                pop.push_back(Agent(section_id, "male", section_id, filename));
            }
            populations.push_back(pop);
        }
        for (int i = 0; i < numFemale; i++) {
            std::vector<Agent> pop;
            for (int j = 0; j < popsize; j++) {
                pop.push_back(Agent(section_id, "female", section_id, filename));
            }
            populations.push_back(pop);
        }
    }

    for(int gen = 0; gen < numberOfGenerations; gen++) {
        // --------------------DATA LOGGING------------------------
        // initialise an empty data dict with just the keys (used for logging data)
        DataArranger dataHelper(data_filename);
        dataHelper.clear();
        dataHelper.addData("gen", gen);
        
        // --------------------------------------------------------
        // Evaluate each policy in each population
        for (int policy_idx = 0; policy_idx < popsize; policy_idx++) {
            // Form the team to be evaluated
            std::vector<Agent> evalagents;
            for (int pop_num = 0; pop_num < numagents; pop_num++) {
                evalagents.push_back(populations[pop_num][policy_idx]);
            }

            // Put the selected agents on a team to evaluate them
            Individual evalteamwrapper = Individual(filename, 0, evalagents);
            evalteamwrapper.evaluate(filename, envs);
            auto cfacrewards = evalteamwrapper.team.replayWithCounterfactual(filename, envs[0], "static");

            // store the generation data
            dataHelper.addData("fitness", evalteamwrapper.fitness);
            dataHelper.addData("trajectories", evalteamwrapper.getTeamTrajectoryAsString());
            dataHelper.write();

            // Compute the difference evaluation for each agent in the evaluated team
            for (int pop_num = 0; pop_num < numagents; pop_num++) {
                // Compute the difference evaluation (team reward - counterfactual reward)
                std::vector<double> dvalues(evalteamwrapper.fitness.size());
                for (size_t t = 0; t < dvalues.size(); ++t) {
                    dvalues[t] = evalteamwrapper.fitness[t] - cfacrewards[pop_num][t]; // Difference evaluation computation
                }

                // Assign the difference evaluation to the policy
                populations[pop_num][policy_idx].differenceEvaluation = dvalues;
            }
        }

        // Initialize random number generator
        std::random_device rd;  // Non-deterministic random number generator
        std::mt19937 rng(rd()); // Seed Mersenne Twister with the random device value

        // Sort each population based on dominance and crowding distance, then perform selection
        for (auto& population : populations) {
            size_t pop_size = population.size();
            std::vector<pagmo::vector_double> input_f; // To store evaluations

            // Collect evaluations from policies
            for (const auto& policy : population) {
                input_f.push_back(policy.differenceEvaluation); // Assuming differenceEvaluation is vector_double
            }

            // Apply multi-objective sorting
            std::vector<pagmo::pop_size_t> sorted_indices = pagmo::sort_population_mo(input_f);

            // Create a mapping from individual index to position in sorted_indices
            std::vector<size_t> positions(pop_size);
            for (size_t pos = 0; pos < sorted_indices.size(); ++pos) {
                size_t idx = sorted_indices[pos];
                positions[idx] = pos; // Lower position means better rank
            }

            // Prepare for selection
            std::vector<Agent> new_population;
            new_population.reserve(pop_size);

            std::uniform_int_distribution<size_t> dist(0, pop_size - 1);

            // Perform binary tournament selection with replacement
            while (new_population.size() < pop_size) {
                // Randomly select two individuals (with possible repeats)
                size_t idx1 = dist(rng);
                size_t idx2 = dist(rng);

                // Determine the winner based on rank (position in sorted_indices)
                size_t winner_idx;
                if (positions[idx1] > positions[idx2]) {
                    winner_idx = idx1;
                } else if (positions[idx1] < positions[idx2]) {
                    winner_idx = idx2;
                } else {
                    // Equal rank, select randomly
                    winner_idx = (rng() % 2 == 0) ? idx1 : idx2;
                }

                // Add the winner to the new population
                new_population.push_back(population[winner_idx]);
            }

            // Replace the old population with the new one
            population.clear();
            for (auto & agent : new_population) {
                agent.addNoiseToPolicy(); // mutate the policy
                population.push_back(agent);
            }
        }
    }
}