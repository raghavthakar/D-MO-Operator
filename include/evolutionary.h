#ifndef EVOLUTIONARY_H
#define EVOLUTIONARY_H

#include "environment.h"
#include "policy.h"
#include "team.h"
#include <vector>
#include <string>
#include <utility>
#include <cmath>
#include <unordered_set>
#include <pagmo/utils/hypervolume.hpp>

class Individual {
    Team team;
public:
    int id;
    std::vector<int> fitness;
    // wrapper around a team to keep it in the population
    Individual(const std::string& filename, int id);
    // evaluate a team by simulating it and adding the rewards
    std::vector<int> evaluate(const std::string& filename, std::vector<Environment> environments);
};

class Evolutionary {
    int numberOfGenerations;
    int numberOfEpisodes;
    int populationSize;
    std::vector<Environment> generateTestEnvironments
        (const std::string& filename);
    double getHypervolume(std::vector<Individual> individuals, int hypervolumeOrigin); // computes the hypervolume of the given list of individuals
    bool dominates(Individual a, Individual b); // finds if the individual a dominates individual b
    std::vector<Individual> findParetoFront(const std::vector<Individual>& population); // finds and returns the pareto front in a population
public:
    std::vector<Individual> population;
    std::vector<Individual> paretoFront;
    Evolutionary(const std::string& filename);
    void evolve(const std::string& filename);
};


#endif // EVOLUTIONARY_H
