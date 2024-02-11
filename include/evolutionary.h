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

class Individual {
    Team team;
public:
    int id;
    // wrapper around a team to keep it in the population
    Individual(const std::string& filename, int id);
    // evaluate a team by simulating it and adding the rewards
    std::vector<int> evaluate(const std::string& filename, std::vector<Environment> environments);
};

class Evolutionary {
    int numberOfGenerations;
    int numberOfEpisodes;
    int populationSize;
public:
    std::vector<Individual> population;
    Evolutionary(const std::string& filename);
    std::vector<Environment> generateTestEnvironments
        (const std::string& filename);
    void evolve(const std::string& filename);
};


#endif // EVOLUTIONARY_H
