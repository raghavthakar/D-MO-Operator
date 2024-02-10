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
    int id;
public:
    // wrapper around a team to keep it in the population
    Individual(const std::string& filename, int id);
    // evaluate a team by simulating it and adding the rewards
    void evaluate(const std::string& filename, Environment environment);
};

class Evolutionary {
    int numberOfGenerations;
    int numberOfEpisodes;
    int populationSize;
public:
    Evolutionary(const std::string& filename);
    std::vector<Individual> population;
};


#endif // EVOLUTIONARY_H
