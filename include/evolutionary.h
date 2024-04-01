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
#include <functional>
#include <pagmo/utils/hypervolume.hpp>

class Individual;
class EvolutionaryUtils;
class Evolutionary;

class Individual {
    Team team;
public:
    int id;
    std::vector<int> fitness;
    std::vector<double> differenceEvaluations;
    // wrapper around a team to keep it in the population
    Individual(const std::string& filename, int id);
    Individual(const std::string& filename, int id, std::vector<Agent> agents);
    // evaluate a team by simulating it and adding the rewards
    std::vector<int> evaluate(const std::string& filename, std::vector<Environment> environments);
    // difference-evaluate the team and update agent-level difference reward
    void differenceEvaluate(const std::string& filename, std::vector<Environment> environments, std::vector<Individual> paretoFront, int paretoIndex, double hypervolume, double lowerBound);
    // return the agents of the team
    std::vector<Agent> getAgents();
};

class Evolutionary {
    int numberOfGenerations;
    int numberOfEpisodes;
    int populationSize;
    unsigned int teamIDCounter; // Tracks the latest team ID
    std::vector<Environment> generateTestEnvironments
        (const std::string& filename);
public:
    std::vector<Individual> population;
    Evolutionary(const std::string& filename);
    void evolve(const std::string& filename);
};

class EvolutionaryUtils {
    int x;
public:
    EvolutionaryUtils();
    double getHypervolume(std::vector<Individual> individuals, int hypervolumeOrigin); // computes the hypervolume of the given list of individuals
    double getHypervolume(std::vector<std::vector<int>> individualFitnesses, int lowerBound); // computes the hypervolume of the given list of individuals
    bool dominates(Individual a, Individual b); // finds if the individual a dominates individual b
    std::vector<Individual> findParetoFront(const std::vector<Individual>& population); // finds and returns the pareto front in a population
    std::vector<Individual> without(const std::vector<Individual> workingPopulation, const std::vector<Individual> toRemoveSolutions);
    std::vector<double> softmax(const std::vector<double>& values);
    int softmaxSelection(std::vector<double> probabilities);
    std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> matrix);
    std::vector<double> getColumn(std::vector<std::vector<double>> matrix, int colNum);
};


#endif // EVOLUTIONARY_H
