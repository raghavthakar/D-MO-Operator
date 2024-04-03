#ifndef EVOLUTIONARYUTILS_H
#define EVOLUTIONARYUTILS_H

#include "environment.h"
#include "policy.h"
#include "team.h"

class EvolutionaryUtils {
    int x;
public:
    EvolutionaryUtils();
    double getHypervolume(std::vector<Individual> individuals, int hypervolumeOrigin); // computes the hypervolume of the given list of individuals
    double getHypervolume(std::vector<std::vector<int>> individualFitnesses, int lowerBound); // computes the hypervolume of the given list of individuals
    bool dominates(Individual a, Individual b); // finds if the individual a dominates individual b
    std::vector<Individual> findParetoFront(const std::vector<Individual>& population); // finds and returns the pareto front in a population
    std::vector<Individual> without(const std::vector<Individual> workingPopulation, const std::vector<Individual> toRemoveSolutions);
    std::vector<Individual> cull(const std::vector<Individual> PF, const int desiredSize);
    std::vector<double> softmax(const std::vector<double>& values);
    int softmaxSelection(std::vector<double> probabilities);
    std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> matrix);
    std::vector<double> getColumn(std::vector<std::vector<double>> matrix, int colNum);

};

#endif // EVOLUTIONARY_H