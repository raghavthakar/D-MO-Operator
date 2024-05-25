#ifndef EVOLUTIONARYUTILS_H
#define EVOLUTIONARYUTILS_H

#include "policy.h"
#include "team.h"
#include "environment.h"
#include <unordered_map>
#include <any>

class Individual;
class EvolutionaryUtils;

class Individual {
    Team team;
public:
    int id;
    std::vector<int> fitness;
    std::vector<double> differenceEvaluations;
    u_int nondominationLevel;
    double crowdingDistance;
    // wrapper around a team to keep it in the population
    Individual(const std::string& filename, int id);
    Individual(const std::string& filename, int id, std::vector<Agent> agents);
    // evaluate a team by simulating it and adding the rewards
    std::vector<int> evaluate(const std::string& filename, std::vector<Environment> environments);
    // difference-evaluate the team and update agent-level difference reward
    void differenceEvaluate(const std::string& filename, std::vector<Environment> environments, std::vector<Individual> paretoFront, int paretoIndex, double hypervolume, double lowerBound);
    // return the agents of the team
    std::vector<Agent> getAgents();
    // return the team of the individual
    std::string getTeamTrajectoryAsString();
    // mutate the agents' policies with some noise
    void mutate();
};

class EvolutionaryUtils {
    int x;
public:
    EvolutionaryUtils();
    std::vector<Environment> generateTestEnvironments(const std::string& filename);
    Individual binaryTournament(std::vector<std::vector<Individual>> paretoFronts, size_t pSize);
    std::vector<Agent> crossover(Individual parent1, Individual parent2);
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

class DataArranger {
    std::unordered_map<std::string, std::string> _data;
public:
    DataArranger();
    void addData(std::string key, double data_);
    void addData(std::string key, std::vector<double> data_);
    void addData(std::string key, std::vector<int> data_);
    void addData(std::string key, std::string data_);
    void clear();
    std::unordered_map<std::string, std::string> get();
};

#endif // EVOLUTIONARY_H