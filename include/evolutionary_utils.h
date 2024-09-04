#ifndef EVOLUTIONARYUTILS_H
#define EVOLUTIONARYUTILS_H

#include "policy.h"
#include "team.h"
#include "environment.h"
#include <unordered_map>
#include <any>
#include <string>

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
    double softmaxTemperature;
    double temperatureDecayFactor;
    double epsilon;
    double epsilonDecayFactor;
public:
    EvolutionaryUtils();
    EvolutionaryUtils(const std::string& config_filename);
    std::vector<Environment> generateTestEnvironments(const std::string& filename);
    Individual binaryTournament(std::vector<std::vector<Individual>> paretoFronts, size_t pSize);
    std::vector<Agent> crossover(Individual parent1, Individual parent2);
    double getHypervolume(std::vector<Individual> individuals, int hypervolumeOrigin); // computes the hypervolume of the given list of individuals
    double getHypervolume(std::vector<std::vector<int>> individualFitnesses, int lowerBound); // computes the hypervolume of the given list of individuals
    bool dominates(Individual a, Individual b); // finds if the individual a dominates individual b
    std::vector<Individual> findParetoFront(const std::vector<Individual>& population); // finds and returns the pareto front in a population
    std::vector<Individual> without(const std::vector<Individual> workingPopulation, const std::vector<Individual> toRemoveSolutions);
    std::vector<Individual> cull(const std::vector<Individual> PF, const int desiredSize);
    int softmaxSelection(std::vector<double> values);
    int rouletteWheelSelection(std::vector<double> values);
    int epsilonGreedySelection(std::vector<double> values);
    std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> matrix);
    std::vector<double> getColumn(std::vector<std::vector<double>> matrix, int colNum);
};

class DataArranger {
    std::map<std::string, std::string> _data;
    std::string _data_filename;
public:
    DataArranger(std::string data_filename_);
    void addData(std::string key, double data_);
    void addData(std::string key, std::vector<double> data_);
    void addData(std::string key, std::vector<int> data_);
    void addData(std::string key, std::string data_);
    void clear();
    std::map<std::string, std::string> get();
    void write();
};

#endif // EVOLUTIONARY_H