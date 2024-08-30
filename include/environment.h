#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "MOREPDomain.h"
#include <vector>
#include <string>
#include <utility>
#include <cmath>
#include <unordered_set>

// Simulation environment class definition
class Environment {
    MOREPDomain rover_env;
public:
    // Method to load configuration from YAML file
    void loadConfig(const std::string& filename);
    std::vector<int> getRewards(std::vector<std::pair<double, double>> agentPositions, int stepNumber);
    std::pair<double, double> moveAgent(std::pair<double, double> currentPos, std::pair<double, double> delta, double maxStepSize);
    // observations of an agent
    std::vector<double> getAgentObservations(std::pair<double, double> agentPos, int numberOfSensors, double observationRadius, std::vector<std::pair<double, double>> agentPositions);
    // Method to return the dimensions of the environment
    std::pair<int, int> getDimensions();
    void reset();
};

#endif // ENVIRONMENT_H
