#ifndef TEAM_H
#define TEAM_H

#include "environment.h"
#include "policy.h"
#include "MOREPBaseAgent.h"
#include <vector>
#include <string>
#include <cmath>
#include <utility>

class Agent {
public:
    MOREPBaseAgent rover;
    // Constructor
    Agent(const std::string& config_filename);
    Agent(const Agent& other);

    // Function to move the agent by dx, dy (within maximum step size)
    void move(std::vector<double> delta, Environment environment);

    // Function to set the agent at the starting position and clear its observations
    void reset();

    // Observe and create state vector
    std::vector<double> observe(Environment environment, std::vector<std::vector<double>> agentPositions);

    // forward pass through the policy
    std::vector<double> forward(const std::vector<double>& input);
    
    // Function to get the current position of the agent
    std::vector<double> getPosition() const;



    // Function to get the maxStepSize of the agent
    int getMaxStepSize() const;

    // Add noise to the contained policy
    void addNoiseToPolicy();
};

class Team {
public:
    std::vector<Agent> agents; // Vector to store agents in team
    int id;
    std::vector<std::vector<std::vector<double>>> teamTrajectory;
    Team();
    Team(const std::string& filename, int id); // Constructor
    Team(const std::string& filename, std::vector<Agent> agents, int id); // Constructor
    void printInfo();
    std::vector<std::vector<double>> simulate(const std::string& filename, Environment environment);
    std::vector<std::vector<double>> replayWithCounterfactual(const std::string& filename, Environment environment, const std::string& counterfactualType);
    void mutate();
};

#endif // TEAM_H
