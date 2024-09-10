#ifndef MOREPBASEAGENT_H
#define MOREPBASEAGENT_H

#include "policy.h"
#include "environment.h"
#include <utility>
#include <vector>

class MOREPBaseAgent {
public:
    double posX; // X-coordinate of the MOREPBaseAgent's position
    double posY; // Y-coordinate of the MOREPBaseAgent's position
    double startingX; // starting coordinate
    double startingY; // starting coordinate
    double maxStepSize; // Maximum step size the MOREPBaseAgent can take
    double observationRadius; // How far can the MOREPBaseAgent see around it?
    int numberOfSensors; // How many sensors are around the MOREPBaseAgent
    double nnWeightMin; // Max and min weights of the policy weights
    double nnWeightMax;
    double noiseMean; // POlicy noise mutate's mean
    double noiseStdDev; // Policy noise mutate's stddev

    // Policy object that stores the MOREPBaseAgent's neural network
    Policy policy;

    // Constructor
    MOREPBaseAgent();
    MOREPBaseAgent(double x, double y, double startingX, double startingY, double maxStepSize, double observationRadius, int numberOfSensors, int numberOfClassIds, double nnWeightMin, double nnWeightMax, double noiseMean, double noiseStdDev);
    MOREPBaseAgent(const MOREPBaseAgent& other);

    // Function to move the MOREPBaseAgent by dx, dy (within maximum step size)
    void move(std::pair<double, double> delta, Environment environment);

    // Function to set the MOREPBaseAgent at the starting position and clear its observations
    void reset();

    // Observe and create state vector
    std::vector<double> observe(Environment environment, std::vector<std::pair<double, double>> agentPositions);

    // Function to get the current position of the MOREPBaseAgent
    std::pair<double, double> getPosition() const;

    // forward pass through the policy
    std::pair<double, double> forward(const std::vector<double>& input);

    // Function to get the maxStepSize of the MOREPBaseAgent
    int getMaxStepSize() const;

    // Add noise to the contained policy
    void addNoiseToPolicy();
};

#endif