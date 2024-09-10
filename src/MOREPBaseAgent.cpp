#include "MOREPBaseAgent.h"

MOREPBaseAgent::MOREPBaseAgent() {
    int x = 2;
}

MOREPBaseAgent::MOREPBaseAgent(double x, double y, double startingX_, double startingY_, double _maxStepSize, double _observationRadius, 
    int _numberOfSensors, int numberOfClassIds, double _nnWeightMin, double _nnWeightMax, double _noiseMean, double _noiseStdDev) : 
    posX(x), posY(y), startingX(startingX_), startingY(startingY_), maxStepSize(_maxStepSize), 
    observationRadius(_observationRadius), numberOfSensors(_numberOfSensors), noiseMean(_noiseMean),
    noiseStdDev(_noiseStdDev), policy(2 + _numberOfSensors * (numberOfClassIds) + _numberOfSensors, _nnWeightMin, _nnWeightMax)  {}

// copy constructor
MOREPBaseAgent::MOREPBaseAgent(const MOREPBaseAgent& other) : posX(other.posX), posY(other.posY), startingX(other.startingX), startingY(other.startingY), maxStepSize(other.maxStepSize), 
    observationRadius(other.observationRadius), numberOfSensors(other.numberOfSensors), nnWeightMin(other.nnWeightMin), 
    nnWeightMax(other.nnWeightMax), noiseMean(other.noiseMean), noiseStdDev(other.noiseStdDev) {
        this->policy = *std::make_shared<Policy>(other.policy);;
}

// Function to move the MOREPBaseAgent by dx, dy (within maximum step size)
void MOREPBaseAgent::move(std::pair<double, double> delta, Environment environment) {
    std::pair<double, double> newPosition = environment.moveAgent(std::make_pair(posX, posY), delta, this->maxStepSize);

    // Update the MOREPBaseAgent's position
    posX = newPosition.first;
    posY = newPosition.second;
}

// Function to set the MOREPBaseAgent at the starting position and clear its observations
void MOREPBaseAgent::reset() {
    posX = this->startingX;
    posY = this->startingY;
}

// forward pass through the policy
std::pair<double, double> MOREPBaseAgent::forward(const std::vector<double>& input) {
    return policy.forward(input);
}

// Adds noise to the contained policy
void MOREPBaseAgent::addNoiseToPolicy() {
    this->policy.addNoise(this->noiseMean, this->noiseStdDev);
}

// Observe and create state vector
// Assumes that POIs have classID 0, 1, 2....
std::vector<double> MOREPBaseAgent::observe(Environment environment, std::vector<std::pair<double, double>> agentPositions) {
    return environment.getAgentObservations(std::make_pair(posX, posY), this->numberOfSensors, this->observationRadius, agentPositions);
}

// Function to get the current position of the MOREPBaseAgent
std::pair<double, double> MOREPBaseAgent::getPosition() const {
    return std::make_pair(posX, posY);
}

int MOREPBaseAgent::getMaxStepSize() const {
    return maxStepSize;
}