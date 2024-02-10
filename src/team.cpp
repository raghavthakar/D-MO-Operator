#include "team.h"
#include "environment.h"
#include "policy.h"
#include <vector>
#include <unordered_set>
#include <string>
#include <iostream>
#include <cmath>
#include <yaml-cpp/yaml.h>

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// Constructor
Agent::Agent(double x, double y, double _maxStepSize, double _observationRadius, 
    int _numberOfSensors, int numberOfClassIds) : posX(x), posY(y), maxStepSize(_maxStepSize), 
    observationRadius(_observationRadius), numberOfSensors(_numberOfSensors),
    policy(2 + _numberOfSensors * (numberOfClassIds + 1), -2, 2)  {}

// Function to move the agent by dx, dy (within maximum step size)
void Agent::move(std::pair<double, double> delta, Environment environment) {
    int environmentXLength = environment.getDimensions().first;
    int environmentYLength = environment.getDimensions().second;

    double dx = delta.first;
    double dy = delta.second;

    // Calculate the distance to move
    double distance = sqrt(dx * dx + dy * dy);

    // Check if the distance is within the maximum step size
    if (distance > maxStepSize) {
        // Scale down the movement to stay within the maximum step size
        double scaleFactor = maxStepSize / distance;
        dx *= scaleFactor;
        dy *= scaleFactor;
    }

    // Calculate the new position within environment limits
    double step_slope = dy / dx;

    if (posX + dx > environmentXLength){
        dx = environmentXLength - posX;
        dy = dx * step_slope;
        step_slope = dy / dx;;
    } else if (posX + dx < 0) {
        dx = -posX;
        dy = dx * step_slope;
        step_slope = dy / dx;;
    }

    if (posY + dy > environmentYLength){
        dy = environmentYLength - posY;
        dx = dy /step_slope;
        step_slope = dy / dx;;
    } else if (posY + dy < 0) {
        dy = -posY;
        dx = dy / step_slope;
        step_slope = dy / dx;;
    }

    // Update the agent's position
    posX += dx;
    posY += dy;
}

// Function to set the agent at the starting position and clear its observations
void Agent::set(int startingX, int startingY) {
    posX = startingX;
    posY = startingY;
}

// Observe and create state vector
// Assumes that POIs have classID 0, 1, 2....
std::vector<double> Agent::observe(Environment environment, std::vector<Agent> agents) {
    std::vector<double> observations; // To store the observations the agent makes

    observations.push_back(posX);
    observations.push_back(posY);

    // Get the POI observations
    // Determine the number of unique class IDs
    std::unordered_set<int> uniqueClassIds;
    for(const auto& poi : environment.getPOIs()) {
        uniqueClassIds.insert(poi.classId);
    }
    int numberOfPOIClasses = uniqueClassIds.size();

    int* POIObservations = new int[numberOfPOIClasses * numberOfSensors]; // Store POI observations excusively
    // Initialize all elements to zero
    for (int i = 0; i < numberOfPOIClasses * numberOfSensors; ++i) {
        POIObservations[i] = 0;
    }

    for(auto& poi : environment.getPOIs()) {
        // Calculate the angle between the central point and point of interest
        double dx = poi.x - posX;
        double dy = poi.y - posY;
        double angle = atan2(dy, dx) * 180 / M_PI; // Convert to degrees
        double distance = sqrt(dx * dx + dy * dy);

        // Normalize the angle to be in the range [0, 360)
        if (angle < 0) {
            angle += 360;
        }

        // Calculate the angle between each cone boundary
        double coneAngle = 360.0 / numberOfSensors;

        // Check which cone the point lies inside
        for (int i = 0; i < numberOfSensors; ++i) {
            double coneStart = i * coneAngle;
            double coneEnd = (i + 1) * coneAngle;
            
            // Adjust for negative angles
            coneStart = (coneStart < 0) ? coneStart + 360 : coneStart;
            coneEnd = (coneEnd < 0) ? coneEnd + 360 : coneEnd;
            
            // Check if the angle falls within the current cone
            if (coneStart <= angle && angle <= coneEnd && distance <= observationRadius) {
                POIObservations[poi.classId * numberOfSensors + i]++; // Incremenet obs at cone index
            }
        }
    }

    int* agentObservations = new int[numberOfSensors]; // Store the other agents observations
    // Initialize all elements to zero
    for (int i = 0; i < numberOfSensors; ++i) {
        agentObservations[i] = 0;
    }

    for (auto& agent : agents) {
        // Calculate the angle between the central point and point of interest
        double dx = agent.getPosition().first - posX;
        double dy = agent.getPosition().second - posY;

        // Do not process the same agent in the observation
        if (dx == 0 && dy == 0)
            continue;
        
        double angle = atan2(dy, dx) * 180 / M_PI; // Convert to degrees
        double distance = sqrt(dx * dx + dy * dy);

        // Normalize the angle to be in the range [0, 360)
        if (angle < 0) {
            angle += 360;
        }

        // Calculate the angle between each cone boundary
        double coneAngle = 360.0 / numberOfSensors;

        // Check which cone the point lies inside
        for (int i = 0; i < numberOfSensors; ++i) {
            double coneStart = i * coneAngle;
            double coneEnd = (i + 1) * coneAngle;
            
            // Adjust for negative angles
            coneStart = (coneStart < 0) ? coneStart + 360 : coneStart;
            coneEnd = (coneEnd < 0) ? coneEnd + 360 : coneEnd;
            
            // Check if the angle falls within the current cone
            if (coneStart <= angle && angle <= coneEnd && distance <= observationRadius) {
                agentObservations[i]++; // Incremenet obs at cone index
            }
        }
    }

    // Append the POIObservations array to the observations vector
    for (int i = 0; i < numberOfPOIClasses * numberOfSensors; ++i) {
        observations.push_back(POIObservations[i]);
    }
    // Append the agentObservations array to the observations vector
    for (int i = 0; i < numberOfSensors; ++i) {
        observations.push_back(agentObservations[i]);
    }
    // Delete the dynamically allocated array to free memory
    delete[] POIObservations;
    delete[] agentObservations;
    
    return observations;
}

// Function to get the current position of the agent
std::pair<double, double> Agent::getPosition() const {
    return std::make_pair(posX, posY);
}

int Agent::getMaxStepSize() const {
    return maxStepSize;
}

Team::Team(const std::string& filename, int id) {
    YAML::Node config = YAML::LoadFile(filename); // Parse YAML from file

    const YAML::Node& team_config = config["team"]; // Team config info
    const YAML::Node& agent_config = config["agent"]; // Agent config info

    bool randomStartPosition = agent_config["randomStartPosition"].as<bool>(); // Are the start pos random?

    for (int i = 0; i < team_config["numberOfAgents"].as<int>(); i++) {
        agents.emplace_back(0, 0, agent_config["maxStepSize"].as<int>(),
            agent_config["observationRadius"].as<double>(),
            agent_config["numberOfSensors"].as<int>(),
            config["environment"]["numberOfClassIds"].as<int>()); // Create agent object and store in vector
    }

    this->id = id; // Store the team id
}

void Team::printInfo() {
    std::cout<<"Team ID: "<<id<<std::endl;
    for (auto& agent : agents) {
        std::cout<<"    Agent position: "<<agent.getPosition().first
        <<","<<agent.getPosition().second<<std::endl;
    }
    std::cout<<"==========================="<<std::endl;
}

// simualate the team in the provided environment
std::vector<std::vector<int>> Team::simulate(const std::string& filename, Environment environment)
{
    YAML::Node config = YAML::LoadFile(filename); // Parse YAML from file
    
    const YAML::Node& agent_config = config["agent"]; // Agent config info
    
    bool randomStartPosition = agent_config["randomStartPosition"].as<bool>(); // Are the start pos random?
    int startingX, startingY;

    for(auto& agent : agents) {
        if (randomStartPosition == true) {
            startingX = rand()%(environment.getDimensions().first+1); // Get random within limits
            startingY = rand()%(environment.getDimensions().second+1);
        } else {
            startingX = agent_config["startingX"].as<int>(); // Read from config
            startingY = agent_config["startingY"].as<int>();
        }

        // reset the agents at the starting positions and clear the observations
        agent.set(startingX, startingY);
    }
    
    // Move as per policy for as many steps as in the episode length
    int episodeLength = config["episode"]["length"].as<int>();
    // Reward at each timestep in this episode
    std::vector<std::vector<int>> rewardHistory; 
    for(int stepNumber = 0; stepNumber < episodeLength; stepNumber++) {
        // Display the current stae of all agents
        printInfo();
        // Get the rewards for the current team configuration
        std::vector<std::pair<double, double>> agentPositions;
        for (auto& agent : agents) {
            agentPositions.push_back(agent.getPosition());
        }

        rewardHistory.push_back(environment.getRewards(agentPositions));
        std::cout<<"The reward is: "<<rewardHistory.back()<<std::endl;

        // Get the observation for each agent and feed it to its network to get the move
        std::vector<std::pair<double, double>> agentDeltas;
        for (auto& agent : agents) {
            agentDeltas.push_back(agent.policy.forward(agent.observe(environment, agents)));
        }

        // Move each agent according to its delta
        for (int i = 0; i < agents.size(); i++) {
            agents[i].move(agentDeltas[i], environment);
        }
    }

    return rewardHistory;
}
