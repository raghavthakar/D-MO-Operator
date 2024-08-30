#include "team.h"
#include "environment.h"
#include "policy.h"
#include <vector>
#include <unordered_set>
#include <string>
#include <iostream>
#include <cmath>
#include <string>
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

// = operator

// Constructor
Agent::Agent(double x, double y, double _maxStepSize, double _observationRadius, 
    int _numberOfSensors, int numberOfClassIds, double _nnWeightMin, double _nnWeightMax, double _noiseMean, double _noiseStdDev) : 
    posX(x), posY(y), maxStepSize(_maxStepSize), 
    observationRadius(_observationRadius), numberOfSensors(_numberOfSensors), noiseMean(_noiseMean),
    noiseStdDev(_noiseStdDev), policy(2 + _numberOfSensors * (numberOfClassIds) + _numberOfSensors, _nnWeightMin, _nnWeightMax)  {}

// copy constructor
Agent::Agent(const Agent& other) : posX(other.posX), posY(other.posY), maxStepSize(other.maxStepSize), 
    observationRadius(other.observationRadius), numberOfSensors(other.numberOfSensors), nnWeightMin(other.nnWeightMin), 
    nnWeightMax(other.nnWeightMax), noiseMean(other.noiseMean), noiseStdDev(other.noiseStdDev) {
        this->policy = *std::make_shared<Policy>(other.policy);;
}

// Function to move the agent by dx, dy (within maximum step size)
void Agent::move(std::pair<double, double> delta, Environment environment) {
    std::pair<double, double> newPosition = environment.moveAgent(std::make_pair(posX, posY), delta, this->maxStepSize);

    // Update the agent's position
    posX = newPosition.first;
    posY = newPosition.second;
}

// Function to set the agent at the starting position and clear its observations
void Agent::set(int startingX, int startingY) {
    posX = startingX;
    posY = startingY;
}

// Adds noise to the contained policy
void Agent::addNoiseToPolicy() {
    this->policy.addNoise(this->noiseMean, this->noiseStdDev);
}

// Observe and create state vector
// Assumes that POIs have classID 0, 1, 2....
std::vector<double> Agent::observe(Environment environment, std::vector<std::pair<double, double>> agentPositions) {
    return environment.getAgentObservations(std::make_pair(posX, posY), this->numberOfSensors, this->observationRadius, agentPositions);
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
        int posX, posY;
        if (randomStartPosition == true) {
            posX = rand() % config["MOREPDomain"]["dimensions"]["xLength"].as<int>();
            posY = rand() % config["MOREPDomain"]["dimensions"]["yLength"].as<int>();
        }
        else {
            posX = config["agent"]["startingX"].as<int>();
            posY = config["agent"]["startingY"].as<int>();
        }
        agents.emplace_back(posX, posY, agent_config["maxStepSize"].as<int>(),
            agent_config["observationRadius"].as<double>(),
            agent_config["numberOfSensors"].as<int>(),
            config["MOREPDomain"]["numberOfClassIds"].as<int>(),
            agent_config["nnWeightMin"].as<double>(),
            agent_config["nnWeightMax"].as<double>(),
            agent_config["noiseMean"].as<double>(),
            agent_config["noiseStdDev"].as<double>()); // Create agent object and store in vector
    }

    this->id = id; // Store the team id

    this->teamTrajectory.clear(); // clears the teamTrajectory of the team
}

Team::Team(const std::string& filename, std::vector<Agent> agents, int id) {
    YAML::Node config = YAML::LoadFile(filename); // Parse YAML from file

    const YAML::Node& team_config = config["team"]; // Team config info
    const YAML::Node& agent_config = config["agent"]; // Agent config info

    bool randomStartPosition = agent_config["randomStartPosition"].as<bool>(); // Are the start pos random?

    this->agents = agents;

    this->id = id; // Store the team id

    this->teamTrajectory.clear(); // clears the teamTrajectory of the team
}

void Team::printInfo() {
    std::cout<<"Team ID: "<<id<<std::endl;
    for (auto& agent : agents) {
        std::cout<<"    Agent position: "<<agent.getPosition().first
        <<","<<agent.getPosition().second<<std::endl;
    }
    std::cout<<"======="<<std::endl;
}

// mutate the policies of the contrained agents
void Team::mutate() {
    for (auto &agent : this->agents) {
        agent.addNoiseToPolicy();
    }
}

// simualate the team in the provided environment. Returns a vecotr of rewards from each timestep
std::vector<std::vector<int>> Team::simulate(const std::string& filename, Environment environment) {

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
    
    // clear the teamTrajectory of the team
    teamTrajectory.clear();
    // Move as per policy for as many steps as in the episode length
    int episodeLength = config["episode"]["length"].as<int>();
    // Reward at each timestep in this episode
    std::vector<std::vector<int>> rewardHistory; 
    for(int stepNumber = 0; stepNumber < episodeLength; stepNumber++) {
        // Display the current stae of all agents
        // printInfo();
        // Get the rewards for the current team configuration
        std::vector<std::pair<double, double>> agentPositions;
        for (auto& agent : agents) {
            agentPositions.push_back(agent.getPosition());
        }

        // push these agent positions to the teamTrajectory
        teamTrajectory.push_back(agentPositions);

        // compute the rewards for these agent positions
        rewardHistory.push_back(environment.getRewards(agentPositions, stepNumber));
        // std::cout<<"The reward is: "<<rewardHistory.back()<<std::endl;

        // Get the observation for each agent and feed it to its network to get the move
        std::vector<std::pair<double, double>> agentDeltas;
        for (auto& agent : agents) {
            agentDeltas.push_back(agent.policy.forward(agent.observe(environment, agentPositions)));
        }

        // Move each agent according to its delta
        for (int i = 0; i < agents.size(); i++) {
            agents[i].move(agentDeltas[i], environment);
        }
    }

    return rewardHistory;
}

// re-evaluate the rewards for the team, given the counterfactual trajectory
// TODO counterfactual evaluation find the rewards for that team
std::vector<std::vector<int>> Team::replayWithCounterfactual(const std::string& filename, Environment environment, const std::string& counterfactualType) {
    std::vector<std::pair<double, double>> counterfactualTrajectory;
    YAML::Node config = YAML::LoadFile(filename); // Parse YAML from file

    if (counterfactualType == "static") {
        double startingX= config["agent"]["startingX"].as<double>();
        double startingY= config["agent"]["startingY"].as<double>();

        // static counterfactual trajectory with length equal to the teamtrajectory
        for (int i=0; i<this->teamTrajectory.size(); i++) {
            counterfactualTrajectory.push_back(std::make_pair(startingX, startingY));
        }
    }

    std::vector<std::vector<int>> replayRewardsWithCounterfactuals; // Stores the replay rewards with counterfactual replacements
    std::vector<std::vector<std::pair<double, double>>> workingTeamTrajectory; // store the team trajectory copy to modify
    // for each agent, loop through the episode, get rewards for each timestep with counterfactial replacements
    for (int agentNum=0; agentNum<this->teamTrajectory[0].size(); agentNum++) { // loop through agents
        workingTeamTrajectory = this->teamTrajectory;

        // Loop through the working trajectory, replacing the agent position at that timestep with position from counterfactual trajectory
        std::vector<int> episodeCounterfactualRewards(config["MOREPDomain"]["numberOfClassIds"].as<int>(), 0); // Sum of timestep rewards 
        for(int timestep = 0; timestep < workingTeamTrajectory.size(); timestep++) {
            workingTeamTrajectory[timestep][agentNum] = counterfactualTrajectory[timestep]; // repalce the agent's position with counterfactual
            std::vector<int> timestepRewards = environment.getRewards(workingTeamTrajectory[timestep], timestep); // get the rewards for the team with counterfactual agent at this timestep
            
            for(int rewIndex = 0; rewIndex < timestepRewards.size(); rewIndex++) {
                episodeCounterfactualRewards[rewIndex] += timestepRewards[rewIndex]; // add tiemstep rewards to the cumulative episode rewards
            }
        }

        // Append the episode rewards with counterfactual for this agent to the replayReward vector
        replayRewardsWithCounterfactuals.push_back(episodeCounterfactualRewards);
    }

    return replayRewardsWithCounterfactuals;   
}