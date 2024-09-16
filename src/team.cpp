#include "team.h"
#include "environment.h"
#include <vector>
#include <unordered_set>
#include <utility>
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

// Constructor for rover
Agent::Agent(const std::string& config_filename) {
    YAML::Node config = YAML::LoadFile(config_filename); // Parse YAML from file

    this->whichDomain = config["experiment"]["domain"].as<std::string>();

    const YAML::Node& agent_config = config["agent"]; // Agent config info

    if (this->whichDomain == rover.whichDomain){
        this->rover = MOREPBaseAgent(agent_config["startingX"].as<int>(),
            agent_config["startingY"].as<int>(),
            agent_config["startingX"].as<int>(),
            agent_config["startingY"].as<int>(),
            agent_config["maxStepSize"].as<int>(),
            agent_config["observationRadius"].as<double>(),
            agent_config["numberOfSensors"].as<int>(),
            config["experiment"]["numberOfObjectives"].as<int>(),
            agent_config["nnWeightMin"].as<double>(),
            agent_config["nnWeightMax"].as<double>(),
            agent_config["noiseMean"].as<double>(),
            agent_config["noiseStdDev"].as<double>());
    }
    else {
        throw std::runtime_error("Domain is unrecognized!");
    }
}

// constructor for beach
Agent::Agent(unsigned short int pos_, std::string gender_, unsigned short int startingPos_, const std::string& config_filename) {
    YAML::Node config = YAML::LoadFile(config_filename); // Parse YAML from file
    const YAML::Node& agent_config = config["agent"]; // Agent config info

    this->whichDomain = config["experiment"]["domain"].as<std::string>();

    // convert string gender to unsigned short int
    unsigned short int gender_int;
    if (gender_ == "male")
        gender_int = 0;
    else if (gender_ == "female")
        gender_int = 1;

    if (this->whichDomain == beachPerson.whichDomain){
        this->beachPerson = MOBPBaseAgent(pos_, 
                                gender_int, 
                                startingPos_,
                                agent_config["nnWeightMin"].as<double>(),
                                agent_config["nnWeightMax"].as<double>(),
                                agent_config["noiseMean"].as<double>(),
                                agent_config["noiseStdDev"].as<double>());
    }
    else {
        throw std::runtime_error("Domain is unrecognized!");
    }
}

// copy constructor
Agent::Agent(const Agent& other) {
    this->whichDomain = other.whichDomain;

    if (this->whichDomain == rover.whichDomain){
        this->rover = MOREPBaseAgent(other.rover);
    }
    else if (this->whichDomain == beachPerson.whichDomain)
    {
        this->beachPerson = MOBPBaseAgent(other.beachPerson);
    }
    else {
        throw std::runtime_error("Domain is unrecognized!");
    }
    
}

// Function to move the agent by dx, dy (within maximum step size)
void Agent::move(std::vector<double> delta, Environment environment) {
    if (this->whichDomain == rover.whichDomain){
        assert(delta.size() == 2 && "Rover delta must be 2!");
        this->rover.move(std::make_pair(delta[0], delta[1]), environment);
    }
    else if (this->whichDomain == beachPerson.whichDomain)
    {
        assert(delta.size() == 1 && "Beach delta cannot be more than 1 element!");
        this->beachPerson.move(delta[0], environment);
    }
    else {
        throw std::runtime_error("Domain is unrecognized!");
    }
}

// Function to set the agent at the starting position and clear its observations
void Agent::reset() {
    if (this->whichDomain == rover.whichDomain){
        this->rover.reset();
    }
    else if (this->whichDomain == beachPerson.whichDomain)
    {
        this->beachPerson.reset();
    }
    else {
        throw std::runtime_error("Domain is unrecognized!");
    }
}

// Adds noise to the contained policy
void Agent::addNoiseToPolicy() {
    if (this->whichDomain == rover.whichDomain){
        this->rover.addNoiseToPolicy();
    }
    else if (this->whichDomain == beachPerson.whichDomain)
    {
        this->beachPerson.addNoiseToPolicy();
    }
    else {
        throw std::runtime_error("Domain is unrecognized!");
    }
}

// Observe and create state vector
// Assumes that POIs have classID 0, 1, 2....
std::vector<double> Agent::observe(Environment environment, std::vector<std::vector<double>> agentPositions) {
    if (this->whichDomain == rover.whichDomain){
        // Convert std::vector<std::vector<double>> to std::vector<std::pair<double, double>>
        std::vector<std::pair<double, double>> convertedPositions;
        convertedPositions.reserve(agentPositions.size());

        for (const auto& position : agentPositions) {
            // Assuming each inner vector has exactly 2 elements
            if (position.size() == 2) {
                convertedPositions.emplace_back(position[0], position[1]);
            } else {
                throw std::runtime_error("Each position vector must have exactly 2 elements.");
            }
        }

        return this->rover.observe(environment, convertedPositions);
    }
    else if (this->whichDomain == beachPerson.whichDomain) {
        return std::vector<double>(1, this->beachPerson.observe());
    }
    else {
        throw std::runtime_error("Domain is unrecognized!");
    }
}

// forward pass through the policy
std::vector<double> Agent::forward(const std::vector<double>& input) {
    if (this->whichDomain == rover.whichDomain){
        auto output = this->rover.forward(input);
        return {output.first, output.second};
    }
    else if (this->whichDomain == beachPerson.whichDomain)
    {
        short int output = this->beachPerson.forward(input);
        return std::vector<double>(1, output);
    }
    else {
        throw std::runtime_error("Domain is unrecognized!");
    }
}

// Function to get the current position of the agent
std::vector<double> Agent::getPosition() const {
    if (this->whichDomain == rover.whichDomain){
        auto position = this->rover.getPosition();
        return {position.first, position.second};
    }
    else if (this->whichDomain == beachPerson.whichDomain)
    {
        auto position = this->beachPerson.getPosition();
        return {(double)position};
    }
    else {
        throw std::runtime_error("Domain is unrecognized!");
    }
}

Team::Team(const std::string& filename, int id) {
    YAML::Node config = YAML::LoadFile(filename); // Parse YAML from file

    this->whichDomain = config["experiment"]["domain"].as<std::string>();


    // initialise agents according to the experiment domain
    if (this->whichDomain == "MOREPDomain") {
        const YAML::Node& team_config = config["MOREPDomain"]["team"]; // Team config info
        for (int i = 0; i < team_config["numberOfAgents"].as<int>(); i++) {
            agents.emplace_back(filename); // Create agent object and store in vector
        }
    }
    // heterogeneous beach agents require work for initialisation
    else if (this->whichDomain == "MOBPDomain") {
        const YAML::Node& MOBP_config = config["MOBPDomain"];

        for (const auto& section : MOBP_config["Sections"]) {
            // get section info from the config
            int maleTourists = section["maleTourists"].as<int>();
            int femaleTourists = section["femaleTourists"].as<int>();
            int id = section["id"].as<int>();
            
            // initialise male agents at this section
            for (int i = 0; i < maleTourists; i++) {
                agents.emplace_back(id, "male", id, filename);
            }

            // initialise female agents at this section
            for (int i = 0; i < femaleTourists; i++) {
                agents.emplace_back(id, "female", id, filename);
            }
        }
    }

    this->id = id; // Store the team id

    this->teamTrajectory.clear(); // clears the teamTrajectory of the team
}

Team::Team(const std::string& filename, std::vector<Agent> agents, int id) {
    YAML::Node config = YAML::LoadFile(filename); // Parse YAML from file

    this->whichDomain = config["experiment"]["domain"].as<std::string>();

    const YAML::Node& team_config = config["team"]; // Team config info

    this->agents = agents;

    this->id = id; // Store the team id

    this->teamTrajectory.clear(); // clears the teamTrajectory of the team
}

void Team::printInfo() {
    std::cout<<"Team ID: "<<id<<std::endl;
    for (auto& agent : agents) {
        std::cout<<"    Agent position: "<<agent.getPosition()[0]
        <<","<<agent.getPosition()[1]<<std::endl;
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
std::vector<std::vector<double>> Team::simulate(const std::string& filename, Environment environment) {

    YAML::Node config = YAML::LoadFile(filename); // Parse YAML from file
    
    const YAML::Node& agent_config = config["agent"]; // Agent config info
    
    bool randomStartPosition = agent_config["randomStartPosition"].as<bool>(); // Are the start pos random?
    int startingX, startingY;

    for(auto& agent : agents) {
        // reset the agents at the starting positions and clear the observations
        agent.reset();
    }
    
    // clear the teamTrajectory of the team
    teamTrajectory.clear();
    // Move as per policy for as many steps as in the episode length
    int episodeLength = config["episode"]["length"].as<int>();
    // Reward at each timestep in this episode
    std::vector<std::vector<double>> rewardHistory; 
    for(int stepNumber = 0; stepNumber < episodeLength; stepNumber++) {
        // Display the current stae of all agents
        // printInfo();
        // Get the rewards for the current team configuration
        std::vector<std::vector<double>> agentPositions;
        for (auto& agent : agents) {
            agentPositions.push_back(agent.getPosition());
        }
        // push these agent positions to the teamTrajectory
        teamTrajectory.push_back(agentPositions);

        std::vector<unsigned short int> agentTypes;
        if (this->whichDomain == "MOBPDomain") { // if beach domain
            for (auto& agent : agents) {
                agentTypes.push_back(agent.beachPerson._gender); // need to send in the agent types as well
            }

            rewardHistory.push_back(environment.getRewards(agentPositions, agentTypes, stepNumber));
        }
        else if (this->whichDomain == "MOREPDomain") { // for the rover domain only the agent poisitions are neneded
            // compute the rewards for these agent positions
            rewardHistory.push_back(environment.getRewards(agentPositions, stepNumber));
        }
        else {
            throw std::runtime_error("Domain is unrecognized!");
        }

        // avoid the last extra simulation step if in the last iteration of the loop
        if (stepNumber == episodeLength - 1) {
            break;
        }

        // Get the observation for each agent and feed it to its network to get the move
        std::vector<std::vector<double>> agentDeltas;
        for (auto& agent : agents) {
            agentDeltas.push_back(agent.forward(agent.observe(environment, agentPositions)));
        }

        // Move each agent according to its delta
        for (int i = 0; i < agents.size(); i++) {
            agents[i].move(agentDeltas[i], environment);
        }
    }

    return rewardHistory;
}

// re-evaluate the rewards for the team, given the counterfactual trajectory
std::vector<std::vector<double>> Team::replayWithCounterfactual(const std::string& filename, Environment environment, const std::string& counterfactualType) {
    // get the counterfactual trajectory
    std::vector<std::vector<double>> counterfactualTrajectory;

    // counterfactual trajectories are identical for all rovers
    if (this->whichDomain == this->agents[0].rover.whichDomain) {
        counterfactualTrajectory = environment.generateCounterfactualTrajectory(filename, counterfactualType, this->teamTrajectory.size());
    }

    std::vector<std::vector<double>> replayRewardsWithCounterfactuals; // Stores the replay rewards with counterfactual replacements
    std::vector<std::vector<std::vector<double>>> workingTeamTrajectory; // store the team trajectory copy to modify
    // for each agent, loop through the episode, get rewards for each timestep with counterfactial replacements
    for (int agentNum=0; agentNum<this->teamTrajectory[0].size(); agentNum++) { // loop through agents
        workingTeamTrajectory = this->teamTrajectory;

        // but for each beach agent the starting position is different
        if (this->whichDomain == this->agents[agentNum].beachPerson.whichDomain) {
            counterfactualTrajectory = environment.generateCounterfactualTrajectory(filename, counterfactualType, this->teamTrajectory.size(), agents[agentNum].beachPerson._startingPos);
        }

        // Loop through the working trajectory, replacing the agent position at that timestep with position from counterfactual trajectory
        std::vector<double> episodeCounterfactualRewards = environment.initialiseEpisodeReward(filename); // Sum of timestep rewards 
        for(int timestep = 0; timestep < workingTeamTrajectory.size(); timestep++) {
            workingTeamTrajectory[timestep][agentNum] = counterfactualTrajectory[timestep]; // repalce the agent's position with counterfactual
            std::vector<double> timestepRewards;

            if (this->whichDomain == this->agents[0].rover.whichDomain) {
                timestepRewards = environment.getRewards(workingTeamTrajectory[timestep], timestep); // get the rewards for the team with counterfactual agent at this timestep
            }
            else if (this->whichDomain == this->agents[0].beachPerson.whichDomain) {
                std::vector<unsigned short int> agentTypes;
                for (auto& agent : this->agents) {
                    agentTypes.push_back(agent.beachPerson._gender); // need to send in the agent types as well
                }
                timestepRewards = environment.getRewards(workingTeamTrajectory[timestep], agentTypes, timestep); // get the rewards for the team with counterfactual agent at this timestep
            }
            
            for(int rewIndex = 0; rewIndex < timestepRewards.size(); rewIndex++) {
                episodeCounterfactualRewards[rewIndex] += timestepRewards[rewIndex]; // add tiemstep rewards to the cumulative episode rewards
            }
        }

        // Append the episode rewards with counterfactual for this agent to the replayReward vector
        replayRewardsWithCounterfactuals.push_back(episodeCounterfactualRewards);
    }

    return replayRewardsWithCounterfactuals;   
}