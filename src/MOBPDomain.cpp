#include "MOBPDomain.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <yaml-cpp/yaml.h>

// Beach Section constructor
BeachSection::BeachSection(unsigned short int section_id_, unsigned int psi_) {
    this->_section_id = section_id_;
    this->_psi = psi_;
}

// Beach Section local capacity reward
double BeachSection::_getLocalCapacityReward(std::vector<unsigned short int> agentPositions) {
    unsigned int numOccupyingAgents = 0; // How many agents are occupying this beach section?
    // count the occupying agents
    for (auto i : agentPositions) { // If position matches section id
        if (i == this->_section_id)
            numOccupyingAgents++;
    }

    double localCapReward = numOccupyingAgents * std::exp(-static_cast<double>(numOccupyingAgents) / this->_psi);
    return localCapReward;
}

// Beac Section local mixture reward
double BeachSection::_getLocalMixtureReward(std::vector<unsigned short int> agentPositions, std::vector<unsigned short int> agentGenderTypes, unsigned short int numBeachSections) {
    if (agentPositions.size() != agentGenderTypes.size()) {
        std::cout << "Agent positions must be equal to agent types. Exiting...\n";
        std::exit(1);
    }

    if (agentPositions.size() <= 0) {
        std::cout << "Agent positions and types must have non-zero length. Exiting...\n";
        std::exit(1);
    }

    std::vector<unsigned int> numOccupyingAgents(2, 0); // Initialize both male and female as 0

    for (std::size_t i = 0; i < agentPositions.size(); i++) {
        if (agentPositions[i] == this->_section_id) {
            numOccupyingAgents[agentGenderTypes[i]]++; // Increment the counter for the corresponding agent type
        }
    }

    unsigned int totalAgents = numOccupyingAgents[male] + numOccupyingAgents[female]; // Total number of agents (male + female)

    // Check for division by zero
    if (totalAgents == 0) {
        return 0.0;
    }

    double localMixReward = (*std::min_element(numOccupyingAgents.begin(), numOccupyingAgents.end())) 
                            / (static_cast<double>(totalAgents) * numBeachSections);

    return localMixReward;
}

// Net local rewards
std::vector<double> BeachSection::getLocalRewards(std::vector<unsigned short int> agentPositions, std::vector<unsigned short int> agentTypes, unsigned short int numBeachSections) {
    std::vector<double> localRewards(2, 0); // TODO make this generic
    localRewards[this->cap] = this->_getLocalCapacityReward(agentPositions);
    localRewards[this->mix] = this->_getLocalMixtureReward(agentPositions, agentTypes, numBeachSections);

    return localRewards;
}

MOBPDomain::MOBPDomain() {
    this->whichDomain = "MOBPDomain";
}

MOBPDomain::MOBPDomain(std::vector<unsigned int> psis) {
    this->whichDomain = "MOBPDomain";

    if (psis.size() <= 0) {
        std::cout<<"Problem size is ill-defined. Exiting...\n";
        std::exit(1);
    }
    // create unique beach sections and assign their IDs
    for (std::size_t i = 0; i < psis.size(); i++) {
        if (psis[i] < 0) {
            std::cout<<"Beach section capacity cannot be < 0. Exiting...\n";
            std::exit(1);
        }
        this->_beachSections.push_back(BeachSection(i, psis[i]));
    }
}

MOBPDomain::MOBPDomain(const std::string& config_filename) {
    this->whichDomain = "MOBPDomain";

    YAML::Node config = YAML::LoadFile(config_filename); // Parse YAML from file
    const YAML::Node& MOBP_config = config["MOBPDomain"];

    // loop through sections from config and initialise them
    for (const auto& section : MOBP_config["Sections"]) {
        // get section info from the config
        int id = section["id"].as<int>();
        int psi = section["psi"].as<int>();        
        if (psi < 0) {
            std::cout<<"Beach section capacity cannot be < 0. Exiting...\n";
            std::exit(1);
        }
        this->_beachSections.push_back(BeachSection(id, psi));   
    }
}

// The agent can only observe which section of the beach it is currently in
unsigned short int MOBPDomain::getAgentObservation(unsigned short int agentPos) {
    if (agentPos < 0 || agentPos >= this->_beachSections.size()) {
        std::cout<<"Agent is out of bounds. Invalid pos. Exiting...\n";
        std::exit(1);
    }

    return agentPos;
}

// move an agent within the domain bounds based on the provided move and return the new position
unsigned short int MOBPDomain::moveAgent(unsigned short int agentPos, short int delta) {
    if (delta < -1 || delta > 1) {
        std::cout<<"Can only move one step at a time, and the provided move is too large. Exiting...\n";
        std::exit(1);
    }

    if (agentPos < 0 || agentPos >= this->_beachSections.size()) {
        std::cout<<"Agent is out of bounds. Invalid pos. Exiting...\n";
        std::exit(1);
    }

    short int newPos = agentPos + delta;
    // cannot move if outside domain limuits
    if (newPos < 0 || newPos >= this->_beachSections.size())
        return agentPos;
    // update position if within limits
    else
        return newPos;
}

// return the sum of local rewards for a given configuration of position and rytes of agents
std::vector<double> MOBPDomain::getRewards(std::vector<unsigned short int> agentPositions, std::vector<unsigned short int> agentTypes) {
    int numBeachSections = this->_beachSections.size();
    std::vector<double> globalRewards = {0, 0}; // TODO make this generic
    for (auto& section : this->_beachSections) {
        std::vector<double> localReward = section.getLocalRewards(agentPositions, agentTypes, numBeachSections);
        if (localReward.size() != globalRewards.size()) {
            std::cout<<"Something is wrong with beach domain reward vector size";
            exit(1);
        }

        // add the loval reward to the global reward
        for (size_t i = 0; i < globalRewards.size(); i++) {
            globalRewards[i] += localReward[i];
        }
    }

    return globalRewards;
}

// generate a counterfactual trajectory 
std::vector<std::vector<unsigned short int>> MOBPDomain::generateCounterfactualTrajectory(const std::string& config_filename, const std::string& counterfactualType, int trajectoryLength, unsigned short int startingPos) {
    std::vector<std::vector<unsigned short int>> counterfactualTrajectory;

    if (counterfactualType == "static") {
        // static counterfactual trajectory with length equal to the teamtrajectory (=episode length)
        for (int i=0; i<trajectoryLength; i++) {
            counterfactualTrajectory.push_back({startingPos});
        }
    }

    return counterfactualTrajectory;
}

// initialise zero rewards for an episode // intiialise zero reward for an episode
std::vector<double> MOBPDomain::initialiseEpisodeReward(const std::string& config_filename) {
    YAML::Node config = YAML::LoadFile(config_filename); // Parse YAML from file
    return std::vector<double>(config["experiment"]["numberOfObjectives"].as<int>(), 0.0);
}

// reset the params of the env
void MOBPDomain::reset() {
    return;
}