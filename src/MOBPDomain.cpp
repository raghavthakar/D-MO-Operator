#include "MOBPDomain.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

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

    auto localCapReward = (numOccupyingAgents) * (std::exp(- numOccupyingAgents / this->_psi));
    return localCapReward;
}

// Beac Section local mixture reward
double BeachSection::_getLocalMixtureReward(std::vector<unsigned short int> agentPositions, std::vector<unsigned short int> agentTypes, unsigned short int numBeachSections) {
    if (agentPositions.size() != agentTypes.size()) {
        std::cout<<"Agent positions must be equal to agent types. Exiting...\n";
        std::exit(1);
    }

    if (agentPositions.size() <= 0) {
        std::cout<<"Agent positions and types must have non-zero length. Exiting...\n";
        std::exit(1);
    }
    
    std::vector<unsigned int> numOccupyingAgents(2, 0); // initialise both male and female as 0

    for (std::size_t i = 0; i < agentPositions.size(); i++) {
        if (agentPositions[i] == this->_section_id) {
            numOccupyingAgents[agentTypes[i]]++; // increment the counter for the corresponding agent type
        }
    }

    // min(male, female) / ((male + female) * numbeachsections)
    auto localMixReward = (*std::min_element(numOccupyingAgents.begin(), numOccupyingAgents.end())) / ((numOccupyingAgents[male] + numOccupyingAgents[female]) * numBeachSections);

    return localMixReward;
}

// Net local rewards
std::vector<double> BeachSection::getLocalRewards(std::vector<unsigned short int> agentPositions, std::vector<unsigned short int> agentTypes, unsigned short int numBeachSections) {
    std::vector<double> localRewards(2, 0);
    localRewards[this->cap] = this->_getLocalCapacityReward(agentPositions);
    localRewards[this->mix] = this->_getLocalMixtureReward(agentPositions, agentTypes, numBeachSections);

    return localRewards;
}

MOBPDomain::MOBPDomain() {
    int x = 2;
}

MOBPDomain::MOBPDomain(std::vector<unsigned int> psis) {
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

// The agent can only observe which section of the beach it is currently in
unsigned short int MOBPDomain::getAgentObservation(unsigned short int agentPos) {
    if (agentPos < 0 || agentPos >= this->_beachSections.size()) {
        std::cout<<"Agent is out of bounds. Invalid pos. Exiting...\n";
        std::exit(1);
    }

    return agentPos;
}

// move an agent within the domain bounds based on the provided move and return the new position
unsigned short int MOBPDomain::moveAgent(unsigned short int agentPos, short int move) {
    if (move < -1 || move > 1) {
        std::cout<<"Can only move one step at a time, and the provided move is too large. Exiting...\n";
        std::exit(1);
    }

    if (agentPos < 0 || agentPos >= this->_beachSections.size()) {
        std::cout<<"Agent is out of bounds. Invalid pos. Exiting...\n";
        std::exit(1);
    }

    short int newPos = agentPos + move;
    // cannot move if outside domain limuits
    if (newPos < 0 || newPos >= this->_beachSections.size())
        return agentPos;
    // update position if within limits
    else
        return newPos;
}
