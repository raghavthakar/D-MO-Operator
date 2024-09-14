#ifndef MOBPDomain_H
#define MOBPDomain_H

// Reference: https://ala2017.cs.universityofgalway.ie/papers/ALA2017_Mannion_Analysing.pdf

#include <vector>
#include <iostream>
#include <string>

class BeachSection {
    unsigned short int _section_id; // mus be unique
    unsigned int _psi; // the capacity of the section
    double _getLocalCapacityReward(std::vector<unsigned short int> agentPositions);
    double _getLocalMixtureReward(std::vector<unsigned short int> agentPositions, std::vector<unsigned short int> agentTypes, unsigned short int numBeachSections);

public:
    enum genderType {male, female};
    enum occupationType {student, working, retired};
    enum objectiveTypedef {cap, mix, occupation};
    BeachSection(unsigned short int section_id_, unsigned int psi_);
    std::vector<double> getLocalRewards(std::vector<unsigned short int> agentPositions, std::vector<unsigned short int> agentTypes, unsigned short int numBeachSections);
};

class MOBPDomain
{
private:
    std::vector<BeachSection> _beachSections;
public:
    std::string whichDomain;
    MOBPDomain();
    MOBPDomain(std::vector<unsigned int> psis);
    MOBPDomain(const std::string& config_filename);
    unsigned short int getAgentObservation(unsigned short int agentPos);
    unsigned short int moveAgent(unsigned short int agentPos, short int move);
    std::vector<double> getRewards(std::vector<unsigned short int> agentPositions, std::vector<unsigned short int> agentTypes);
    std::vector<std::vector<unsigned short int>> generateCounterfactualTrajectory(const std::string& config_filename, const std::string& counterfactualType, int trajectoryLength, unsigned short int startingPos);
    std::vector<double> initialiseEpisodeReward(const std::string& config_filename);
    void reset();
};

#endif // MOBPDomain_H

