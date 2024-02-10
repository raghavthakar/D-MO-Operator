#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <vector>
#include <string>
#include <utility>
#include <cmath>
#include <unordered_set>

// Point of Interest (POI) class definition
class POI {
public:
    int id;
    int classId;
    double x, y; // Coordinates of the POI
    double observationRadius; // Observation radius of the POI
    int coupling; // Minimum number of agents needed in vicinity to generate reward 
    int reward; // Generated by each POI if coupling criteria is met
    int penalty; // Penalty given to each agent per timestep

    POI(int id, int classId, double x, double y, 
        double observationRadius, int coupling, int reward, int penalty);
};

// Simulation environment class definition
class Environment {
private:
    std::vector<POI> pois; // Vector to store POIs
    int xLength, yLength; // Environment dimensions

public:
    // Method to load configuration from YAML file
    void loadConfig(const std::string& filename);

    // Method to return the POIs in the environment
    std::vector<POI> getPOIs();

    // Method to return the dimensions of the environment
    std::pair<int, int> getDimensions();

    // compute the rewards generated by the provided agents configuration
    std::vector<int> getRewards(std::vector<std::pair<double, double>> agentPositions) {
        std::vector<int> rewardVector;

        // Determine the number of unique class IDs
        std::unordered_set<int> uniqueClassIds;
        for(const auto& poi : pois) {
            uniqueClassIds.insert(poi.classId);
        }
        int numberOfPOIClasses = uniqueClassIds.size();

        // As many elements in the reward vector as objectives (POI classes)
        for(int i = 0; i < numberOfPOIClasses; i++)
            rewardVector.push_back(0);
        
        // loop through each POI, and add its reward accordingly
        for (const auto& poi : pois) {
            int numberOfCloseAgents = 0;

            for (const auto agentPosition : agentPositions) {
                double posX = agentPosition.first;
                double posY = agentPosition.second;

                double dx = poi.x - posX;
                double dy = poi.y - posY;
                double distance = sqrt(dx * dx + dy * dy);

                if(distance <= poi.observationRadius)
                    numberOfCloseAgents++;
            }

            rewardVector[poi.classId] = (numberOfCloseAgents >= poi.coupling) ? 
                (rewardVector[poi.classId] + poi.reward) :
                (rewardVector[poi.classId]);
        }

        // Add in the penalties of each agent to each objective reward
        for(int i = 0; i < rewardVector.size(); i++)
            rewardVector[i] += agentPositions.size() * pois[0].penalty;

        return rewardVector;
    }

    // Method to print information about loaded POIs
    void printInfo() const;

};

#endif // ENVIRONMENT_H
