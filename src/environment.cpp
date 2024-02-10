#include "environment.h"
#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>

POI::POI(int id, int classId, double x, double y, double observationRadius, int coupling, 
    int reward, int penalty)
    : id(id), classId(classId), x(x), y(y), observationRadius(observationRadius), coupling(coupling), 
    reward(reward), penalty(penalty) {}

void Environment::loadConfig(const std::string& filename) {
    YAML::Node config = YAML::LoadFile(filename); // Parse YAML from file

    const YAML::Node& dimensions = config["environment"]["dimensions"];
    xLength = dimensions["xLength"].as<double>();
    yLength = dimensions["yLength"].as<double>(); // Read the environment dimensions into the object

    for (const auto& poi : config["environment"]["pois"]) {
        pois.emplace_back(poi["id"].as<int>(), poi["classId"].as<int>(), poi["x"].as<double>(), 
        poi["y"].as<double>(), poi["observationRadius"].as<double>(),
        config["environment"]["coupling"].as<int>(),
        config["environment"]["reward"].as<int>(),
        config["environment"]["penalty"].as<int>()); // Create POI object and add to vector
    }
}

void Environment::printInfo() const {
    std::cout << "POIs in the environment:" << std::endl;
    for (const auto& poi : pois) {
        std::cout << "ID: " << poi.id << ", Class: " << poi.classId 
        << ", Coordinates: (" << poi.x << ", " << poi.y << "), Observation Radius: " 
        << poi.observationRadius << std::endl;
    }
}

std::vector<POI> Environment::getPOIs() {
    return pois;
}

std::pair<int, int> Environment::getDimensions() {
    return std::make_pair(xLength, yLength);
}
