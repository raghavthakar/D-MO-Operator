#ifndef MOBPBASEAGENT_H
#define MOBPBASEAGENT_H

#include "environment.h"
#include "torch/torch.h"

class MOBPPolicy : public torch::nn::Module {
public:
    MOBPPolicy();
    MOBPPolicy(int inputSize, double  weightLowerLimit, double weightUpperLimit);
    MOBPPolicy(const MOBPPolicy& other);
    short int forward (const std::vector<double>& input);
    void display();
    void addNoise(double mean, double stddev);
    std::string getPolicyAsString();
private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    void displayWeights(const torch::Tensor& weight);
    std::string getWeightsAsString(const torch::Tensor& weight);
};

class MOBPBaseAgent {
public:
    unsigned short int _pos;
    unsigned short int _startingPos;
    unsigned short int _gender;
    unsigned short int _occupation;
    double _nnWeightMin; // Max and min weights of the policy weights
    double _nnWeightMax;
    double _noiseMean; // POlicy noise mutate's mean
    double _noiseStdDev; // MOBPPolicy noise mutate's stddev

    MOBPPolicy policy;

    MOBPBaseAgent();
    MOBPBaseAgent(unsigned short int pos_, unsigned short int gender_, unsigned short int occupation_, unsigned short int startingPos_, double nnWeightMin_, double nnWeightMax_, double noiseMean_, double noiseStdDev_);
    MOBPBaseAgent(const MOBPBaseAgent& other);
    void move(unsigned short int delta, Environment environment);
    void reset();
    unsigned short int getPosition() const;
    short int forward(const std::vector<double>& input);
    unsigned short int observe();
};

#endif