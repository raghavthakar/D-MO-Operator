#ifndef MOREPPOLICY_H
#define MOREPPOLICY_H

#include "torch/torch.h"

class MOREPPolicy : public torch::nn::Module {
public:
    MOREPPolicy();
    MOREPPolicy(int inputSize, double weightLowerLimit, double weightUpperLimit);
    MOREPPolicy(const MOREPPolicy& other);

    std::pair<double, double> forward(const std::vector<double>& input);
    void display();

    void addNoise(double mean, double stddev);
    std::string getPolicyAsString();

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    void displayWeights(const torch::Tensor& weight);
    std::string getWeightsAsString(const torch::Tensor& weight);
};

#endif // POLICY_H