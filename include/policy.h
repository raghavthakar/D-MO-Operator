#ifndef POLICY_H
#define POLICY_H

#include "torch/torch.h"

class Policy : public torch::nn::Module {
public:
    Policy();
    Policy(int inputSize, double weightLowerLimit, double weightUpperLimit);

    std::pair<double, double> forward(const std::vector<double>& input);
    void display();

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    void displayWeights(const torch::Tensor& weight);
};

#endif // POLICY_H