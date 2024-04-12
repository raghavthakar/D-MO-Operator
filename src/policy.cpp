#include "policy.h"
#include "torch/torch.h"
#include <iostream>

Policy::Policy() {}

Policy::Policy(int inputSize, double weightLowerLimit, double weightUpperLimit) {
    // Define the layers of the neural network
    fc1 = register_module("fc1", torch::nn::Linear(inputSize, 6));
    fc2 = register_module("fc2", torch::nn::Linear(6, 6));
    fc3 = register_module("fc3", torch::nn::Linear(6, 2));

    // Initialize weights with random values
    torch::NoGradGuard no_grad; // Disable gradient computation temporarily
    torch::nn::init::uniform_(fc1->weight, weightLowerLimit, weightUpperLimit);
    torch::nn::init::uniform_(fc2->weight, weightLowerLimit, weightUpperLimit);
    torch::nn::init::uniform_(fc3->weight, weightLowerLimit, weightUpperLimit);
}

// copy construct (deep copy)
Policy::Policy(const Policy& other) {
    // Define the layers of the neural network
    fc1 = register_module("fc1", torch::nn::Linear(other.fc1->weight.size(1), other.fc1->weight.size(0)));
    fc2 = register_module("fc2", torch::nn::Linear(other.fc2->weight.size(1), other.fc2->weight.size(0)));
    fc3 = register_module("fc3", torch::nn::Linear(other.fc3->weight.size(1), other.fc3->weight.size(0)));

    // Copy the weights from the original policy to the new one
    torch::NoGradGuard no_grad; // Disable gradient computation temporarily
    fc1->weight.copy_(other.fc1->weight);
    fc1->bias.copy_(other.fc1->bias);
    fc2->weight.copy_(other.fc2->weight);
    fc2->bias.copy_(other.fc2->bias);
    fc3->weight.copy_(other.fc3->weight);
    fc3->bias.copy_(other.fc3->bias);
}

std::pair<double, double> Policy::forward(const std::vector<double>& input) {
    // Convert input vector to a torch::Tensor
    torch::Tensor x = torch::tensor(input).view({1, -1});

    // Apply the layers sequentially with ReLU activation
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = torch::tanh(fc3->forward(x));

    // Extract output values from tensor and return as pair
    auto output_tensor = x.squeeze();
    double delta_x = output_tensor[0].item<double>();
    double delta_y = output_tensor[1].item<double>();
    return {delta_x, delta_y};
}

// Display method to print out the weights of each layer
void Policy::display() {
    std::cout << "Weights of fc1:\n" << getWeightsAsString(fc1->weight) << std::endl;
    std::cout << "Weights of fc2:\n" << getWeightsAsString(fc2->weight) << std::endl;
    std::cout << "Weights of fc3:\n" << getWeightsAsString(fc3->weight) << std::endl;
}

// Display method to print out the weights of each layer
std::string Policy::getPolicyAsString() {
    std::stringstream output;
    output << "Weights of fc1:\n" << getWeightsAsString(fc1->weight) << std::endl;
    output << "Weights of fc2:\n" << getWeightsAsString(fc2->weight) << std::endl;
    output << "Weights of fc3:\n" << getWeightsAsString(fc3->weight) << std::endl;

    return output.str();
}

// Function to display the weights of a tensor
void Policy::displayWeights(const torch::Tensor& weight) {
    auto weight_accessor = weight.accessor<float, 2>();
    for (int i = 0; i < weight_accessor.size(0); ++i) {
        for (int j = 0; j < weight_accessor.size(1); ++j) {
            std::cout << weight_accessor[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Function to return the weights of a tensor as a string
std::string Policy::getWeightsAsString(const torch::Tensor& weight) {
    std::stringstream output;
    auto weight_accessor = weight.accessor<float, 2>();
    for (int i = 0; i < weight_accessor.size(0); ++i) {
        for (int j = 0; j < weight_accessor.size(1); ++j) {
            output << weight_accessor[i][j] << " ";
        }
        output << "\n"; // Add newline after each row
    }
    return output.str();
}

// Function to add the prescrived noise to the policy weights
void Policy::addNoise(double mean, double stddev) {
    // Add noise to the weights of each linear layer
        torch::NoGradGuard no_grad; // Disable gradient tracking

        // Add noise to the weights of the first linear layer (fc1)
        if (fc1) {
            auto weights1 = fc1->weight.data();
            weights1.add_(torch::randn_like(weights1) * stddev + mean);
        }

        if (fc2) {
            auto weights2 = fc2->weight.data();
            weights2.add_(torch::randn_like(weights2) * stddev + mean);
        }

        if (fc1) {
            auto weights3 = fc3->weight.data();
            weights3.add_(torch::randn_like(weights3) * stddev + mean);
        }
}
