#include "MOBPBaseAgent.h"
#include "torch/torch.h"
#include <iostream>

MOBPPolicy::MOBPPolicy() {}

MOBPPolicy::MOBPPolicy(int inputSize, double weightLowerLimit, double weightUpperLimit) {
    // Define the layers of the neural network
    fc1 = register_module("fc1", torch::nn::Linear(inputSize, 2));  // Input layer (1 neuron) to hidden layer (2 neurons)
    fc2 = register_module("fc2", torch::nn::Linear(2, 3));          // Hidden layer (2 neurons) to output layer (3 neurons)

    // Initialize weights with random values
    torch::NoGradGuard no_grad; // Disable gradient computation temporarily
    torch::nn::init::uniform_(fc1->weight, weightLowerLimit, weightUpperLimit);
    torch::nn::init::uniform_(fc2->weight, weightLowerLimit, weightUpperLimit);
}


// copy construct (deep copy)
MOBPPolicy::MOBPPolicy(const MOBPPolicy& other) {
    // Define the layers of the neural network
    fc1 = register_module("fc1", torch::nn::Linear(other.fc1->weight.size(1), other.fc1->weight.size(0)));
    fc2 = register_module("fc2", torch::nn::Linear(other.fc2->weight.size(1), other.fc2->weight.size(0)));

    // Copy the weights from the original MOBPPolicy to the new one
    torch::NoGradGuard no_grad; // Disable gradient computation temporarily
    fc1->weight.copy_(other.fc1->weight);
    fc1->bias.copy_(other.fc1->bias);
    fc2->weight.copy_(other.fc2->weight);
    fc2->bias.copy_(other.fc2->bias);
}

short int MOBPPolicy::forward(const std::vector<double>& input) {
    // Convert input vector to a torch::Tensor (1D input tensor)
    torch::Tensor x = torch::tensor(input).view({1, -1});

    // Apply the layers sequentially with ReLU activation in the hidden layer
    x = torch::relu(fc1->forward(x));  // Input to hidden layer (ReLU)
    
    // Output layer, no activation (deterministic policy)
    x = fc2->forward(x);

    // Get the index of the maximum value from the output (determining the action)
    auto action_index = std::get<1>(torch::max(x, /*dim=*/1));

    // Map index to action: 0 -> -1, 1 -> 0, 2 -> +1
    int action = action_index.item<int>() - 1;

    return action;  // Return the action: -1, 0, or +1
}

// Display method to print out the weights of each layer
void MOBPPolicy::display() {
    std::cout << "Weights of fc1:\n" << getWeightsAsString(fc1->weight) << std::endl;
    std::cout << "Weights of fc2:\n" << getWeightsAsString(fc2->weight) << std::endl;
}

// Display method to print out the weights of each layer
std::string MOBPPolicy::getPolicyAsString() {
    std::stringstream output;
    output << "Weights of fc1:\n" << getWeightsAsString(fc1->weight) << std::endl;
    output << "Weights of fc2:\n" << getWeightsAsString(fc2->weight) << std::endl;

    return output.str();
}

// Function to display the weights of a tensor
void MOBPPolicy::displayWeights(const torch::Tensor& weight) {
    auto weight_accessor = weight.accessor<float, 2>();
    for (int i = 0; i < weight_accessor.size(0); ++i) {
        for (int j = 0; j < weight_accessor.size(1); ++j) {
            std::cout << weight_accessor[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Function to return the weights of a tensor as a string
std::string MOBPPolicy::getWeightsAsString(const torch::Tensor& weight) {
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

// Function to add the prescrived noise to the MOBPPolicy weights
void MOBPPolicy::addNoise(double mean, double stddev) {
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
}


MOBPBaseAgent::MOBPBaseAgent() {
    this->whichDomain = "MOBPDomain";
}

MOBPBaseAgent::MOBPBaseAgent(unsigned short int pos_, unsigned short int gender_, unsigned short int startingPos_, double nnWeightMin_, double nnWeightMax_, double noiseMean_, double noiseStdDev_) :
    _pos(pos_),
    _startingPos(startingPos_),
    _gender(gender_),
    _nnWeightMin(nnWeightMin_),
    _nnWeightMax(nnWeightMax_),
    _noiseMean(noiseMean_),
    _noiseStdDev(noiseStdDev_) {
        this->whichDomain = "MOBPDomain";
        this->policy = MOBPPolicy(1, nnWeightMin_, nnWeightMax_);
    }

MOBPBaseAgent::MOBPBaseAgent(const MOBPBaseAgent& other) :
    _pos(other._pos),
    _startingPos(other._startingPos),
    _gender(other._gender),
    _nnWeightMin(other._nnWeightMin),
    _nnWeightMax(other._nnWeightMax),
    _noiseMean(other._noiseMean),
    _noiseStdDev(other._noiseStdDev),
    whichDomain(other.whichDomain) {
        this->policy = *std::make_shared<MOBPPolicy>(other.policy);
    }

// update the agent's position respecting the env constraints
void MOBPBaseAgent::move(unsigned short int delta, Environment environment) {
    unsigned short int newPos = environment.moveAgent(this->_pos, delta);
    this->_pos = newPos;
}

// reset the agent's position to its starting position
void MOBPBaseAgent::reset() {
    this->_pos = this->_startingPos;
}

// return the current position of the agent 
unsigned short int MOBPBaseAgent::getPosition() const {
    return this -> _pos;
}

// forward pass through the policy
short int MOBPBaseAgent::forward(const std::vector<double>& input) {
    return policy.forward(input);
}

// return the agent's observations (which is just its current position)
unsigned short int MOBPBaseAgent::observe() {
    return this->getPosition();
}

// mutate the agent's policy a little bit
void MOBPBaseAgent::addNoiseToPolicy() {
    this->policy.addNoise(this->_noiseMean, this->_noiseStdDev);
}