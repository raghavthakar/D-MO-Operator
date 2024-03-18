#include <iostream>
#include <chrono>
#include "evolutionary.h"

int main() {
    // Start the timer
    auto start = std::chrono::steady_clock::now();

    // Run your evolutionary algorithm
    Evolutionary evo("../config/config.yaml");
    evo.evolve("../config/config.yaml");

    // End the timer
    auto end = std::chrono::steady_clock::now();

    // Calculate the elapsed time
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    // Print out the elapsed time
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}