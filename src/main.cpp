#include <iostream>
#include <chrono>
#include "MOD.h" // Include your Evolutionary class header here
#include "NSGA_II.h"

int main(int argc, char* argv[]) {
    /*
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <config_filename>" << " <data_filename>" << std::endl;
        return 1; // Return error if the filename is not provided
    }
    */

    // Extract filename from command-line arguments
    std::string filename = "/home/raghav/research/multiobjective_diff_evals/MOD/config/config.yaml";

    // Start the timer
    auto start = std::chrono::steady_clock::now();

    NSGA_II nsga(filename);
    // nsga.evolve(filename);

    // Run your evolutionary algorithm
    MOD evo(filename);
    evo.evolve(filename);

    // End the timer
    auto end = std::chrono::steady_clock::now();

    // Calculate the elapsed time
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    // Print out the elapsed time
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}