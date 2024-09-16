#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include "MOD.h" // Include your Evolutionary class header here
#include "NSGA_II.h"
#include "MOD_ablated.h"
#include "MOD_team_ablated.h"

// Function to get current date and time as a string
std::string getCurrentDateTimeString() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_time_t);
    
    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

int main(int argc, char* argv[]) {

    // Start the timer
    auto start = std::chrono::steady_clock::now();
    
    if (argc == 4) {
        std::string config_filename = argv[1];
        std::string data_filename_prefix = argv[2];
        std::string alg = argv[3];

        auto currentDateTimeString = getCurrentDateTimeString();

        if (alg == "nsga") {
            NSGA_II nsga(config_filename);
            nsga.evolve(config_filename, data_filename_prefix + alg + currentDateTimeString + ".csv");
        } else if (alg == "mod") {
            MOD evo(config_filename);
            evo.evolve(config_filename, data_filename_prefix + alg + currentDateTimeString + ".csv");
        } else if (alg == "mod_abl") {
            MODAblated abl(config_filename);
            abl.evolve(config_filename, data_filename_prefix + alg + currentDateTimeString + ".csv");
        } else if (alg == "mod_team_abl") {
            MODTeamAblated team_abl(config_filename);
            team_abl.evolve(config_filename, data_filename_prefix + alg + currentDateTimeString + ".csv");
        }
    } else {
        // Extract filename from command-line arguments
        std::string project_root = "/home/thakarr/D-MO-Operator/";
        std::string config_filename = project_root + "config/config.yaml";
        std::string data_filename_root = project_root + "experiments/data/mobpd_test_data/"; // Default data filename with current date and time
        // create a copy of the config file
        auto currentDateTimeString = getCurrentDateTimeString();

        std::ifstream configSrc(config_filename, std::ios::binary);
        std::ofstream configDst(data_filename_root + currentDateTimeString + "_config.yaml", std::ios::binary);
        configDst << configSrc.rdbuf();

        configSrc.close();
        configDst.close();

        MOD evo(config_filename);
        evo.evolve(config_filename, data_filename_root + currentDateTimeString + "_MOD_.csv");
        MODTeamAblated team_abl(config_filename);
        team_abl.evolve(config_filename, data_filename_root + currentDateTimeString + "_MOD_TEAM_ABLATED_.csv");
        MODAblated abl(config_filename);
        abl.evolve(config_filename, data_filename_root + currentDateTimeString + "_MOD_ABLATED_.csv");
        NSGA_II nsga(config_filename);
        nsga.evolve(config_filename, data_filename_root + currentDateTimeString + "_NSGA_II_.csv");
    }
    

    // End the timer
    auto end = std::chrono::steady_clock::now();

    // Calculate the elapsed time
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    // Print out the elapsed time
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}
