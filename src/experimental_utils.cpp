#include "experimental_utils.h"

// initialise this so that the gen logging component is setup
ExperimentalUtils::ExperimentalUtils() {
    gen.label = "Generation";
    gen.level = 0;
}

// insert the data into the logging component (hierarchical, domain-specific)
void ExperimentalUtils::add(int level_=0, std::string label_, std::vector<std::any> data_) {
    // create a logging components out of the raw data
    loggingComponent newDataPiece{label_, level_, data_};
}