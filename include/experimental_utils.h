#ifndef EXPERIMENTALUTILS_H
#define EXPERIMENTALUTILS_H

#include <vector>
#include <any>
#include "json.hpp"

class ExperimentalUtils {
private:
    // generic component. several of these can make up one logging action
    struct loggingComponent {
        std::string label;
        u_int level;
        std::vector<std::any> data; // self-explanatory
        std::vector<std::any> children; // can contain more logging components to create a clear heirarchy of data
    };
public:
    loggingComponent gen; // will contain all the required logging data for that generation
    ExperimentalUtils(); // constructor
    void add(int level_=0, std::string label_, std::vector<std::any> data_); // insert the data into the logging component (domain-specific)
};

#endif // EXPERIMENTALUTILS_H 