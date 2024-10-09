#ifndef DNSGA_H
#define DNSGA_H

#include <string>
#include <vector>

class DNSGA;

class DNSGA {
    std::vector<int> generateUniqueRandomIntegers(int N);
public:
    DNSGA(int d);
    void evolve(const std::string & filename, const std::string& data_filename);
};

#endif