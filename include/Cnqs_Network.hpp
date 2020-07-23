#ifndef CNQS_NETWORK_HPP
#define CNQS_NETWORK_HPP

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace Cnqs {

class Network {
public:
    Network(int numRotor,
            const std::vector<std::tuple<int, int, double>> &edgeList);

    int numRotor() const { return numRotor_; }

    const std::vector<std::tuple<int, int, double>> &edgeList() const {
        return edgeList_;
    }

    double eigValLowerBound() const;

    std::string description() const;

private:
    int numRotor_;
    std::vector<std::tuple<int, int, double>> edgeList_;
};

} // namespace Cnqs

std::ostream &operator<<(std::ostream &os, const Cnqs::Network &network);

#endif
