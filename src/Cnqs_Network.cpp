#include "Cnqs_Network.hpp"

#include <stdexcept>

Cnqs::Network::Network(
    int numRotor, const std::vector<std::tuple<int, int, double>> &edgeList)
    : numRotor_(numRotor), edgeList_(edgeList) {
    // validate inputs
    if (numRotor_ < 2) {
        throw std::domain_error(
            "==Cnqs::Problem::Problem== Need at least two quantum rotors");
    }

    for (auto &edge : edgeList_) {
        int j = std::get<0>(edge);
        int k = std::get<1>(edge);
        double g = std::get<2>(edge);

        if (j == k) {
            throw std::domain_error(
                "==Cnqs::Problem::Problem== Self-loops are not allowed in "
                "network");
        }

        // switch order to ensure j < k
        if (j > k) {
            int temp = j;
            j = k;
            k = temp;

            edge = std::make_tuple(j, k, g);
        }

        if (j < 0 || k >= numRotor_) {
            throw std::domain_error(
                "==Cnqs::Problem::Problem== Edge specification is not valid");
        }
    }
}

double Cnqs::Network::eigValLowerBound() const {
    double mu = -1.0e-09;

    for (const auto &edge : edgeList_) {
        mu -= std::abs(std::get<2>(edge));
    }

    return mu;
}

std::string Cnqs::Network::description() const {
    const int numEdge = edgeList_.size();

    std::string description;
    description += "{\n";
    description += "    \"num_rotor\": " + std::to_string(numRotor_) + ",\n";
    description += "    \"edges\": [\n";
    for (int i = 0; i < numEdge; ++i) {
        char buffer[81];
        std::sprintf(buffer, "{\"nodes\": [%d, %d], \"weight\": %.3e}",
                     std::get<0>(edgeList_[i]), std::get<1>(edgeList_[i]),
                     std::get<2>(edgeList_[i]));
        if (i < numEdge - 1) {
            description += "        " + std::string(buffer) + ",\n";
        } else {
            description += "        " + std::string(buffer) + "\n";
        }
    }
    description += "    ]\n";
    description += "}";

    return description;
}

std::ostream &operator<<(std::ostream &os, const Cnqs::Network &network) {
    os << network.description();
    return os;
}
