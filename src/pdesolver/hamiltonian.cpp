#include "cnqs/pdesolver/hamiltonian.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <stdexcept>

cnqs::pdesolver::Hamiltonian::Hamiltonian(
    long numRotor, double vertexWeight,
    const std::vector<std::tuple<long, long, double>> &edgeList)
    : numRotor_(numRotor), vertexWeight_(vertexWeight), edgeList_(edgeList) {
    // validate inputs
    if (numRotor_ < 2) {
        throw std::invalid_argument("At least two rotors are required");
    }

    if (vertexWeight_ < static_cast<double>(0)) {
        throw std::invalid_argument("Vertex weight must be non-negative");
    }

    for (auto &edge : edgeList_) {
        long j = std::get<0>(edge);
        long k = std::get<1>(edge);
        double beta = std::get<2>(edge);

        if (j == k) {
            throw std::invalid_argument("Self-loops are not allowed");
        }

        // switch order to ensure j < k
        if (j > k) {
            std::swap(j, k);
            edge = std::make_tuple(j, k, beta);
        }

        if (j < 0 || k >= numRotor_) {
            throw std::invalid_argument("Edge specification is not valid");
        }
    }
}

cnqs::pdesolver::Hamiltonian::Hamiltonian(
    const std::string &hamiltonianFileName) {
    YAML::Node hamiltonian = YAML::LoadFile(hamiltonianFileName);

    numRotor_ = hamiltonian["num_rotor"].as<long>();
    if (numRotor_ < 2) {
        throw std::invalid_argument("At least two rotors are required");
    }

    vertexWeight_ = hamiltonian["vertex_weight"].as<double>();
    if (vertexWeight_ < static_cast<double>(0)) {
        throw std::invalid_argument("Vertex weight must be non-negative");
    }

    const auto &edgeList = hamiltonian["edges"];
    if (!edgeList.IsSequence()) {
        throw std::invalid_argument(
            "The edges must be specified as a YAML array");
    }

    edgeList_.reserve(edgeList.size());
    for (const auto &edge : edgeList) {
        long j = edge["j"].as<long>();
        long k = edge["k"].as<long>();
        const double beta = edge["beta"].as<double>();

        // switch order to ensure j < k
        if (j > k) {
            std::swap(j, k);
        }

        if (j < 0 || k >= numRotor_) {
            throw std::invalid_argument("Edge specification is not valid");
        }

        edgeList_.emplace_back(j, k, beta);
    }
}

double cnqs::pdesolver::Hamiltonian::sumEdgeWeights() const {
    double sum = static_cast<double>(0);

    for (const auto &edge : edgeList_) {
        sum += std::get<2>(edge);
    }

    return sum;
}

double cnqs::pdesolver::Hamiltonian::sumAbsEdgeWeights() const {
    double absSum = static_cast<double>(0);

    for (const auto &edge : edgeList_) {
        absSum += std::abs(std::get<2>(edge));
    }

    return absSum;
}
