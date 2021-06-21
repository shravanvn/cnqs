#include "cnqs/pdesolver/hamiltonian.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <stdexcept>

cnqs::pdesolver::Hamiltonian::Hamiltonian(
    long num_rotor, double vertex_weight,
    const std::vector<std::tuple<long, long, double>> &edge_list)
    : num_rotor_(num_rotor),
      vertex_weight_(vertex_weight),
      edge_list_(edge_list) {
    // validate inputs
    if (num_rotor_ < 2) {
        throw std::invalid_argument("At least two rotors are required");
    }

    if (vertex_weight_ < 0.0) {
        throw std::invalid_argument("Vertex weight must be non-negative");
    }

    for (auto &edge : edge_list_) {
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

        if (j < 0 || k >= num_rotor_) {
            throw std::invalid_argument("Edge specification is not valid");
        }
    }
}

cnqs::pdesolver::Hamiltonian::Hamiltonian(
    const std::string &hamiltonian_file_name) {
    YAML::Node hamiltonian = YAML::LoadFile(hamiltonian_file_name);

    num_rotor_ = hamiltonian["num_rotor"].as<long>();
    if (num_rotor_ < 2) {
        throw std::invalid_argument("At least two rotors are required");
    }

    vertex_weight_ = hamiltonian["vertex_weight"].as<double>();
    if (vertex_weight_ < 0.0) {
        throw std::invalid_argument("Vertex weight must be non-negative");
    }

    const auto &edge_list = hamiltonian["edges"];
    if (!edge_list.IsSequence()) {
        throw std::invalid_argument(
            "The edges must be specified as a YAML array");
    }

    edge_list_.reserve(edge_list.size());
    for (const auto &edge : edge_list) {
        long j = edge["j"].as<long>();
        long k = edge["k"].as<long>();
        const double beta = edge["beta"].as<double>();

        // switch order to ensure j < k
        if (j > k) {
            std::swap(j, k);
        }

        if (j < 0 || k >= num_rotor_) {
            throw std::invalid_argument("Edge specification is not valid");
        }

        edge_list_.emplace_back(j, k, beta);
    }
}

double cnqs::pdesolver::Hamiltonian::SumEdgeWeights() const {
    double sum = 0.0;

    for (const auto &edge : edge_list_) {
        sum += std::get<2>(edge);
    }

    return sum;
}

double cnqs::pdesolver::Hamiltonian::SumAbsEdgeWeights() const {
    double abs_sum = 0.0;

    for (const auto &edge : edge_list_) {
        abs_sum += std::abs(std::get<2>(edge));
    }

    return abs_sum;
}
