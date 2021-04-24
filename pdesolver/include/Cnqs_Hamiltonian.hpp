#ifndef CNQS_HAMILTONIAN_HPP
#define CNQS_HAMILTONIAN_HPP

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace Cnqs {

// =============================================================================
// Declarations
// =============================================================================

/// Quantum rotor hamiltonian
///
/// This class provides an implementation for encoding quantum rotor Hamiltonian
/// described by an undirected graph with vertices \f$\mathcal{V} = \{0, 1,
/// \ldots, d - 1\}\f$ and edges \f$\mathcal{E} \subseteq \{(j, k) \in
/// \mathcal{V} \times \mathcal{V}: j < k\}\f$. Each edge \f$(j, k)\f$ has an
/// associated weight \f$\beta_{jk} \in \mathbb{R}\f$ describing the strength of
/// interaction between rotors \f$j, k \in \mathcal{V}\f$.
template <class Real, class Index>
class Hamiltonian {
public:
    /// Construct a new Hamiltonian object from node and edge specifications
    Hamiltonian(Index numRotor, Real vertexWeight,
                const std::vector<std::tuple<Index, Index, Real>> &edgeList);

    /// Construct a new Hamiltonian object from YAML formatted file
    Hamiltonian(const std::string &hamiltonianFileName);

    /// Number of rotors in the Hamiltonian network
    const Index &numRotor() const { return numRotor_; }

    /// Vertex weight of rotors in the Hamiltonian network
    const Real &vertexWeight() const { return vertexWeight_; }

    /// List of edges in the Hamiltonian network
    const std::vector<std::tuple<Index, Index, Real>> &edgeList() const {
        return edgeList_;
    }

    /// Sum of edge weights
    Real sumEdgeWeights() const;

    /// Sum of absolute value of edge weights
    Real sumAbsEdgeWeights() const;

private:
    Index numRotor_;
    Real vertexWeight_;
    std::vector<std::tuple<Index, Index, Real>> edgeList_;
};

// =============================================================================
// Implementations
// =============================================================================

template <class Real, class Index>
Hamiltonian<Real, Index>::Hamiltonian(
    Index numRotor, Real vertexWeight,
    const std::vector<std::tuple<Index, Index, Real>> &edgeList)
    : numRotor_(numRotor), vertexWeight_(vertexWeight), edgeList_(edgeList) {
    // validate inputs
    if (numRotor_ < 2) {
        throw std::invalid_argument("At least two rotors are required");
    }

    if (vertexWeight_ < static_cast<Real>(0)) {
        throw std::invalid_argument("Vertex weight must be non-negative");
    }

    for (auto &edge : edgeList_) {
        Index j = std::get<0>(edge);
        Index k = std::get<1>(edge);
        Real beta = std::get<2>(edge);

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

template <class Real, class Index>
Hamiltonian<Real, Index>::Hamiltonian(const std::string &hamiltonianFileName) {
    YAML::Node hamiltonian = YAML::LoadFile(hamiltonianFileName);

    numRotor_ = hamiltonian["num_rotor"].as<Index>();
    if (numRotor_ < 2) {
        throw std::invalid_argument("At least two rotors are required");
    }

    vertexWeight_ = hamiltonian["vertex_weight"].as<Real>();
    if (vertexWeight_ < static_cast<Real>(0)) {
        throw std::invalid_argument("Vertex weight must be non-negative");
    }

    const auto &edgeList = hamiltonian["edges"];
    if (!edgeList.IsSequence()) {
        throw std::invalid_argument(
            "The edges must be specified as a YAML array");
    }

    edgeList_.reserve(edgeList.size());
    for (const auto &edge : edgeList) {
        Index j = edge["j"].as<Index>();
        Index k = edge["k"].as<Index>();
        const Real beta = edge["beta"].as<Real>();

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

template <class Real, class Index>
Real Hamiltonian<Real, Index>::sumEdgeWeights() const {
    Real sum = static_cast<Real>(0);

    for (const auto &edge : edgeList_) {
        sum += std::get<2>(edge);
    }

    return sum;
}

template <class Real, class Index>
Real Hamiltonian<Real, Index>::sumAbsEdgeWeights() const {
    Real absSum = static_cast<Real>(0);

    for (const auto &edge : edgeList_) {
        absSum += std::abs(std::get<2>(edge));
    }

    return absSum;
}

}  // namespace Cnqs

#endif
