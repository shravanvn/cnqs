#ifndef CNQS_NETWORK_HPP
#define CNQS_NETWORK_HPP

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

/// Quantum rotor network
///
/// This class provides an implementation for describing quantum rotor network
/// described by an undirected graph with vertices \f$\mathcal{V} = \{0, 1,
/// \ldots, d - 1\}\f$ and edges \f$\mathcal{E} \subseteq \{(j, k) \in
/// \mathcal{V} \times \mathcal{V}: j < k\}\f$. Each edge \f$(j, k)\f$ has an
/// associated weight \f$\beta_{jk} \in \mathbb{R}\f$ describing the strength of
/// interaction between rotors \f$j, k \in \mathcal{V}\f$.
template <class Real, class Index>
class Network {
public:
    /// Construct a new Network object from node and edge specifications
    Network(Index numRotor,
            const std::vector<std::tuple<Index, Index, Real>> &edgeList);

    /// Construct a new Network object from YAML formatted file
    Network(const std::string &networkFileName);

    /// Number of rotors in the network
    const Index &numRotor() const { return numRotor_; }

    /// List of edges in the network
    const std::vector<std::tuple<Index, Index, Real>> &edgeList() const {
        return edgeList_;
    }

    /// Sum of edge weights
    Real sumWeights() const;

    /// Sum of absolute value of edge weights
    Real sumAbsWeights() const;

private:
    Index numRotor_;
    std::vector<std::tuple<Index, Index, Real>> edgeList_;
};

// =============================================================================
// Implementations
// =============================================================================

template <class Real, class Index>
Network<Real, Index>::Network(
    Index numRotor, const std::vector<std::tuple<Index, Index, Real>> &edgeList)
    : numRotor_(numRotor), edgeList_(edgeList) {
    // validate inputs
    if (numRotor_ < 2) {
        throw std::invalid_argument("At least two rotors are required");
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
Network<Real, Index>::Network(const std::string &networkFileName) {
    YAML::Node network = YAML::LoadFile(networkFileName);

    const auto &edgeList = network["edges"];
    if (!edgeList.IsSequence()) {
        throw std::invalid_argument(
            "The edges must be specified as a YAML array");
    }

    numRotor_ = network["num_rotor"].as<Index>();
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
Real Network<Real, Index>::sumWeights() const {
    Real sum = static_cast<Real>(0);

    for (const auto &edge : edgeList_) {
        sum += std::get<2>(edge);
    }

    return sum;
}

template <class Real, class Index>
Real Network<Real, Index>::sumAbsWeights() const {
    Real absSum = static_cast<Real>(0);

    for (const auto &edge : edgeList_) {
        absSum += std::abs(std::get<2>(edge));
    }

    return absSum;
}

}  // namespace Cnqs

#endif
