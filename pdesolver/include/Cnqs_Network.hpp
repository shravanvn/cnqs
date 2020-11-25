#ifndef CNQS_NETWORK_HPP
#define CNQS_NETWORK_HPP

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace Cnqs {

/**
 * @brief Quantum rotor network
 *
 * This class provides an implementation for describing quantum rotor network
 * described by an undirected graph with vertices \f$\mathcal{V} = \{0, 1,
 * \ldots, d - 1\}\f$ and edges \f$\mathcal{E} \subseteq \{(j, k) \in
 * \mathcal{V} \times \mathcal{V}: j \neq k\}\f$. Each edge \f$(j, k)\f$ has an
 * associated weight \f$\beta_{jk} \in \mathbb{R}\f$ describing the strength of
 * interaction between rotors \f$j, k \in \mathcal{V}\f$.
 */
template <class Scalar, class Index>
class Network {
public:
    /**
     * @brief Construct a new Network object from node and edge specifications
     *
     * @param [in] numRotor Number of rotors \f$d\f$.
     * @param [in] edgeList Edges connecting the rotors. Each edge is specified
     * as \f$(j, k, w_{jk})\f$ where \f$0 \leq j, k \leq d - 1\f$ are the edge
     * nodes and \f$\beta_{jk} \in \mathbb{R}\f$ is the edge weight.
     */
    Network(Index numRotor,
            const std::vector<std::tuple<Index, Index, Scalar>> &edgeList);

    /**
     * @brief Construct a new Network object from JSON formatted file
     *
     * @param [in] networkFileName JSON file name.
     */
    Network(const std::string &networkFileName);

    /**
     * @brief Number of rotors in the network
     *
     * @return Number of rotors \f$d\f$.
     */
    const Index &numRotor() const { return numRotor_; }

    /**
     * @brief Edge specifications of the network
     *
     * @return Edge list; each element of the list is \f$(j, k, \beta_{jk})\f$.
     */
    const std::vector<std::tuple<Index, Index, Scalar>> &edgeList() const {
        return edgeList_;
    }

    /**
     * @brief Lower bound on minimial eigenvalue of associated Hamiltonian
     *
     * Given a rotor with \f$d\f$ rotors and edges \f$\mathcal{E}\f$, one can
     * define an associated Hamiltonian on the Hilbert space
     * \f$\mathcal{H}^1([0, 2\pi]^d)\f$ as follows
     *
     * \f[
     *     H \psi(\theta) = -\frac{1}{2} \sum_{j = 0}^{d - 1}
     *     \frac{\partial^2 \psi}{\partial \theta_j^2} (\theta) -
     *     \sum_{(j, k) \in \mathcal{E}} \beta_{jk} \cos(\theta_j - \theta_k)
     *     \psi(\theta)
     * \f]
     *
     * This function returns a lower bound for the minimal eigenvalue of this
     * operator.
     *
     * @return Lower bound for eigenvalue of associated Hamiltonian.
     */
    Scalar eigValLowerBound() const;

    /**
     * @brief Describe the quantum rotor network in JSON format
     *
     * @return Description of the network.
     */
    nlohmann::json description() const;

    /**
     * @brief Print Network object to output streams
     *
     * @param [in,out] os Output stream
     * @param [in] network Quanturm rotor network object
     * @return Output stream
     */
    friend std::ostream &operator<<(std::ostream &os, const Network &network) {
        os << network.description().dump(4);
        return os;
    }

private:
    Index numRotor_;
    std::vector<std::tuple<Index, Index, Scalar>> edgeList_;
};

template <class Scalar, class Index>
Network<Scalar, Index>::Network(
    Index numRotor,
    const std::vector<std::tuple<Index, Index, Scalar>> &edgeList)
    : numRotor_(numRotor), edgeList_(edgeList) {
    // validate inputs
    if (numRotor_ < 2) {
        throw std::domain_error(
            "==Cnqs::Problem::Problem== Need at least two quantum rotors");
    }

    for (auto &edge : edgeList_) {
        Index j = std::get<0>(edge);
        Index k = std::get<1>(edge);
        Scalar beta = std::get<2>(edge);

        if (j == k) {
            throw std::domain_error(
                "==Cnqs::Problem::Problem== Self-loops are not allowed in "
                "network");
        }

        // switch order to ensure j < k
        if (j > k) {
            Index temp = j;
            j = k;
            k = temp;

            edge = std::make_tuple(j, k, beta);
        }

        if (j < 0 || k >= numRotor_) {
            throw std::domain_error(
                "==Cnqs::Problem::Problem== Edge specification is not valid");
        }
    }
}

template <class Scalar, class Index>
Network<Scalar, Index>::Network(const std::string &networkFileName) {
    nlohmann::json description;

    {
        std::ifstream networkFile(networkFileName);
        networkFile >> description;
    }

    const auto &edgeList = description["edges"];

    numRotor_ = description["num_rotor"];
    edgeList_.reserve(edgeList.size());

    for (const auto &edge : edgeList) {
        Index j = edge["j"];
        Index k = edge["k"];
        const Scalar beta = edge["beta"];

        // switch order to ensure j < k
        if (j > k) {
            const Index temp = j;
            j = k;
            k = temp;
        }

        if (j < 0 || k >= numRotor_) {
            throw std::domain_error(
                "==Cnqs::Problem::Problem== Edge specification is not valid");
        }

        edgeList_.emplace_back(j, k, beta);
    }
}

template <class Scalar, class Index>
Scalar Network<Scalar, Index>::eigValLowerBound() const {
    Scalar mu = 1.0e-09;

    for (const auto &edge : edgeList_) {
        mu += std::abs(std::get<2>(edge));
    }

    return -3.0 * mu;
}

template <class Scalar, class Index>
nlohmann::json Network<Scalar, Index>::description() const {
    nlohmann::json description;

    description["num_rotor"] = numRotor_;
    description["edges"] = nlohmann::json::array();

    for (const auto &edge : edgeList_) {
        nlohmann::json edgeJson;

        edgeJson["j"] = std::get<0>(edge);
        edgeJson["k"] = std::get<1>(edge);
        edgeJson["beta"] = std::get<2>(edge);

        description["edges"].push_back(edgeJson);
    }

    return description;
}

}  // namespace Cnqs

#endif
