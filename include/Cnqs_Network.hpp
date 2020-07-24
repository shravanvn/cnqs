#ifndef CNQS_NETWORK_HPP
#define CNQS_NETWORK_HPP

#include <iostream>
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
 * associated weight \f$g_{jk} \in \mathbb{R}\f$ describing the strength of
 * interaction between rotors \f$j, k \in \mathcal{V}\f$.
 */
class Network {
public:
    /**
     * @brief Construct a new Network object from node and edge specifications
     *
     * @param [in] numRotor Number of rotors \f$d\f$.
     * @param [in] edgeList Edges connecting the rotors. Each edge is specified
     * as \f$(j, k, w_{jk})\f$ where \f$0 \leq j, k \leq d - 1\f$ are the edge
     * nodes and \f$g_{jk} \in \mathbb{R}\f$ is the edge weight.
     */
    Network(int numRotor,
            const std::vector<std::tuple<int, int, double>> &edgeList);

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
    const int &numRotor() const { return numRotor_; }

    /**
     * @brief Edge specifications of the network
     *
     * @return Edge list; each element of the list is \f$(j, k, g_{jk})\f$.
     */
    const std::vector<std::tuple<int, int, double>> &edgeList() const {
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
     *     \sum_{(j, k) \in \mathcal{E}} g_{jk} \cos(\theta_j - \theta_k)
     *     \psi(\theta)
     * \f]
     *
     * This function returns a lower bound for the minimal eigenvalue of this
     * operator.
     *
     * @return Lower bound for eigenvalue of associated Hamiltonian.
     */
    double eigValLowerBound() const;

    /**
     * @brief Describe the quantum rotor network in JSON format
     *
     * @return Description of the network.
     */
    std::string description() const;

private:
    int numRotor_;
    std::vector<std::tuple<int, int, double>> edgeList_;
};

} // namespace Cnqs

/**
 * @brief Write quantum rotor network to output streams
 *
 * @param [in,out] os Output stream.
 * @param [in] network Quantum rotor network.
 * @return Output stream.
 */
std::ostream &operator<<(std::ostream &os, const Cnqs::Network &network);

#endif
