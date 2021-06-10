#ifndef CNQS_PDESOLVER_HAMILTONIAN_HPP
#define CNQS_PDESOLVER_HAMILTONIAN_HPP

#include <string>
#include <tuple>
#include <vector>

namespace cnqs {

namespace pdesolver {

/// Quantum rotor hamiltonian
///
/// This class provides an implementation for encoding quantum rotor Hamiltonian
/// described by an undirected graph with vertices \f$\mathcal{V} = \{0, 1,
/// \ldots, d - 1\}\f$ and edges \f$\mathcal{E} \subseteq \{(j, k) \in
/// \mathcal{V} \times \mathcal{V}: j < k\}\f$. Each edge \f$(j, k)\f$ has an
/// associated weight \f$\beta_{jk} \in \mathbb{R}\f$ describing the strength of
/// interaction between rotors \f$j, k \in \mathcal{V}\f$.
class Hamiltonian {
public:
    /// Construct a new Hamiltonian object from node and edge specifications
    Hamiltonian(long numRotor, double vertexWeight,
                const std::vector<std::tuple<long, long, double>> &edgeList);

    /// Construct a new Hamiltonian object from YAML formatted file
    Hamiltonian(const std::string &hamiltonianFileName);

    /// Number of rotors in the Hamiltonian network
    const long &numRotor() const { return numRotor_; }

    /// Vertex weight of rotors in the Hamiltonian network
    const double &vertexWeight() const { return vertexWeight_; }

    /// List of edges in the Hamiltonian network
    const std::vector<std::tuple<long, long, double>> &edgeList() const {
        return edgeList_;
    }

    /// Sum of edge weights
    double sumEdgeWeights() const;

    /// Sum of absolute value of edge weights
    double sumAbsEdgeWeights() const;

private:
    long numRotor_;
    double vertexWeight_;
    std::vector<std::tuple<long, long, double>> edgeList_;
};

}  // namespace pdesolver

}  // namespace cnqs

#endif
