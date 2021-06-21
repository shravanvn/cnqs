#ifndef CNQS_PDESOLVER_HAMILTONIAN_HPP
#define CNQS_PDESOLVER_HAMILTONIAN_HPP

#include <string>
#include <tuple>
#include <vector>

namespace cnqs {

namespace pdesolver {

/// @brief Quantum rotor hamiltonian
///
/// This class provides an implementation for encoding quantum rotor Hamiltonian
/// described by an undirected graph with vertices \f$\mathcal{V} = \{0, 1,
/// \ldots, d - 1\}\f$ and edges \f$\mathcal{E} \subseteq \{(j, k) \in
/// \mathcal{V} \times \mathcal{V}: j < k\}\f$. Each edge \f$(j, k)\f$ has an
/// associated weight \f$\beta_{jk} \in \mathbb{R}\f$ describing the strength of
/// interaction between rotors \f$j, k \in \mathcal{V}\f$.
class Hamiltonian {
public:
    /// @brief Construct Hamiltonian from node and edge specifications
    Hamiltonian(long num_rotor, double vertex_weight,
                const std::vector<std::tuple<long, long, double>> &edge_list);

    /// @brief Construct Hamiltonian from YAML formatted file
    Hamiltonian(const std::string &hamiltonian_file_name);

    /// @brief Default destructor
    ~Hamiltonian() = default;

    /// @brief Default copy constructor
    Hamiltonian(const Hamiltonian &) = default;

    /// @brief Default move constructor
    Hamiltonian(Hamiltonian &&) = default;

    /// @brief Default copy assignment
    Hamiltonian &operator=(const Hamiltonian &) = default;

    /// @brief Default move assignment
    Hamiltonian &operator=(Hamiltonian &&) = default;

    /// @brief Number of rotors in the quantum rotor network
    const long &NumRotor() const { return num_rotor_; }

    /// @brief Vertex weight of rotors in the quantum rotor network
    const double &VertexWeight() const { return vertex_weight_; }

    /// @brief List of edges in the quantum rotor network
    const std::vector<std::tuple<long, long, double>> &EdgeList() const {
        return edge_list_;
    }

    /// @brief Sum of edge weights
    double SumEdgeWeights() const;

    /// @brief Sum of absolute value of edge weights
    double SumAbsEdgeWeights() const;

private:
    long num_rotor_;
    double vertex_weight_;
    std::vector<std::tuple<long, long, double>> edge_list_;
};

}  // namespace pdesolver

}  // namespace cnqs

#endif
