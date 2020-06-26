#ifndef CNQS_HAMILTONIAN_DIRECT_HPP
#define CNQS_HAMILTONIAN_DIRECT_HPP

#include <tuple>
#include <vector>

#include "cnqs_hamiltonian.hpp"
#include "cnqs_state.hpp"

class CnqsHamiltonianDirect : public CnqsHamiltonian {
  public:
    CnqsHamiltonianDirect(int d, int n, std::vector<std::tuple<int, int>> edges,
                          double g, double J)
        : CnqsHamiltonian(d, n, edges, g, J) {}

    ~CnqsHamiltonianDirect() = default;

    CnqsState initialize_state() const;

    CnqsState operator*(const CnqsState &state) const;
};

#endif
