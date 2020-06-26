#ifndef CNQS_HAMILTONIAN_FOURIER_HPP
#define CNQS_HAMILTONIAN_FOURIER_HPP

#include <tuple>
#include <vector>

#include "cnqs_hamiltonian.hpp"
#include "cnqs_state.hpp"

class CnqsHamiltonianFourier : public CnqsHamiltonian {
  public:
    CnqsHamiltonianFourier(int d, int n,
                           std::vector<std::tuple<int, int>> edges, double g,
                           double J)
        : CnqsHamiltonian(d, n, edges, g, J), max_freq_((n - 1) / 2) {
        if (n % 2 == 0) {
            throw std::domain_error("Number of Fourier modes must be odd");
        }
    }

    ~CnqsHamiltonianFourier() = default;

    CnqsState initialize_state() const;

    CnqsState operator*(const CnqsState &state) const;

    int SquaredDistanceFromCenter(int linear_index) const;

  private:
    bool initial_state_index_qualifies(int linear_index) const;

    int max_freq_;
};

#endif
