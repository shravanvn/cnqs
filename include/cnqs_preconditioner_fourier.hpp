#ifndef CNQS_PRECONDITIONER_FOURIER_HPP
#define CNQS_PRECONDITIONER_FOURIER_HPP

#include "cnqs_hamiltonian_fourier.hpp"
#include "cnqs_preconditioner.hpp"
#include "cnqs_state.hpp"

class CnqsPreconditionerFourier : public CnqsPreconditioner {
  public:
    CnqsPreconditionerFourier(const CnqsHamiltonianFourier *hamiltonian)
        : hamiltonian_(hamiltonian) {}

    ~CnqsPreconditionerFourier() = default;

    const CnqsState solve(const CnqsState &state) const;

    const CnqsState trans_solve(const CnqsState &state) const {
        return solve(state);
    }

  private:
    const CnqsHamiltonianFourier *hamiltonian_;
};

#endif
