#include "cnqs_preconditioner_fourier.hpp"

const CnqsState CnqsPreconditionerFourier::solve(const CnqsState &state) const {
    int size = state.size();
    CnqsState new_state(size);

    double fact = 0.5 * hamiltonian_->g() * hamiltonian_->J();

    for (int i = 0; i < size; ++i) {
        new_state(i) =
            state(i) /
            (fact * hamiltonian_->SquaredDistanceFromCenter(i) + 2.0 * fact);
    }

    return new_state;
}
