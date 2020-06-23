#include "cnqs_hamiltonian.hpp"

unsigned long powint(unsigned long n, unsigned long d) {
    if (d == 0) {
        return 1;
    }

    if (d == 1) {
        return n;
    }

    return powint(n, d / 2) * powint(n, d - d / 2);
}

CnqsState CnqsHamiltonian::initialize_state() const {
    unsigned long num_element = powint(n_, d_);

    CnqsState state(num_element);
    state = 1.0;

    const double PI = 4.0 * std::atan(1.0);

    for (unsigned long j = 0; j < d_; ++j) {
        // ======================
        // unwrapping the indices
        // ======================
        //
        // split the d indices into three groups:
        //
        //      state[i[0], ..., i[d - 1]] = state[i[<j], i[j], i[>j]]
        //
        // with
        //
        //      i[<j] = (    i[0], ..., i[j - 1])
        //      i[>j] = (i[j + 1], ..., i[d - 1])
        //
        // in column-major format, this (i[<j], i[j], i[>j]) index flattens to
        //
        //       i[<j] + f[j] * i[j] + f[>j] * i[>j]
        //
        // with
        //
        //      0 <= i[<j] < n[<j]
        //      0 <=  i[j] <  n[j],     f[j] = n[<j]
        //      0 <= i[>j] < n[>j],    f[>j] = n[<j] * n[j]
        //
        // and
        //
        //      n[<j] = n[0] * ... * n[j - 1]
        //      n[>j] = n[j + 1] * ... * n[d - 1] = num_elements(state) / f[>j]
        // ======================

        // index i_lj := i[<j]
        //      max value: n_lj := n[<j]

        unsigned long n_lj = powint(n_, j);

        // index i_j := i[j]
        //      multiplicative factor in unwrapping: f_j = n[<j]
        //      max value: n_j := n[j]

        unsigned long f_j = n_lj;
        unsigned long n_j = n_;

        // index i_gj := i[>j]
        //      multiplicative factor in unwrapping: f_gj = f[>j]
        //      max value: n_gj := n[>j]

        unsigned long f_gj = powint(n_, j + 1);
        unsigned long n_gj = num_element / f_gj;

        // update values

        for (unsigned long i_gj = 0; i_gj < n_gj; ++i_gj) {
            for (unsigned long i_j = 0; i_j < n_j; ++i_j) {
                double sin_factor = std::sin((2.0 * PI * i_j) / n_j);

                for (unsigned long i_lj = 0; i_lj < n_lj; ++i_lj) {
                    state(i_lj + f_j * i_j + f_gj * i_gj) *= sin_factor;
                }
            }
        }
    }

    return state;
}

CnqsState CnqsHamiltonian::operator*(const CnqsState &state) const {
    return state;
}
