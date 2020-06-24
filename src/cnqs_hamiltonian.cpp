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

CnqsHamiltonian::CnqsHamiltonian(
    unsigned long d, unsigned long n,
    std::vector<const std::tuple<unsigned long, unsigned long>> edges, double g,
    double J)
    : d_(d), n_(n), edges_(edges), g_(g), J_(J), num_element_(powint(n_, d_)),
      num_edge_(edges_.size()), theta_(std::vector<double>(n)) {
    // compute 2 * pi
    const double TWO_PI = 8.0 * std::atan(1.0);

    // assign theta values
    for (unsigned long i = 0; i < n_; ++i) {
        theta_[i] = (TWO_PI * i) / n_;
    }
}

CnqsState CnqsHamiltonian::initialize_state() const {
    CnqsState state(num_element_);
    state = 1.0;

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

        unsigned long f_gj = f_j * n_j;
        unsigned long n_gj = num_element_ / f_gj;

        // update values

        for (unsigned long i_gj = 0; i_gj < n_gj; ++i_gj) {
            for (unsigned long i_j = 0; i_j < n_j; ++i_j) {
                double sin_factor = std::sin(theta_[i_j]);

                for (unsigned long i_lj = 0; i_lj < n_lj; ++i_lj) {
                    state(i_lj + f_j * i_j + f_gj * i_gj) *= sin_factor;
                }
            }
        }
    }

    return state;
}

CnqsState CnqsHamiltonian::operator*(const CnqsState &state) const {
    CnqsState new_state(num_element_);

    // differential operator

    double fact = g_ * J_ * n_ * n_ / 24.0;

    for (unsigned long j = 0; j < d_; ++j) {
        unsigned long n_lj = powint(n_, j);

        unsigned long f_j = n_lj;
        unsigned long n_j = n_;

        unsigned long f_gj = f_j * n_j;
        unsigned long n_gj = num_element_ / f_gj;

        for (unsigned long i_gj = 0; i_gj < n_gj; ++i_gj) {
            for (unsigned long i_j = 0; i_j < n_j; ++i_j) {
                unsigned long i_j_2m = (i_j - 2) % n_;
                unsigned long i_j_1m = (i_j - 1) % n_;
                unsigned long i_j_1p = (i_j + 1) % n_;
                unsigned long i_j_2p = (i_j + 2) % n_;

                for (unsigned long i_lj = 0; i_lj < n_lj; ++i_lj) {
                    unsigned long i_2p = i_lj + f_j * i_j_2p + f_gj * i_gj;
                    unsigned long i_1p = i_lj + f_j * i_j_1p + f_gj * i_gj;
                    unsigned long i = i_lj + f_j * i_j + f_gj * i_gj;
                    unsigned long i_1m = i_lj + f_j * i_j_1m + f_gj * i_gj;
                    unsigned long i_2m = i_lj + f_j * i_j_2m + f_gj * i_gj;

                    new_state(i) -= fact * (-state(i_2p) + 16.0 * state(i_1p) -
                                            30.0 * state(i) +
                                            16.0 * state(i_1m) - state(i_2m));
                }
            }
        }
    }

    // multiplication with the cross-terms

    for (unsigned long e = 0; e < num_edge_; ++e) {

        // select the endpoints (j, k) of an edge, ensuring j < k

        unsigned long j = std::get<0>(edges_[e]);
        unsigned long k = std::get<1>(edges_[e]);

        if (j > k) {
            unsigned long temp = j;
            j = k;
            k = temp;
        }

        // split indices into five groups: i[<j], i[j], i[j<<k], i[k], i[>k]

        unsigned long n_lj = powint(n_, j);

        unsigned long f_j = n_lj;
        unsigned long n_j = n_;

        unsigned long f_jk = f_j * n_j;
        unsigned long n_jk = powint(n_, k - j - 1);

        unsigned long f_k = f_jk * n_jk;
        unsigned long n_k = n_;

        unsigned long f_gk = f_k * n_k;
        unsigned long n_gk = num_element_ / f_gk;

        // update new state

        for (unsigned long i_gk = 0; i_gk < n_gk; ++i_gk) {
            for (unsigned long i_k = 0; i_k < n_k; ++i_k) {
                for (unsigned long i_jk = 0; i_jk < n_jk; ++i_jk) {
                    for (unsigned long i_j = 0; i_j < n_j; ++i_j) {
                        for (unsigned long i_lj = 0; i_lj < n_lj; ++i_lj) {
                            unsigned long i = i_lj + f_j * i_j + f_jk * i_jk +
                                              f_k * i_k + f_gk * i_gk;
                            new_state(i) -=
                                J_ * std::cos(theta_[i_j] - theta_[i_k]) *
                                state(i);
                        }
                    }
                }
            }
        }
    }

    return new_state;
}
