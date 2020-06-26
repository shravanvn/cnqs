#include "cnqs_hamiltonian_direct.hpp"

#include <cmath>

#include "utils.hpp"

CnqsState CnqsHamiltonianDirect::initialize_state() const {
    CnqsState state(num_element_);
    state = 1.0;

    for (int j = 0; j < d_; ++j) {
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
        int n_lj = IntPow(n_, j);

        // index i_j := i[j]
        //      multiplicative factor in unwrapping: f_j = n[<j]
        //      max value: n_j := n[j]
        int f_j = n_lj;
        int n_j = n_;

        // index i_gj := i[>j]
        //      multiplicative factor in unwrapping: f_gj = f[>j]
        //      max value: n_gj := n[>j]
        int f_gj = f_j * n_j;
        int n_gj = num_element_ / f_gj;

        // update values
        for (int i_gj = 0; i_gj < n_gj; ++i_gj) {
            for (int i_j = 0; i_j < n_j; ++i_j) {
                for (int i_lj = 0; i_lj < n_lj; ++i_lj) {
                    state(i_lj + f_j * i_j + f_gj * i_gj) *=
                        std::cos(theta_[i_j]);
                }
            }
        }
    }

    return state;
}

CnqsState CnqsHamiltonianDirect::operator*(const CnqsState &state) const {
    CnqsState new_state(num_element_);

    // differential operator
    double h = theta_[1] - theta_[0];
    double fact = g_ * J_ / (24.0 * h * h);

    for (int j = 0; j < d_; ++j) {
        int n_lj = IntPow(n_, j);

        int f_j = n_lj;
        int n_j = n_;

        int f_gj = f_j * n_j;
        int n_gj = num_element_ / f_gj;

        for (int i_gj = 0; i_gj < n_gj; ++i_gj) {
            for (int i_j = 0; i_j < n_j; ++i_j) {
                int i_j_2m = (i_j - 2 >= 0) ? i_j - 2 : n_ + i_j - 2;
                int i_j_1m = (i_j - 1 >= 0) ? i_j - 1 : n_ + i_j - 1;
                int i_j_1p = (i_j + 1 < n_) ? i_j + 1 : i_j + 1 - n_;
                int i_j_2p = (i_j + 2 < n_) ? i_j + 2 : i_j + 2 - n_;

                for (int i_lj = 0; i_lj < n_lj; ++i_lj) {
                    int i_2m = i_lj + f_j * i_j_2m + f_gj * i_gj;
                    int i_1m = i_lj + f_j * i_j_1m + f_gj * i_gj;
                    int i = i_lj + f_j * i_j + f_gj * i_gj;
                    int i_1p = i_lj + f_j * i_j_1p + f_gj * i_gj;
                    int i_2p = i_lj + f_j * i_j_2p + f_gj * i_gj;

                    new_state(i) -= fact * (-state(i_2m) + 16.0 * state(i_1m) -
                                            30.0 * state(i) +
                                            16.0 * state(i_1p) - state(i_2p));
                }
            }
        }
    }

    // multiplication with the cross-terms
    for (int e = 0; e < num_edge_; ++e) {
        // select the endpoints (j, k) of an edge, ensuring j < k
        int j = std::get<0>(edges_[e]);
        int k = std::get<1>(edges_[e]);

        if (j > k) {
            int temp = j;
            j = k;
            k = temp;
        }

        // split indices into five groups: i[<j], i[j], i[j<<k], i[k], i[>k]
        int n_lj = IntPow(n_, j);

        int f_j = n_lj;
        int n_j = n_;

        int f_jk = f_j * n_j;
        int n_jk = IntPow(n_, k - j - 1);

        int f_k = f_jk * n_jk;
        int n_k = n_;

        int f_gk = f_k * n_k;
        int n_gk = num_element_ / f_gk;

        // update new state
        for (int i_gk = 0; i_gk < n_gk; ++i_gk) {
            for (int i_k = 0; i_k < n_k; ++i_k) {
                for (int i_jk = 0; i_jk < n_jk; ++i_jk) {
                    for (int i_j = 0; i_j < n_j; ++i_j) {
                        for (int i_lj = 0; i_lj < n_lj; ++i_lj) {
                            int i = i_lj + f_j * i_j + f_jk * i_jk + f_k * i_k +
                                    f_gk * i_gk;
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
