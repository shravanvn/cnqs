#include "cnqs_hamiltonian_fourier.hpp"

#include <cmath>

#include "utils.hpp"

bool CnqsHamiltonianFourier::initial_state_index_qualifies(
    int linear_index) const {
    for (int i = 0; i < d_; ++i) {
        int dim_index = linear_index % n_;

        if ((dim_index != max_freq_ - 1) && (dim_index != max_freq_ + 1)) {
            return false;
        }

        linear_index /= n_;
    }

    return true;
}

CnqsState CnqsHamiltonianFourier::initialize_state() const {
    CnqsState state(num_element_);

    double value = std::pow(4.0 * std::atan(1.0), d_);

    for (int i = 0; i < num_element_; ++i) {
        if (initial_state_index_qualifies(i)) {
            state(i) = value;
        }
    }

    return state;
}

CnqsState CnqsHamiltonianFourier::operator*(const CnqsState &state) const {
    CnqsState new_state(num_element_);

    // differential part
    double fact = 0.5 * g_ * J_;

    for (int i = 0; i < num_element_; ++i) {
        new_state(i) += fact * SquaredDistanceFromCenter(i) * state(i);
    }

    // mutliplication with cross terms
    fact = -0.5 * J_;
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
                            int i_jp = i_j + 1;
                            int i_km = i_k - 1;
                            double s_jpkm = 0.0;
                            if (i_jp < n_j && i_km >= 0) {
                                s_jpkm = state(i_lj + f_j * i_jp + f_jk * i_jk +
                                               f_k * i_km + f_gk * i_gk);
                            }

                            int i_jm = i_j - 1;
                            int i_kp = i_k + 1;
                            double s_jmkp = 0.0;
                            if (i_jm >= 0 && i_kp < n_k) {
                                s_jmkp = state(i_lj + f_j * i_jm + f_jk * i_jk +
                                               f_k * i_kp + f_gk * i_gk);
                            }

                            new_state(i_lj + f_j * i_j + f_jk * i_jk +
                                      f_k * i_k + f_gk * i_gk) +=
                                fact * (s_jpkm + s_jmkp);
                        }
                    }
                }
            }
        }
    }

    return new_state;
}

int CnqsHamiltonianFourier::SquaredDistanceFromCenter(int linear_index) const {
    int squared_distance = 0;

    for (int i = 0; i < d_; ++i) {
        squared_distance += IntPow(linear_index % n_ - max_freq_, 2);
        linear_index /= n_;
    }

    return squared_distance;
}
