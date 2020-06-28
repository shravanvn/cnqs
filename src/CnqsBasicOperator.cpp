#include "CnqsBasicOperator.hpp"

CnqsBasicOperator::CnqsBasicOperator(
    int d, int n, const std::vector<std::tuple<int, int>> &edges, double g,
    double J)
    : CnqsOperator(d, n, edges, g, J), theta_(std::vector<double>(n_, 0.0)) {
    // compute 2 * pi
    const double TWO_PI = 8.0 * std::atan(1.0);

    // calculate theta
    for (int i = 0; i < n_; ++i) {
        theta_[i] = (i * TWO_PI) / n_;
    }
}

void CnqsBasicOperator::ConstructInitialState(CnqsVector &state) const {
    TestCompatibility(state);

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
        int n_lj = num_element_[j];

        // index i_j := i[j]
        //      multiplicative factor in unwrapping: f_j = n[<j]
        //      max value: n_j := n[j]
        int f_j = n_lj;
        int n_j = n_;

        // index i_gj := i[>j]
        //      multiplicative factor in unwrapping: f_gj = f[>j]
        //      max value: n_gj := n[>j]
        int f_gj = f_j * n_j;
        int n_gj = num_element_[d_ - j - 1];

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
}

void CnqsBasicOperator::Apply(const CnqsVector &input_state,
                              CnqsVector &output_state) const {
    TestCompatibility(input_state);
    TestCompatibility(output_state);

    output_state = 0.0;

    // differential operator
    double h = theta_[1] - theta_[0];
    double fact = g_ * J_ / (24.0 * h * h);

    for (int j = 0; j < d_; ++j) {
        int n_lj = num_element_[j];

        int f_j = n_lj;
        int n_j = n_;

        int f_gj = f_j * n_j;
        int n_gj = num_element_[d_ - j - 1];

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

                    output_state(i) -=
                        fact * (-input_state(i_2m) + 16.0 * input_state(i_1m) -
                                30.0 * input_state(i) +
                                16.0 * input_state(i_1p) - input_state(i_2p));
                }
            }
        }
    }

    // multiplication with the cross-terms
    for (const std::tuple<int, int> &edge : edges_) {
        // select the endpoints (j, k) of an edge
        int j = std::get<0>(edge);
        int k = std::get<1>(edge);

        // split indices into five groups: i[<j], i[j], i[j<<k], i[k], i[>k]
        int n_lj = num_element_[j];

        int f_j = n_lj;
        int n_j = n_;

        int f_jk = f_j * n_j;
        int n_jk = num_element_[k - j - 1];

        int f_k = f_jk * n_jk;
        int n_k = n_;

        int f_gk = f_k * n_k;
        int n_gk = num_element_[d_ - k - 1];

        // update new state
        for (int i_gk = 0; i_gk < n_gk; ++i_gk) {
            for (int i_k = 0; i_k < n_k; ++i_k) {
                for (int i_jk = 0; i_jk < n_jk; ++i_jk) {
                    for (int i_j = 0; i_j < n_j; ++i_j) {
                        for (int i_lj = 0; i_lj < n_lj; ++i_lj) {
                            int i = i_lj + f_j * i_j + f_jk * i_jk + f_k * i_k +
                                    f_gk * i_gk;
                            output_state(i) -=
                                J_ * std::cos(theta_[i_j] - theta_[i_k]) *
                                input_state(i);
                        }
                    }
                }
            }
        }
    }
}

std::string CnqsBasicOperator::Describe() const {
    std::string description = "{\n";
    description += "    \"name\": \"CnqsBasicOperator\",\n";
    description += "    \"num_rotor\": " + std::to_string(d_) + ",\n";

    int num_edge = edges_.size();

    description += "    \"edge_list\": [\n";
    for (int edge_id = 0; edge_id < num_edge; ++edge_id) {
        int j = std::get<0>(edges_[edge_id]);
        int k = std::get<1>(edges_[edge_id]);

        if (edge_id != num_edge - 1) {
            description += "        [" + std::to_string(j) + ", " +
                           std::to_string(k) + "],\n";
        } else {
            description += "        [" + std::to_string(j) + ", " +
                           std::to_string(k) + "]\n";
        }
    }
    description += "    ],\n";

    description += "    \"g\": " + std::to_string(g_) + ",\n";
    description += "    \"J\": " + std::to_string(J_) + ",\n";
    description += "    \"num_grid_point\": " + std::to_string(n_) + "\n";
    description += "}";

    return description;
}
