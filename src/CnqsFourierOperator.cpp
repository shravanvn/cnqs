#include "CnqsFourierOperator.hpp"

#include <cmath>
#include <stdexcept>

CnqsFourierOperator::CnqsFourierOperator(
    int num_rotor, int max_freq, const std::vector<std::tuple<int, int>> &edges,
    double g, double J)
    : num_rotor_(num_rotor),
      max_freq_(max_freq),
      edges_(edges),
      g_(g),
      J_(J),
      num_tot_freq_(2 * max_freq + 1),
      unfolding_factor_(std::vector<int>(num_rotor + 1)) {
    // validate inputs
    if (num_rotor_ < 2) {
        throw std::domain_error(
            "==CnqsFourierOperator== Need at least two quantum rotors");
    }

    if (max_freq_ < 1) {
        throw std::domain_error(
            "==CnqsFourierOperator== Maximum frequency cut-off must be at "
            "least one");
    }

    for (std::tuple<int, int> &edge : edges_) {
        int j = std::get<0>(edge);
        int k = std::get<1>(edge);

        // switch order to ensure j < k
        if (j > k) {
            int temp = j;
            j = k;
            k = temp;

            edge = std::make_tuple(j, k);
        }

        // check validity of boundary node identifiers
        if (j < 0 || k >= num_rotor_) {
            throw std::domain_error(
                "==CnqsFourierOperator== Edge specification is not valid");
        }
    }

    if (g_ * J_ <= 0) {
        throw std::domain_error(
            "==CnqsFourierOperator== Values of g and J may not lead to "
            "positive definite operator");
    }

    // compute the unfolding factors
    unfolding_factor_[0] = 1;
    for (int i = 0; i < num_rotor_; ++i) {
        unfolding_factor_[i + 1] = num_tot_freq_ * unfolding_factor_[i];
    }
}

void CnqsFourierOperator::TestCompatibility(
    const CnqsVector &cnqs_vector) const {
    if (cnqs_vector.Size() != unfolding_factor_[num_rotor_]) {
        throw std::length_error(
            "==CnqsFourierOperator== Vector length is not compatible with "
            "operator");
    }
}

bool CnqsFourierOperator::IndexQualifiesForInitialState(int i) const {
    for (int j = 0; j < num_rotor_; ++j) {
        if (i == 0) {
            return false;
        }

        int i_j = i % num_tot_freq_;

        if ((i_j != max_freq_ - 1) && (i_j != max_freq_ + 1)) {
            return false;
        }

        i /= num_tot_freq_;
    }

    return true;
}

void CnqsFourierOperator::ConstructInitialState(CnqsVector &state) const {
    TestCompatibility(state);

    double value = std::pow(4.0 * std::atan(1.0), num_rotor_);

    for (int i = 0; i < unfolding_factor_[num_rotor_]; ++i) {
        if (IndexQualifiesForInitialState(i)) {
            state(i) = value;
        }
    }
}

double CnqsFourierOperator::SquaredDistanceFromCenter(int i) const {
    int squared_distance = 0;

    for (int j = 0; j < num_rotor_; ++j) {
        int d_j = i % num_tot_freq_ - max_freq_;
        squared_distance += d_j * d_j;

        i /= num_tot_freq_;
    }

    return squared_distance;
}

void CnqsFourierOperator::Apply(const CnqsVector &input_state,
                                CnqsVector &output_state) const {
    TestCompatibility(input_state);
    TestCompatibility(output_state);

    output_state = 0.0;

    // differential part
    double fact = 0.5 * g_ * J_;

    for (int i = 0; i < unfolding_factor_[num_rotor_]; ++i) {
        output_state(i) += fact * SquaredDistanceFromCenter(i) * input_state(i);
    }

    // mutliplication with cross terms
    fact = -0.5 * J_;
    for (const auto &edge : edges_) {
        // select the endpoints (j, k) of an edge
        int j = std::get<0>(edge);
        int k = std::get<1>(edge);

        // split indices into five groups: i[<j], i[j], i[j<<k], i[k], i[>k]
        int n_lj = unfolding_factor_[j];

        int f_j = n_lj;
        int n_j = num_tot_freq_;

        int f_jk = f_j * n_j;
        int n_jk = unfolding_factor_[k - j - 1];

        int f_k = f_jk * n_jk;
        int n_k = num_tot_freq_;

        int f_gk = f_k * n_k;
        int n_gk = unfolding_factor_[num_rotor_ - k - 1];

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
                                s_jpkm = input_state(i_lj + f_j * i_jp +
                                                     f_jk * i_jk + f_k * i_km +
                                                     f_gk * i_gk);
                            }

                            int i_jm = i_j - 1;
                            int i_kp = i_k + 1;
                            double s_jmkp = 0.0;
                            if (i_jm >= 0 && i_kp < n_k) {
                                s_jmkp = input_state(i_lj + f_j * i_jm +
                                                     f_jk * i_jk + f_k * i_kp +
                                                     f_gk * i_gk);
                            }

                            output_state(i_lj + f_j * i_j + f_jk * i_jk +
                                         f_k * i_k + f_gk * i_gk) +=
                                fact * (s_jpkm + s_jmkp);
                        }
                    }
                }
            }
        }
    }
}

std::string CnqsFourierOperator::Describe() const {
    std::string description = "{\n";
    description += "    \"name\": \"CnqsFourierOperator\",\n";
    description += "    \"num_rotor\": " + std::to_string(num_rotor_) + ",\n";

    int num_edge = edges_.size();

    description += "    \"edges\": [\n";
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
    description += "    \"max_freq\": " + std::to_string(max_freq_) + "\n";
    description += "}";

    return description;
}
