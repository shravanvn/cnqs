#include "CnqsFourierPreconditioner.hpp"

#include <cmath>
#include <stdexcept>

CnqsFourierPreconditioner::CnqsFourierPreconditioner(int num_rotor,
                                                     int max_freq, double g,
                                                     double J, double shift)
    : num_rotor_(num_rotor),
      max_freq_(max_freq),
      g_(g),
      J_(J),
      shift_(shift),
      n_(2 * max_freq + 1),
      num_element_(1) {
    if (num_rotor_ < 2) {
        throw std::domain_error(
            "==CnqsFourierPreconditioner== Need at least two rotors");
    }

    if (max_freq_ < 1) {
        throw std::domain_error(
            "==CnqsFourierPreconditioner== Maximum frequency cut-off must be "
            "at least one");
    }

    if (g_ * J_ <= 0) {
        throw std::domain_error(
            "==CnqsFourierPreconditioner== Values of g and J may not lead to "
            "positive definite preconditioner");
    }

    if (shift_ > -1.0e-15) {
        throw std::domain_error(
            "==CnqsFourierPreconditioner== Value of shift may not lead to "
            "positive definite preconditioner");
    }

    for (int i = 0; i < num_rotor_; ++i) {
        num_element_ *= n_;
    }
}

void CnqsFourierPreconditioner::TestCompatibility(
    const CnqsVector &cnqs_vector) const {
    if (cnqs_vector.Size() != num_element_) {
        throw std::length_error(
            "==CnqsFourierPreconditioner== Vector length is not compatible "
            "with operator");
    }
}

double CnqsFourierPreconditioner::SquaredDistanceFromCenter(int i) const {
    int squared_distance = 0;

    for (int j = 0; j < num_rotor_; ++j) {
        int d_j = i % n_ - max_freq_;
        squared_distance += d_j * d_j;

        i /= n_;
    }

    return squared_distance;
}

void CnqsFourierPreconditioner::Solve(const CnqsVector &input_state,
                                      CnqsVector &output_state) const {
    TestCompatibility(input_state);
    TestCompatibility(output_state);

    double fact = 0.5 * g_ * J_;

    for (int i = 0; i < num_element_; ++i) {
        output_state(i) =
            input_state(i) / (fact * SquaredDistanceFromCenter(i) - shift_);
    }
}

std::string CnqsFourierPreconditioner::Describe() const {
    std::string description = "{\n";
    description += "    \"name\": \"CnqsFourierPreconditioner\",\n";
    description += "    \"num_rotor\": " + std::to_string(num_rotor_) + ",\n";
    description += "    \"g\": " + std::to_string(g_) + ",\n";
    description += "    \"J\": " + std::to_string(J_) + ",\n";
    description += "    \"shift\": " + std::to_string(shift_) + ",\n";
    description += "    \"max_freq\": " + std::to_string(max_freq_) + "\n";
    description += "}";

    return description;
}
