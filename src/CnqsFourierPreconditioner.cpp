#include "CnqsFourierPreconditioner.hpp"

#include <cmath>

CnqsFourierPreconditioner::CnqsFourierPreconditioner(int d, int n, double g,
                                                     double J, double shift)
    : d_(d), n_(n), g_(g), J_(J), shift_(shift), max_freq_((n - 1) / 2) {
    if (d_ < 2) {
        throw std::domain_error(
            "==CnqsFourierPreconditioner== Need at least two rotors");
    }

    if (n_ < 3 || n_ % 2 == 0) {
        throw std::domain_error(
            "==CnqsFourierPreconditioner== Total number of Fourier modes must "
            "be odd and at least 3");
    }

    if (std::abs(shift) < 1.0e-15) {
        throw std::domain_error(
            "==CnqsFourierPreconditioner== Shift is too close to zero "
            "(singular preconditioner)");
    }

    num_element_ = 1;
    for (int i = 0; i < d_; ++i) {
        num_element_ *= n_;
    }
}

double CnqsFourierPreconditioner::SquaredDistanceFromCenter(int i) const {
    int squared_distance = 0;

    for (int j = 0; j < d_; ++j) {
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

void CnqsFourierPreconditioner::Describe(std::string &description) const {
    description += "CnqsPreconditioner {\n";
    description += "     name : cnqs Fourier preconditioner\n";
    description += "        d : " + std::to_string(d_) + "\n";
    description += "        n : " + std::to_string(n_) + "\n";
    description += "        g : " + std::to_string(g_) + "\n";
    description += "        J : " + std::to_string(J_) + "\n";
    description += "    shift : " + std::to_string(shift_) + "\n";
    description += "}";
}
