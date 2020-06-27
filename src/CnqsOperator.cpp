#include "CnqsOperator.hpp"

#include <stdexcept>

CnqsOperator::CnqsOperator(int d, int n,
                           const std::vector<std::tuple<int, int>> &edges,
                           double g, double J, const std::string &name)
    : d_(d),
      n_(n),
      edges_(edges),
      g_(g),
      J_(J),
      name_(name),
      num_element_(std::vector<int>(d + 1)) {
    if (d_ < 2) {
        throw std::domain_error("==CnqsOperator== Need at least two rotors");
    }

    if (n_ < 2) {
        throw std::domain_error(
            "==CnqsOperator== Need at least two grid points");
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

        // check validity
        if (j < 0 || k >= d_) {
            throw std::domain_error(
                "==CnqsOperator== Edge specification is not valid");
        }
    }

    if (g_ * J_ <= 0) {
        throw std::domain_error(
            "==CnqsOperator== Parameter choice may not lead to positive "
            "definite operator");
    }

    num_element_[0] = 1;
    for (int i = 0; i < d_; ++i) {
        num_element_[i + 1] = n_ * num_element_[i];
    }
}

void CnqsOperator::ShiftedApply(const CnqsVector &input_state, double shift,
                                CnqsVector &output_state) const {
    Apply(input_state, output_state);
    output_state -= shift * input_state;
}

std::ostream &operator<<(std::ostream &os, const CnqsOperator &cnqs_operator) {
    os << "==CnqsOperator==" << std::endl
       << "    name         : " << cnqs_operator.name_ << std::endl
       << "    num_rotor    : " << cnqs_operator.d_ << std::endl
       << "    num_grid_pts : " << cnqs_operator.n_ << std::endl
       << "    edge_list    : ";

    for (const auto &edge : cnqs_operator.edges_) {
        os << "(" << std::get<0>(edge) << ", " << std::get<1>(edge) << "), ";
    }

    os << std::endl
       << "    g            : " << cnqs_operator.g_ << std::endl
       << "    J            : " << cnqs_operator.J_ << std::endl
       << "==CnqsOperator==" << std::flush;

    return os;
}
