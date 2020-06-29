#include "CnqsOperator.hpp"

void CnqsOperator::ShiftedApply(const CnqsVector &input_state, double shift,
                                CnqsVector &output_state) const {
    Apply(input_state, output_state);
    output_state -= shift * input_state;
}

std::ostream &operator<<(std::ostream &os, const CnqsOperator &cnqs_operator) {
    os << cnqs_operator.Describe() << std::flush;
    return os;
}
