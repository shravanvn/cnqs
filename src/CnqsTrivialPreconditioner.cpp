#include "CnqsTrivialPreconditioner.hpp"

void CnqsTrivialPreconditioner::TestCompatibility(
    const CnqsVector &vector) const {}

void CnqsTrivialPreconditioner::Solve(const CnqsVector &input_vector,
                                      CnqsVector &output_vector) const {
    output_vector = input_vector;
}

void CnqsTrivialPreconditioner::Describe(std::string &description) const {
    description += "CnqsPreconditioner {\n";
    description += "    name : cnqs trivial preconditioner\n";
    description += "}";
}
