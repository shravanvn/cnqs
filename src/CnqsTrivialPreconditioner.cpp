#include "CnqsTrivialPreconditioner.hpp"

void CnqsTrivialPreconditioner::TestCompatibility(
    const CnqsVector &vector) const {}

void CnqsTrivialPreconditioner::Solve(const CnqsVector &input_vector,
                                      CnqsVector &output_vector) const {
    output_vector = input_vector;
}

std::string CnqsTrivialPreconditioner::Describe() const {
    std::string description = "{\n";
    description += "    \"name\": \"CnqsTrivialPreconditioner\"\n";
    description += "}";

    return description;
}
