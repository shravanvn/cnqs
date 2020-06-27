#ifndef CNQS_TRIVIAL_PRECONDITIONER_HPP
#define CNQS_TRIVIAL_PRECONDITIONER_HPP

#include <string>

#include "CnqsPreconditioner.hpp"
#include "CnqsVector.hpp"

class CnqsTrivialPrecondtioner : public CnqsPreconditioner {
public:
    CnqsTrivialPrecondtioner()
        : CnqsPreconditioner("cnqs trivial preconditioner") {}

    ~CnqsTrivialPrecondtioner() = default;

    void TestCompatibility(const CnqsVector &state) const {}

    void Solve(const CnqsVector &input_state, CnqsVector &output_state) const {
        output_state = input_state;
    }
};

#endif
