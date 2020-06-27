#ifndef IML_CNQS_INTERFACE_HPP
#define IML_CNQS_INTERFACE_HPP

#include <memory>

#include "CnqsOperator.hpp"
#include "CnqsPreconditioner.hpp"
#include "CnqsVector.hpp"

inline double dot(const CnqsVector &v1, const CnqsVector &v2) {
    return v1.Dot(v2);
}

inline double norm(const CnqsVector &v) { return v.Norm(); }

class ImlCnqsOperator {
public:
    ImlCnqsOperator(const std::shared_ptr<const CnqsOperator> &cnqs_operator,
                    double shift)
        : operator_(cnqs_operator), shift_(shift) {}

    ~ImlCnqsOperator() = default;

    CnqsVector operator*(const CnqsVector &input_vector) const {
        CnqsVector output_vector(input_vector.Size());
        operator_->ShiftedApply(input_vector, shift_, output_vector);
        return output_vector;
    }

    CnqsVector trans_mult(const CnqsVector &input_vector) const {
        return *this * input_vector;
    }

private:
    std::shared_ptr<const CnqsOperator> operator_;
    double shift_;
};

class ImlCnqsPreconditioner {
public:
    ImlCnqsPreconditioner(
        const std::shared_ptr<const CnqsPreconditioner> &preconditioner)
        : preconditioner_(preconditioner) {}

    ~ImlCnqsPreconditioner() = default;

    CnqsVector solve(const CnqsVector &input_vector) const {
        CnqsVector output_vector(input_vector.Size());
        preconditioner_->Solve(input_vector, output_vector);
        return output_vector;
    }

    CnqsVector trans_solve(const CnqsVector &input_vector) const {
        return this->solve(input_vector);
    }

private:
    std::shared_ptr<const CnqsPreconditioner> preconditioner_;
};

#endif
