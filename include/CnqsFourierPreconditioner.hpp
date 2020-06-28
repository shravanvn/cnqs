#ifndef CNQS_FOURIER_PRECONDITIONER_HPP
#define CNQS_FOURIER_PRECONDITIONER_HPP

#include <stdexcept>
#include <string>

#include "CnqsPreconditioner.hpp"
#include "CnqsVector.hpp"

class CnqsFourierPreconditioner : public CnqsPreconditioner {
public:
    CnqsFourierPreconditioner(int d, int n, double g, double J, double shift);

    ~CnqsFourierPreconditioner() = default;

    void TestCompatibility(const CnqsVector &state) const {
        if (state.Size() != num_element_) {
            throw std::length_error(
                "==CnqsFourierPreconditioner== Vector length is not compatible "
                "with operator");
        }
    }

    void Solve(const CnqsVector &input_state, CnqsVector &output_state) const;

private:
    double SquaredDistanceFromCenter(int i) const;

    int d_;
    int n_;
    double g_;
    double J_;
    double shift_;
    int max_freq_;
    int num_element_;
};

#endif
