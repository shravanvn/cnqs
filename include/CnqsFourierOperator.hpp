#ifndef CNQS_FOURIER_OPERATOR_HPP
#define CNQS_FOURIER_OPERATOR_HPP

#include <tuple>
#include <vector>

#include "CnqsOperator.hpp"

class CnqsFourierOperator : public CnqsOperator {
public:
    CnqsFourierOperator(int d, int n,
                        const std::vector<std::tuple<int, int>> &edges,
                        double g, double J);

    ~CnqsFourierOperator() = default;

    void ConstructInitialState(CnqsVector &state) const;

    void Apply(const CnqsVector &input_state, CnqsVector &output_state) const;

private:
    bool IndexQualifiesForInitialState(int i) const;

    double SquaredDistanceFromCenter(int i) const;

    int max_freq_;
};

#endif
