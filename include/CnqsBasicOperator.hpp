#ifndef CNQS_BASIC_OPERATOR_HPP
#define CNQS_BASIC_OPERATOR_HPP

#include <tuple>
#include <vector>

#include "CnqsOperator.hpp"

class CnqsBasicOperator : public CnqsOperator {
public:
    CnqsBasicOperator(int d, int n,
                      const std::vector<std::tuple<int, int>> &edges, double g,
                      double J);

    ~CnqsBasicOperator() = default;

    void ConstructInitialState(CnqsVector &state) const;

    void Apply(const CnqsVector &input_state, CnqsVector &output_state) const;

private:
    std::vector<double> theta_;
};

#endif
