#ifndef CNQS_OPERATOR_HPP
#define CNQS_OPERATOR_HPP

#include <iostream>
#include <tuple>
#include <vector>

#include "CnqsVector.hpp"

class CnqsOperator {
  public:
    CnqsOperator(int d, int n, const std::vector<std::tuple<int, int>> &edges,
                 double g, double J, const std::string &name);

    ~CnqsOperator() = default;

    virtual void ConstructInitialState(CnqsVector &state) const = 0;

    virtual void Apply(const CnqsVector &input_state,
                       CnqsVector &output_state) const = 0;

    double EigValLowerBound() const { return mu_; }

    friend std::ostream &operator<<(std::ostream &os,
                                    const CnqsOperator &cnqs_operator);

  protected:
    int d_;
    int n_;
    std::vector<std::tuple<int, int>> edges_;
    double g_;
    double J_;
    std::string name_;
    double mu_;
};

#endif
