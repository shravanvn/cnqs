#ifndef CNQS_OPERATOR_HPP
#define CNQS_OPERATOR_HPP

#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "CnqsVector.hpp"

class CnqsOperator {
public:
    CnqsOperator(int d, int n, const std::vector<std::tuple<int, int>> &edges,
                 double g, double J, const std::string &name);

    virtual ~CnqsOperator() = default;

    void TestCompatibility(const CnqsVector &state) const {
        if (state.Size() != num_element_[d_]) {
            throw std::length_error(
                "==CnqsOperator== Vector length is not compatible with "
                "operator");
        }
    }

    virtual void ConstructInitialState(CnqsVector &state) const = 0;

    virtual void Apply(const CnqsVector &input_state,
                       CnqsVector &output_state) const = 0;

    double EigValLowerBound() const { return -edges_.size() * J_; }

    void ShiftedApply(const CnqsVector &input_state, double shift,
                      CnqsVector &output_state) const;

    friend std::ostream &operator<<(std::ostream &os,
                                    const CnqsOperator &cnqs_operator);

protected:
    int d_;
    int n_;
    std::vector<std::tuple<int, int>> edges_;
    double g_;
    double J_;
    std::string name_;
    std::vector<int> num_element_;
};

#endif
