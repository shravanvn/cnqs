#ifndef CNQS_PROBLEM_HPP
#define CNQS_PROBLEM_HPP

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "Cnqs_Network.hpp"

namespace Cnqs {

class Problem {
public:
    virtual ~Problem() = default;

    virtual double
    runInversePowerIteration(int numPowerIter, double tolPowerIter,
                             int numCgIter, double tolCgIter,
                             const std::string &fileName) const = 0;

    virtual std::string description() const = 0;
};

} // namespace Cnqs

std::ostream &operator<<(std::ostream &os, const Cnqs::Problem &problem);

#endif
