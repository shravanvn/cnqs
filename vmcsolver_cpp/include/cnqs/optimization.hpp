#ifndef CNQS_OPTIMIZATION_HPP
#define CNQS_OPTIMIZATION_HPP

#include "cnqs/config.hpp"
#include "cnqs/nqs.hpp"

namespace cnqs {

void StochasticReconfiguration(const Config &config, Nqs &nqs,
                               const std::vector<double> &gradient_avg,
                               const std::vector<double> &gradient_tensor_avg,
                               const std::vector<double> &energy_gradient_avg,
                               double energy_avg, double &gradient_norm);

}

#endif
