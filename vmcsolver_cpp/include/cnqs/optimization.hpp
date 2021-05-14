#ifndef CNQS_OPTIMIZATION_HPP
#define CNQS_OPTIMIZATION_HPP

#include "cnqs/config.hpp"
#include "cnqs/nqs.hpp"

namespace cnqs {

void StochasticReconfiguration(
    const Config &config, Nqs &nqs, double local_energy_avg,
    const std::vector<double> &log_psi_gradient_avg,
    const std::vector<double> &log_psi_gradient_outer_avg,
    const std::vector<double> &local_energy_log_psi_gradient_avg,
    double &gradient_norm);

}

#endif
