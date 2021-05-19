#ifndef CNQS_VMCSOLVER_OPTIMIZATION_HPP
#define CNQS_VMCSOLVER_OPTIMIZATION_HPP

#include "cnqs/vmcsolver/config.hpp"
#include "cnqs/vmcsolver/nqs.hpp"

namespace cnqs {

namespace vmcsolver {

/// Stochastic reconfiguration
void StochasticReconfiguration(
    const Config &config, Nqs &nqs, double local_energy_avg,
    const std::vector<double> &log_psi_gradient_avg,
    const std::vector<double> &log_psi_gradient_outer_avg,
    const std::vector<double> &local_energy_log_psi_gradient_avg,
    double &gradient_norm);

}  // namespace vmcsolver

}  // namespace cnqs

#endif
