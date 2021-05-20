#ifndef CNQS_VMCSOLVER_SAMPLER_HPP
#define CNQS_VMCSOVLER_SAMPLER_HPP

#include <random>
#include <vector>

#include "cnqs/vmcsolver/config.hpp"
#include "cnqs/vmcsolver/nqs.hpp"

namespace cnqs {

namespace vmcsolver {

/// Metropolis sampler
void MetropolisSampler(int step, const Config &config, Nqs &nqs,
                       double &local_energy_avg, double &local_energy_std,
                       std::vector<double> &log_psi_gradient_avg,
                       std::vector<double> &log_psi_gradient_outer_avg,
                       std::vector<double> &local_energy_log_psi_gradient_avg,
                       double &acceptance_rate, std::mt19937 &rng);

}  // namespace vmcsolver

}  // namespace cnqs

#endif
