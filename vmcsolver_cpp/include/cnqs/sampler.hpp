#ifndef CNQS_SAMPLER_HPP
#define CNQS_SAMPLER_HPP

#include <random>
#include <vector>

#include "cnqs/config.hpp"
#include "cnqs/nqs.hpp"

namespace cnqs {

void MetropolisSampler(const Config &config, Nqs &nqs, double &local_energy_avg,
                       double &local_energy_std,
                       std::vector<double> &log_psi_gradient_avg,
                       std::vector<double> &log_psi_gradient_outer_avg,
                       std::vector<double> &local_energy_log_psi_gradient_avg,
                       double &acceptance_rate, std::mt19937 &rng);

}

#endif
