#ifndef CNQS_SAMPLER_HPP
#define CNQS_SAMPLER_HPP

#include <random>
#include <vector>

#include "cnqs/config.hpp"
#include "cnqs/nqs.hpp"

namespace cnqs {

void MetropolisSampler(const Config &config, Nqs &nqs,
                       std::vector<double> &gradient_avg,
                       std::vector<double> &gradient_tensor_avg,
                       std::vector<double> &energy_gradient_avg,
                       double &energy_avg, double &energy_std,
                       double &acceptance_rate, std::mt19937 &rng);

}

#endif
