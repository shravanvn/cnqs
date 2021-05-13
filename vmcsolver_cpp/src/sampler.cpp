#include "cnqs/sampler.hpp"

#include <cmath>

void cnqs::MetropolisSampler(const cnqs::Config &config, cnqs::Nqs &nqs,
                             std::vector<double> &gradient_avg,
                             std::vector<double> &gradient_tensor_avg,
                             std::vector<double> &energy_gradient_avg,
                             double &energy_avg, double &energy_std,
                             double &acceptance_rate, std::mt19937 &rng) {
    int num_vars = nqs.NumVars();

    for (int i = 0; i < num_vars; ++i) {
        gradient_avg[i] = 0.0;
    }
    for (int i = 0; i < num_vars * num_vars; ++i) {
        gradient_tensor_avg[i] = 0.0;
    }
    for (int i = 0; i < num_vars; ++i) {
        energy_gradient_avg[i] = 0.0;
    }
    energy_avg = 0.0;
    energy_std = 0.0;

    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    int num_acceptance = 0;
    int count = 0;
    for (int t = 0; t < config.metropolis_num_steps; ++t) {
        cnqs::Nqs nqs_new = nqs.ProposeUpdate(config, rng);

        double log_psi_old = nqs.LogPsi();
        double log_psi_new = nqs_new.LogPsi();

        double log_p = std::log(uniform(rng));

        if (log_psi_new > log_psi_old + 0.5 * log_p) {
            nqs = nqs_new;
            ++num_acceptance;
        }

        if ((t >= config.metropolis_warm_steps) &&
            ((t - config.metropolis_warm_steps) %
                 config.metropolis_cherry_pick ==
             0)) {
            double local_energy;
            std::vector<double> gradient(num_vars);
            nqs.LocalEnergyAndLogPsiGradient(config, local_energy, gradient);

            for (int i = 0; i < num_vars; ++i) {
                gradient_avg[i] += gradient[i];

                for (int j = 0; j < num_vars; ++j) {
                    gradient_tensor_avg[i + j * num_vars] +=
                        gradient[i] * gradient[j];
                }

                energy_gradient_avg[i] += local_energy * gradient[i];
            }

            energy_avg += local_energy;
            energy_std += local_energy * local_energy;

            ++count;
        }
    }

    for (int i = 0; i < num_vars; ++i) {
        gradient_avg[i] /= count;

        for (int j = 0; j < num_vars; ++j) {
            gradient_tensor_avg[i + j * num_vars] /= count;
        }

        energy_gradient_avg[i] /= count;
    }

    energy_avg /= count;
    energy_std /= count;

    energy_std -= energy_avg * energy_avg;
    energy_std *= count / (count - 1.0);
    energy_std = std::sqrt(energy_std);

    acceptance_rate =
        static_cast<double>(num_acceptance) / config.metropolis_num_steps;
}
