#include "cnqs/vmcsolver/optimization.hpp"

#include <lapack.hh>

void cnqs::vmcsolver::StochasticReconfiguration(
    const cnqs::vmcsolver::Config &config, cnqs::vmcsolver::Nqs &nqs,
    double local_energy_avg, const std::vector<double> &log_psi_gradient_avg,
    const std::vector<double> &log_psi_gradient_outer_avg,
    const std::vector<double> &local_energy_log_psi_gradient_avg,
    double &gradient_norm) {
    int num_vars = nqs.NumVars();

    // compute gradient and gradient norm
    std::vector<double> gradient(num_vars);
    gradient_norm = 0.0;
    for (int i = 0; i < num_vars; ++i) {
        gradient[i] = local_energy_log_psi_gradient_avg[i] -
                      log_psi_gradient_avg[i] * local_energy_avg;
        gradient_norm = gradient[i] * gradient[i];
    }
    gradient_norm = std::sqrt(gradient_norm);

    // compute Fisher information matrix
    std::vector<double> fisher(num_vars * num_vars);
    for (int i = 0; i < num_vars; ++i) {
        for (int j = 0; j < num_vars; ++j) {
            fisher[i + j * num_vars] =
                log_psi_gradient_outer_avg[i + j * num_vars] -
                log_psi_gradient_avg[i] * log_psi_gradient_avg[i];
        }
    }

    // regularize Fisher information matrix
    for (int i = 0; i < num_vars; ++i) {
        fisher[i + i * num_vars] += config.stochastic_reconfig_regularization;
    }

    // construct update for variational parameters
    std::vector<int64_t> ipiv(num_vars);
    int64_t status = lapack::gesv(num_vars, 1, fisher.data(), num_vars,
                                  ipiv.data(), gradient.data(), num_vars);

    if (status != 0) {
        throw std::logic_error("Linear solve failed in optimization step");
    }

    for (int i = 0; i < num_vars; ++i) {
        gradient[i] *= -config.gradient_descent_learning_rate;
    }

    // update state
    nqs.UpdateVars(gradient);
}
