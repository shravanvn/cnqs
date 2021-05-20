#ifndef CNQS_VMCSOLVER_NQS_HPP
#define CNQS_VMCSOLVER_NQS_HPP

#include <random>
#include <string>
#include <vector>

#include "cnqs/vmcsolver/config.hpp"

namespace cnqs {

namespace vmcsolver {

/// Continuous neural quantum state implemented with RBM
class Nqs {
public:
    Nqs(const Config &config);

    int NumVars() const { return vars_.size(); }

    const std::vector<double> &State() const { return theta_; }

    double VisibleBiasNorm() const;

    double HiddenBiasNorm() const;

    double LogPsi() const;

    void LocalEnergyAndLogPsiGradient(
        const Config &config, double &local_energy,
        std::vector<double> &log_psi_gradient) const;

    Nqs ProposeUpdate(const Config &config, std::mt19937 &rng) const;

    void Output(const std::string &file_name) const;

    void RandInit(std::mt19937 &rng);

    void UpdateVars(const std::vector<double> &vars_diff);

private:
    void Recompute();

    const double *WeightData() const;

    int n_;
    int h_;
    std::vector<double> vars_;
    std::vector<double> theta_;
    std::vector<double> x_;
    std::vector<double> x_act_;
    std::vector<double> r_;
    std::vector<double> g_r_;
    std::vector<double> g_r_over_r_;
};

}  // namespace vmcsolver

}  // namespace cnqs

#endif
