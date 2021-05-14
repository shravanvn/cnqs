#ifndef CNQS_NQS_HPP
#define CNQS_NQS_HPP

#include <random>
#include <string>
#include <vector>

#include "cnqs/config.hpp"

namespace cnqs {

class Nqs {
public:
    Nqs(const Config &config);

    void RandInit(std::mt19937 &rng);

    void UpdateVars(const std::vector<double> &vars_diff);

    int NumVars() const { return vars_.size(); }

    double VisibleBiasNorm() const;

    double HiddenBiasNorm() const;

    double LogPsi() const;

    void LocalEnergyAndLogPsiGradient(
        const Config &config, double &local_energy,
        std::vector<double> &log_psi_gradient) const;

    void Output(const std::string &file_name) const;

    Nqs ProposeUpdate(const Config &config, std::mt19937 &rng) const;

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

}  // namespace cnqs

#endif
