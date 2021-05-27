#include "cnqs/vmcsolver/nqs.hpp"

#include <blas.hh>
#include <boost/math/special_functions/bessel.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

static const double PI = 3.14159265358979323846;

static double Center(double theta) {
    double n = std::floor((theta + PI) / (2.0 * PI));
    return theta - 2.0 * n * PI;
}

cnqs::vmcsolver::Nqs::Nqs(const Config &config) {
    n_ = config.hamiltonian_num_rotor;
    h_ = config.rbm_num_hidden;
    vars_.resize(2 * n_ + 2 * h_ + h_ * n_);
    theta_.resize(n_);
    x_.resize(2 * n_);
    x_act_.resize(2 * h_);
    r_.resize(h_);
    g_r_.resize(h_);
    g_r_over_r_.resize(h_);
}

void cnqs::vmcsolver::Nqs::RandInit(std::mt19937 &rng) {
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);

    int offset = 0;
    for (int i = 0; i < 2 * n_; ++i) {
        vars_[offset] = uniform(rng);
        ++offset;
    }
    for (int i = 0; i < 2 * h_; ++i) {
        vars_[offset] = uniform(rng);
        ++offset;
    }
    for (int i = 0; i < h_ * n_; ++i) {
        vars_[offset] = normal(rng);
        ++offset;
    }

    for (int i = 0; i < n_; ++i) {
        theta_[i] = PI * uniform(rng);
    }

    for (int i = 0; i < n_; ++i) {
        x_[i] = std::cos(theta_[i]);
        x_[i + n_] = std::sin(theta_[i]);
    }

    Recompute();
}

void cnqs::vmcsolver::Nqs::UpdateVars(const std::vector<double> &vars_diff) {
    if (vars_.size() != vars_diff.size()) {
        throw std::invalid_argument(
            "Length of new vars is incompatible with Nqs object");
    }

    const int num_var = 2 * n_ + 2 * h_ + h_ * n_;

    for (int i = 0; i < num_var; ++i) {
        vars_[i] += vars_diff[i];
    }

    Recompute();
}

double cnqs::vmcsolver::Nqs::VisibleBiasNorm() const {
    double norm_squared = 0.0;
    for (int i = 0; i < 2 * n_; ++i) {
        norm_squared += vars_[i] * vars_[i];
    }

    return std::sqrt(norm_squared);
}

double cnqs::vmcsolver::Nqs::HiddenBiasNorm() const {
    double norm_squared = 0.0;
    for (int i = 0; i < 2 * h_; ++i) {
        norm_squared += vars_[2 * n_ + i] * vars_[2 * n_ + i];
    }

    return std::sqrt(norm_squared);
}

double cnqs::vmcsolver::Nqs::LogPsi() const {
    double log_psi = 0.0;

    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < 2; ++j) {
            log_psi += vars_[i + j * n_] * x_[i + j * n_];
        }
    }

    for (int i = 0; i < h_; ++i) {
        log_psi += std::log(boost::math::cyl_bessel_i(0.0, r_[i]));
    }

    return log_psi;
}

void cnqs::vmcsolver::Nqs::LocalEnergyAndLogPsiGradient(
    const cnqs::vmcsolver::Config &config, double &local_energy,
    std::vector<double> &log_psi_gradient) const {
    // compute gradient of log amplitude
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < 2; ++j) {
            log_psi_gradient[i + j * n_] = x_[i + j * n_];
        }
    }

    for (int i = 0; i < h_; ++i) {
        for (int j = 0; j < 2; ++j) {
            log_psi_gradient[2 * n_ + i + j * h_] =
                g_r_over_r_[i] * x_act_[i + j * h_];
        }
    }

    {
        std::vector<double> temp(h_ * n_);
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
                   h_, n_, 2, 1.0, x_act_.data(), h_, x_.data(), n_, 0.0,
                   temp.data(), h_);

        for (int i = 0; i < h_; ++i) {
            for (int j = 0; j < n_; ++j) {
                log_psi_gradient[2 * n_ + 2 * h_ + i + j * h_] =
                    g_r_over_r_[i] * temp[i + j * h_];
            }
        }
    }

    // compute local kinetic energy
    local_energy = 0.0;
    {
        // z_bar = log_psi_hidden(nqs)
        std::vector<double> z_bar(2 * h_);
        for (int i = 0; i < 2 * h_; ++i) {
            z_bar[i] = log_psi_gradient[2 * n_ + i];
        }

        // z_act_bar = visible_bias + weights.T * z_bar
        std::vector<double> z_act_bar(2 * n_);
        for (int i = 0; i < 2 * n_; ++i) {
            z_act_bar[i] = vars_[i];
        }

        blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                   n_, 2, h_, 1.0, WeightData(), h_, z_bar.data(), h_, 1.0,
                   z_act_bar.data(), n_);

        // z_bar_2 = z_bar[:, new, :] * z_bar[:, :, new]
        std::vector<double> z_bar_2(4 * h_);
        for (int i = 0; i < h_; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    z_bar_2[i + j * h_ + k * 2 * h_] =
                        z_bar[i + k * h_] * z_bar[i + j * h_];
                }
            }
        }

        // z_act_bar_2 = z_act_bar[:, new, :] * z_act_bar[:, :, new]
        std::vector<double> z_act_bar_2(4 * n_);
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    z_act_bar_2[i + j * n_ + k * 2 * n_] =
                        z_act_bar[i + k * n_] * z_act_bar[i + j * n_];
                }
            }
        }

        // u = x_act / r[:, new]
        std::vector<double> u(2 * h_);
        for (int i = 0; i < h_; ++i) {
            for (int j = 0; j < 2; ++j) {
                u[i + j * h_] = x_act_[i + j * h_] / r_[i];
            }
        }

        // u_2 = u[:, new, :] * u[:, :, new]
        std::vector<double> u_2(4 * h_);
        for (int i = 0; i < h_; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    u_2[i + j * h_ + k * 2 * h_] =
                        u[i + k * h_] * u[i + j * h_];
                }
            }
        }

        // scale = 1 - 2 * g_r_over_r
        std::vector<double> scale(h_);
        for (int i = 0; i < h_; ++i) {
            scale[i] = 1.0 - 2.0 * g_r_over_r_[i];
        }

        // shift = g_r_over_r (np.tensordot, axes=0) identity(2)
        std::vector<double> shift(4 * h_);
        for (int i = 0; i < h_; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    shift[i + j * h_ + k * 2 * h_] =
                        (j == k) ? g_r_over_r_[i] : 0.0;
                }
            }
        }

        // z_z_bar = shift + us_2 * scale[:, new, new]
        std::vector<double> z_z_bar(4 * h_);
        for (int i = 0; i < h_; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    z_z_bar[i + j * h_ + k * 2 * h_] =
                        shift[i + j * h_ + k * 2 * h_] +
                        u_2[i + j * h_ + k * 2 * h_] * scale[i];
                }
            }
        }

        // z_cov = z_z_bar - z_bar_2
        std::vector<double> z_cov(4 * h_);
        for (int i = 0; i < 4 * h_; ++i) {
            z_cov[i] = z_z_bar[i] - z_bar_2[i];
        }

        // metric = z_act_bar_2 + weights**2 (np.tensordot, axes=(0, 0)) z_cov
        std::vector<double> metric(4 * n_);
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    metric[i + j * n_ + k * 2 * n_] =
                        z_act_bar_2[i + j * n_ + k * 2 * n_];

                    for (int l = 0; l < h_; ++l) {
                        metric[i + j * n_ + k * 2 * n_] +=
                            vars_[2 * n_ + 2 * h_ + l + i * h_] *
                            vars_[2 * n_ + 2 * h_ + l + i * h_] *
                            z_cov[l + j * h_ + k * 2 * h_];
                    }
                }
            }
        }

        // x_perp = [-x[:, 1], x[:, 0]]
        std::vector<double> x_perp(2 * n_);
        for (int i = 0; i < n_; ++i) {
            x_perp[i] = -x_[i + n_];
            x_perp[i + n_] = x_[i];
        }

        // x_perp_2 = x_perp[:, new, :] * x_perp[:, :, new]
        std::vector<double> x_perp_2(4 * n_);
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    x_perp_2[i + j * n_ + k * 2 * n_] =
                        x_perp[i + k * n_] * x_perp[i + j * n_];
                }
            }
        }

        for (int i = 0; i < 4 * n_; ++i) {
            local_energy += x_perp_2[i] * metric[i];
        }

        for (int i = 0; i < 2 * n_; ++i) {
            local_energy -= x_[i] * z_act_bar[i];
        }
    }

    // compute local energy
    local_energy *= -0.5 * config.hamiltonian_vertex_weight;

    for (int i = 0; i < config.hamiltonian_num_edges; ++i) {
        int j = config.hamiltonian_edge_vertex_0[i];
        int k = config.hamiltonian_edge_vertex_1[i];
        double beta = config.hamiltonian_edge_weights[i];

        local_energy +=
            2.0 * beta * (1.0 - x_[j] * x_[k] - x_[j + n_] * x_[k + n_]);
    }
}

void cnqs::vmcsolver::Nqs::Output(const std::string &file_name) const {
    std::ofstream output_file(file_name);

    output_file << "NQS" << std::endl;
    output_file << std::endl;

    output_file << "num_visible" << std::endl;
    output_file << n_ << std::endl;
    output_file << std::endl;

    output_file << "num_hidden" << std::endl;
    output_file << h_ << std::endl;
    output_file << std::endl;

    output_file << std::scientific;

    output_file << "variational_parameters" << std::endl;
    for (const auto &v : vars_) {
        output_file << std::setw(24) << std::setprecision(17) << v << std::endl;
    }
    output_file << std::endl;

    output_file << "state" << std::endl;
    for (const auto &t : theta_) {
        output_file << std::setw(24) << std::setprecision(17) << t << std::endl;
    }
    output_file << std::endl;

    output_file << std::defaultfloat;
}

cnqs::vmcsolver::Nqs cnqs::vmcsolver::Nqs::ProposeUpdate(
    const Config &config, std::mt19937 &rng) const {
    // sample site
    std::uniform_int_distribution<int> uniform_int(0, n_ - 1);
    int site = uniform_int(rng);

    // sample bump
    std::uniform_real_distribution<double> uniform_real(
        -config.metropolis_bump_size, config.metropolis_bump_size);
    double bump = uniform_real(rng);

    // new theta
    double theta = Center(theta_[site] + bump);

    // new NQS
    cnqs::vmcsolver::Nqs nqs_new(*this);
    nqs_new.theta_[site] = theta;
    nqs_new.x_[site] = std::cos(theta);
    nqs_new.x_[site + n_] = std::sin(theta);
    nqs_new.Recompute();

    return nqs_new;
}

void cnqs::vmcsolver::Nqs::Recompute() {
    // x_act = weights * x + hidden_bias
    for (int i = 0; i < 2 * h_; ++i) {
        x_act_[i] = vars_[2 * n_ + i];
    }

    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, h_,
               2, n_, 1.0, WeightData(), h_, x_.data(), n_, 1.0, x_act_.data(),
               h_);

    // r = hypot(x_act[:, 0], x_act[:, 1])
    // g_r = Bessel_I_1(r) / Bessel_I_0(r)
    // g_r_over_r = g_r / r
    for (int i = 0; i < h_; ++i) {
        r_[i] = hypot(x_act_[i], x_act_[i + h_]);
        g_r_[i] = boost::math::cyl_bessel_i(1.0, r_[i]) /
                  boost::math::cyl_bessel_i(0.0, r_[i]);
        g_r_over_r_[i] = g_r_[i] / r_[i];
    }
}

const double *cnqs::vmcsolver::Nqs::WeightData() const {
    return vars_.data() + 2 * n_ + 2 * h_;
}
