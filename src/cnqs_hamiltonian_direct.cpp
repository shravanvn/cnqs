#include "cnqs_hamiltonian_direct.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include <hdf5.h>

#include "cg.h"
#include "cnqs_trivial_preconditioner.hpp"

static const double TWO_PI = 8.0 * std::atan(1.0);

static int powint(int n, int d) {
    if (d < 0) {
        throw std::domain_error(
            "powint is only intended for non-negative integer arguments");
    }

    if (d == 0) {
        return 1;
    }

    if (d == 1) {
        return n;
    }

    return powint(n, d / 2) * powint(n, d - d / 2);
}

class CnqsShiftedHamiltonian {
  public:
    CnqsShiftedHamiltonian(const CnqsHamiltonianDirect *hamiltonian, double mu)
        : hamiltonian_(hamiltonian), mu_(mu) {}

    ~CnqsShiftedHamiltonian() = default;

    CnqsState operator*(const CnqsState &state) const {
        return (*hamiltonian_) * state - mu_ * state;
    }

    inline CnqsState trans_mult(const CnqsState &state) const {
        return *this * state;
    }

  private:
    const CnqsHamiltonianDirect *hamiltonian_;
    double mu_;
};

CnqsHamiltonianDirect::CnqsHamiltonianDirect(
    int d, int n, std::vector<std::tuple<int, int>> edges, double g, double J)
    : d_(d), n_(n), edges_(edges), g_(g), J_(J), num_element_(powint(n_, d_)),
      num_edge_(edges_.size()), theta_(std::vector<double>(n)) {
    // assign theta values
    for (int i = 0; i < n_; ++i) {
        theta_[i] = (TWO_PI * i) / n_;
    }
}

CnqsState CnqsHamiltonianDirect::initialize_state() const {
    CnqsState state(num_element_);
    state = 1.0;

    for (int j = 0; j < d_; ++j) {
        // ======================
        // unwrapping the indices
        // ======================
        //
        // split the d indices into three groups:
        //
        //      state[i[0], ..., i[d - 1]] = state[i[<j], i[j], i[>j]]
        //
        // with
        //
        //      i[<j] = (    i[0], ..., i[j - 1])
        //      i[>j] = (i[j + 1], ..., i[d - 1])
        //
        // in column-major format, this (i[<j], i[j], i[>j]) index flattens to
        //
        //       i[<j] + f[j] * i[j] + f[>j] * i[>j]
        //
        // with
        //
        //      0 <= i[<j] < n[<j]
        //      0 <=  i[j] <  n[j],     f[j] = n[<j]
        //      0 <= i[>j] < n[>j],    f[>j] = n[<j] * n[j]
        //
        // and
        //
        //      n[<j] = n[0] * ... * n[j - 1]
        //      n[>j] = n[j + 1] * ... * n[d - 1] = num_elements(state) / f[>j]
        // ======================

        // index i_lj := i[<j]
        //      max value: n_lj := n[<j]
        int n_lj = powint(n_, j);

        // index i_j := i[j]
        //      multiplicative factor in unwrapping: f_j = n[<j]
        //      max value: n_j := n[j]
        int f_j = n_lj;
        int n_j = n_;

        // index i_gj := i[>j]
        //      multiplicative factor in unwrapping: f_gj = f[>j]
        //      max value: n_gj := n[>j]
        int f_gj = f_j * n_j;
        int n_gj = num_element_ / f_gj;

        // update values
        for (int i_gj = 0; i_gj < n_gj; ++i_gj) {
            for (int i_j = 0; i_j < n_j; ++i_j) {
                double sin_factor = std::sin(theta_[i_j]);

                for (int i_lj = 0; i_lj < n_lj; ++i_lj) {
                    state(i_lj + f_j * i_j + f_gj * i_gj) *= sin_factor;
                }
            }
        }
    }

    return state;
}

CnqsState CnqsHamiltonianDirect::operator*(const CnqsState &state) const {
    CnqsState new_state(num_element_);

    // differential operator
    double h = TWO_PI / n_;
    double fact = g_ * J_ / (24.0 * h * h);

    for (int j = 0; j < d_; ++j) {
        int n_lj = powint(n_, j);

        int f_j = n_lj;
        int n_j = n_;

        int f_gj = f_j * n_j;
        int n_gj = num_element_ / f_gj;

        for (int i_gj = 0; i_gj < n_gj; ++i_gj) {
            for (int i_j = 0; i_j < n_j; ++i_j) {
                int i_j_2m = (i_j - 2 >= 0) ? i_j - 2 : n_ + i_j - 2;
                int i_j_1m = (i_j - 1 >= 0) ? i_j - 1 : n_ + i_j - 1;
                int i_j_1p = (i_j + 1 < n_) ? i_j + 1 : i_j + 1 - n_;
                int i_j_2p = (i_j + 2 < n_) ? i_j + 2 : i_j + 2 - n_;

                for (int i_lj = 0; i_lj < n_lj; ++i_lj) {
                    int i_2m = i_lj + f_j * i_j_2m + f_gj * i_gj;
                    int i_1m = i_lj + f_j * i_j_1m + f_gj * i_gj;
                    int i = i_lj + f_j * i_j + f_gj * i_gj;
                    int i_1p = i_lj + f_j * i_j_1p + f_gj * i_gj;
                    int i_2p = i_lj + f_j * i_j_2p + f_gj * i_gj;

                    new_state(i) -= fact * (-state(i_2m) + 16.0 * state(i_1m) -
                                            30.0 * state(i) +
                                            16.0 * state(i_1p) - state(i_2p));
                }
            }
        }
    }

    // multiplication with the cross-terms
    for (int e = 0; e < num_edge_; ++e) {
        // select the endpoints (j, k) of an edge, ensuring j < k
        int j = std::get<0>(edges_[e]);
        int k = std::get<1>(edges_[e]);

        if (j > k) {
            int temp = j;
            j = k;
            k = temp;
        }

        // split indices into five groups: i[<j], i[j], i[j<<k], i[k], i[>k]
        int n_lj = powint(n_, j);

        int f_j = n_lj;
        int n_j = n_;

        int f_jk = f_j * n_j;
        int n_jk = powint(n_, k - j - 1);

        int f_k = f_jk * n_jk;
        int n_k = n_;

        int f_gk = f_k * n_k;
        int n_gk = num_element_ / f_gk;

        // update new state
        for (int i_gk = 0; i_gk < n_gk; ++i_gk) {
            for (int i_k = 0; i_k < n_k; ++i_k) {
                for (int i_jk = 0; i_jk < n_jk; ++i_jk) {
                    for (int i_j = 0; i_j < n_j; ++i_j) {
                        for (int i_lj = 0; i_lj < n_lj; ++i_lj) {
                            int i = i_lj + f_j * i_j + f_jk * i_jk + f_k * i_k +
                                    f_gk * i_gk;
                            new_state(i) -=
                                J_ * std::cos(theta_[i_j] - theta_[i_k]) *
                                state(i);
                        }
                    }
                }
            }
        }
    }

    return new_state;
}

void CnqsHamiltonianDirect::inverse_power_iteration(
    int cg_max_iter, double cg_tol, int power_max_iter, double power_tol,
    const std::string &file_name) const {
    // open HDF5 file
    hid_t file_id =
        H5Fcreate(file_name.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    if (file_id < 0) {
        throw std::runtime_error("--HDF5-- Could not create file " + file_name);
    }

    // output parameters
    std::cout << "============================================================="
                 "========="
              << std::endl
              << "          parameters       conjugate gradient  inverse power "
                 "iteration"
              << std::endl
              << "-------------------- ------------------------ "
                 "------------------------"
              << std::endl
              << "   maximum iteration " << std::setw(24) << cg_max_iter << " "
              << std::setw(24) << power_max_iter << std::endl
              << " specified tolerance " << std::setw(24) << cg_tol << " "
              << std::setw(24) << power_tol << std::endl;

    // set format of floating point outputs in inverse power iteration
    std::cout << std::scientific
              << "============================================================="
                 "========="
              << std::endl
              << " inv_iter                   lambda     d_lambda   cg_iter    "
                 "   cg_tol"
              << std::endl
              << "--------- ------------------------ ------------ --------- "
                 "------------"
              << std::endl;

    // shifted operator
    CnqsShiftedHamiltonian shifted_hamiltonian(this, -num_edge_ * J_);

    // trivial preconditioner
    CnqsTrivialPreconditioner preconditioner;

    // initial estimate for eigenstate and eigenvalue
    CnqsState state = this->initialize_state();
    state = (1.0 / norm(state)) * state;

    double lambda = dot(state, *this * state);

    state.save(file_id, 0);
    std::cout << std::setw(9) << 0 << " " << std::setw(24)
              << std::setprecision(16) << lambda << std::endl;

    // inverse power iteration

    for (int power_iter = 1; power_iter <= power_max_iter; ++power_iter) {
        // initialize CG parameters
        int max_iter = cg_max_iter;
        double tol = cg_tol;

        // solve using CG and normalize
        CnqsState new_state(num_element_);
        int cg_return = CG(shifted_hamiltonian, new_state, state,
                           preconditioner, max_iter, tol);

        if (cg_return != 0) {
            throw std::runtime_error("CG iteration did not converge");
        }

        new_state = (1.0 / norm(new_state)) * new_state;

        // compute new eigenvalue
        double new_lambda = dot(new_state, *this * new_state);
        double d_lambda = std::abs(lambda - new_lambda);

        // save new eigenvector estimate to file
        new_state.save(file_id, power_iter);

        // report diagnostics
        std::cout << std::setw(9) << power_iter << " " << std::setw(24)
                  << std::setprecision(16) << new_lambda << " " << std::setw(12)
                  << std::setprecision(6) << d_lambda << " " << std::setw(9)
                  << max_iter << " " << std::setw(12) << tol << std::endl;

        if (d_lambda < power_tol) {
            break;
        }

        // update eigenvalue and eigenstate estiamtes
        state = new_state;
        lambda = new_lambda;
    }

    std::cout << "============================================================="
                 "========="
              << std::endl;

    // close HDF5 file
    herr_t status = H5Fclose(file_id);

    if (status < 0) {
        throw std::runtime_error("--HDF5-- Could not close file " + file_name);
    }
}
