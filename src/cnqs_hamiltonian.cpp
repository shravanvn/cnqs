#include "cnqs_hamiltonian.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "cg.h"

#include "utils.hpp"

static const double TWO_PI = 8.0 * std::atan(1.0);

class CnqsShiftedHamiltonian {
  public:
    CnqsShiftedHamiltonian(const CnqsHamiltonian *hamiltonian, double mu)
        : hamiltonian_(hamiltonian), mu_(mu) {}

    ~CnqsShiftedHamiltonian() = default;

    CnqsState operator*(const CnqsState &state) const {
        return (*hamiltonian_) * state - mu_ * state;
    }

    CnqsState trans_mult(const CnqsState &state) const { return *this * state; }

  private:
    const CnqsHamiltonian *hamiltonian_;
    double mu_;
};

CnqsHamiltonian::CnqsHamiltonian(int d, int n,
                                 std::vector<std::tuple<int, int>> edges,
                                 double g, double J)
    : d_(d), n_(n), edges_(edges), g_(g), J_(J), num_element_(IntPow(n_, d_)),
      num_edge_(edges_.size()), theta_(std::vector<double>(n)) {
    // assign theta values
    for (int i = 0; i < n_; ++i) {
        theta_[i] = (TWO_PI * i) / n_;
    }
}

void CnqsHamiltonian::inverse_power_iteration(
    const CnqsPreconditioner &preconditioner, int cg_max_iter, double cg_tol,
    int power_max_iter, double power_tol, const std::string &file_name) const {
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

    // initial estimate for eigenstate and eigenvalue
    CnqsState state = this->initialize_state();
    state = (1.0 / norm(state)) * state;

    double lambda = dot(state, *this * state);

    // initial diagnostics
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

    // save final state to file
    state.save(file_name);
}
