#include "cnqs/pdesolver/tensor_train_problem.hpp"

#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

// Tensor Train problem --------------------------------------------------------

cnqs::pdesolver::TensorTrainProblem::TensorTrainProblem(
    const std::shared_ptr<const cnqs::pdesolver::Hamiltonian> &hamiltonian,
    long max_freq)
    : hamiltonian_(hamiltonian), max_freq_(max_freq) {}

double cnqs::pdesolver::TensorTrainProblem::RunInversePowerIteration(
    long num_power_iter, double tol_power_iter, long num_cg_iter,
    double tol_cg_iter, const std::string &file_name) const {
    const double sum_edge_weight = hamiltonian_->SumEdgeWeights();
    const double mu = -4.0 * hamiltonian_->SumAbsEdgeWeights();

    const tensorfact::TtTensor<double> laplacian = ConstructLaplacian();

    tensorfact::TtTensor<double> x = ConstructInitialState(2);
    double xNorm = x.FrobeniusNorm();
    x /= xNorm;

    tensorfact::TtTensor<double> y =
        ApplyShiftedHamiltonian(laplacian, sum_edge_weight, mu, x);
    double lambda = x.Dot(y) + mu;

    std::cout << std::scientific;
    std::cout << "============================================================="
                 "======================"
              << std::endl
              << " inv_iter                   lambda     d_lambda   cg_iter    "
                 "   cg_tol  max_tt_rank"
              << std::endl
              << "--------- ------------------------ ------------ --------- "
                 "------------ ------------"
              << std::endl;
    long max_rank = 0;
    for (const auto &r : x.Rank()) {
        max_rank = std::max(max_rank, r);
    }
    std::cout << std::setw(9) << 0 << " " << std::setw(24)
              << std::setprecision(16) << lambda
              << "                                     " << std::setw(12)
              << max_rank << std::endl;

    for (long i = 1; i <= num_power_iter; ++i) {
        long num_iter = num_cg_iter;
        double tol = tol_cg_iter;

        if (ConjugateGradient(laplacian, sum_edge_weight, mu, y, x, num_iter,
                              tol) != 0) {
            throw std::runtime_error("Conjugate gradient did not converge");
        }

        x = std::move(y);
        xNorm = x.FrobeniusNorm();
        x /= xNorm;

        y = ApplyShiftedHamiltonian(laplacian, sum_edge_weight, mu, x);
        const double lambda_new = x.Dot(y) + mu;

        const double d_lambda = std::abs(lambda - lambda_new);
        lambda = lambda_new;

        max_rank = 0;
        for (const auto &r : x.Rank()) {
            max_rank = std::max(max_rank, r);
        }

        std::cout << std::setw(9) << i << " " << std::setw(24)
                  << std::setprecision(16) << lambda << " " << std::setw(12)
                  << std::setprecision(6) << d_lambda << " " << std::setw(9)
                  << num_iter << " " << std::setw(12) << tol << " "
                  << std::setw(12) << max_rank << std::endl;

        // check for convergence
        if (d_lambda < tol_power_iter) {
            break;
        }
    }

    std::cout << "============================================================="
                 "========="
              << std::endl;

    if (file_name.compare("") != 0) {
        x.WriteToFile(file_name);
    }

    return lambda;
}

tensorfact::TtTensor<double>
cnqs::pdesolver::TensorTrainProblem::ConstructInitialState(long rank) const {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    tensorfact::TtTensor<double> tt_tensor(hamiltonian_->NumRotor(),
                                           2 * max_freq_ + 1, rank);

    for (auto &p : tt_tensor.Param()) {
        p = distribution(generator);
    }

    return tt_tensor;
}

tensorfact::TtTensor<double>
cnqs::pdesolver::TensorTrainProblem::ConstructLaplacian() const {
    const long num_rotor = hamiltonian_->NumRotor();
    tensorfact::TtTensor<double> laplacian(num_rotor, 2 * max_freq_ + 1, 2);

    for (long d = 0; d < num_rotor; ++d) {
        if (d == 0) {
            for (long j = 0; j < 2 * max_freq_ + 1; ++j) {
                laplacian.Param(0, j, 0, d) = std::pow(j - max_freq_, 2);
                laplacian.Param(0, j, 1, d) = 1.0;
            }
        } else if (d == num_rotor - 1) {
            for (long j = 0; j < 2 * max_freq_ + 1; ++j) {
                laplacian.Param(0, j, 0, d) = 1.0;
                laplacian.Param(1, j, 0, d) = std::pow(j - max_freq_, 2);
            }
        } else {
            for (long j = 0; j < 2 * max_freq_ + 1; ++j) {
                laplacian.Param(0, j, 0, d) = 1.0;
                laplacian.Param(1, j, 0, d) = std::pow(j - max_freq_, 2);
                laplacian.Param(0, j, 1, d) = 0.0;
                laplacian.Param(1, j, 1, d) = 1.0;
            }
        }
    }

    return laplacian;
}

tensorfact::TtTensor<double>
cnqs::pdesolver::TensorTrainProblem::ApplyShiftedHamiltonian(
    const tensorfact::TtTensor<double> &laplacian, double sum_edge_weight,
    double mu, const tensorfact::TtTensor<double> &x) const {
    tensorfact::TtTensor<double> y = laplacian * x;
    y.Round(1.0e-15);

    y *= 0.5 * hamiltonian_->VertexWeight();

    const auto &edgeList = hamiltonian_->EdgeList();
    for (const auto &edge : edgeList) {
        const auto &j = std::get<0>(edge);
        const auto &k = std::get<1>(edge);
        const auto &beta = std::get<2>(edge);

        y -= x.Shift(j, -1).Shift(k, 1);
        y -= x.Shift(j, 1).Shift(k, -1);

        y.Round(1.0e-15);
    }

    y += (2.0 * sum_edge_weight - mu) * x;
    y.Round(1.0e-15);

    return y;
}

// Returns 0 if succesful, 1 otherwise.
//
// On successful return, num_iter and tol records the number of iterations
// and acheived tolerance on convergence.
//
// Based on IML++ iterative solver template library.
int cnqs::pdesolver::TensorTrainProblem::ConjugateGradient(
    const tensorfact::TtTensor<double> &laplacian, double sum_edge_weight,
    double mu, tensorfact::TtTensor<double> &x,
    const tensorfact::TtTensor<double> &b, long &num_iter, double &tol) const {
    double resid, alpha, beta, rho, rho_1;
    tensorfact::TtTensor<double> p, z, q;

    double normb = b.FrobeniusNorm();
    tensorfact::TtTensor<double> r =
        b - ApplyShiftedHamiltonian(laplacian, sum_edge_weight, mu, x);
    r.Round(1.0e-15);

    if (std::abs(normb) < 1.0e-15) {
        normb = 1.0;
    }

    resid = r.FrobeniusNorm() / normb;
    if (resid <= tol) {
        tol = resid;
        num_iter = 0;
        return 0;
    }

    for (long i = 1; i <= num_iter; i++) {
        z = r.Copy();  // no preconditioner! M.Solve(r);
        rho = r.Dot(z);

        if (i == 1)
            p = z;
        else {
            beta = rho / rho_1;
            p = z + beta * p;
            p.Round(1.0e-15);
        }

        q = ApplyShiftedHamiltonian(laplacian, sum_edge_weight, mu, p);
        alpha = rho / p.Dot(q);

        x += alpha * p;
        x.Round(1.0e-15);

        r -= alpha * q;
        r.Round(1.0e-15);

        resid = r.FrobeniusNorm() / normb;
        if (resid <= tol) {
            tol = resid;
            num_iter = i;
            return 0;
        }

        rho_1 = rho;
    }

    tol = resid;
    return 1;
}
