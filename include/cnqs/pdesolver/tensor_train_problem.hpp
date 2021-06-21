#ifndef CNQS_PDESOVLER_TENSORTRAINPROBLEM_HPP
#define CNQS_PDESOLVER_TENSORTRAINPROBLEM_HPP

#include <memory>
#include <string>

#include "cnqs/pdesolver/hamiltonian.hpp"
#include "cnqs/pdesolver/problem.hpp"
#include "tensorfact/tt_tensor.hpp"

namespace cnqs {

namespace pdesolver {

class TensorTrainProblem : public Problem {
public:
    TensorTrainProblem(const std::shared_ptr<const Hamiltonian> &hamiltonian,
                       long max_freq);

    double RunInversePowerIteration(long num_power_iter, double tol_power_iter,
                                    long num_cg_iter, double tol_cg_iter,
                                    const std::string &file_name) const;

private:
    tensorfact::TtTensor<double> ConstructInitialState(long rank) const;

    tensorfact::TtTensor<double> ConstructLaplacian() const;

    tensorfact::TtTensor<double> ApplyShiftedHamiltonian(
        const tensorfact::TtTensor<double> &laplacian, double sum_edge_weight,
        double mu, const tensorfact::TtTensor<double> &x) const;

    int ConjugateGradient(const tensorfact::TtTensor<double> &laplacian,
                          double sum_edge_weight, double mu,
                          tensorfact::TtTensor<double> &x,
                          const tensorfact::TtTensor<double> &b, long &num_iter,
                          double &tol) const;

    std::shared_ptr<const Hamiltonian> hamiltonian_;
    long max_freq_;
};

}  // namespace pdesolver

}  // namespace cnqs

#endif
