#include "InvPowerIter.hpp"

#include <iomanip>
#include <stdexcept>

#include "ImlCnqsInterface.hpp"
#include "cg.h"

void InvPowerIter::FindMinimalEigenState(CnqsVector &vector) const {
    // output floating point numbers in scientific format
    std::cout << std::scientific;

    // output table header
    std::cout << "============================================================="
                 "========="
              << std::endl
              << " inv_iter                   lambda     d_lambda   cg_iter    "
                 "   cg_tol"
              << std::endl
              << "--------- ------------------------ ------------ --------- "
                 "------------"
              << std::endl;

    // construct initial state based on operator
    operator_->ConstructInitialState(vector);

    // create IML++ comptaible operator and preconditioner
    ImlCnqsOperator iml_operator(operator_, shift_);
    ImlCnqsPreconditioner iml_preconditioner(preconditioner_);

    // allocate memory for one additional vector
    CnqsVector new_vector(vector.Size());

    // intialize inverse power iteration
    vector.Normalize();
    operator_->Apply(vector, new_vector);
    double lambda = vector.Dot(new_vector);
    std::cout << std::setw(9) << 0 << " " << std::setw(24)
              << std::setprecision(16) << lambda << std::endl;

    // main loop of inverse power iteration
    for (int power_iter = 1; power_iter <= power_max_iter_; ++power_iter) {
        // initialize CG parameters
        int cg_max_iter = cg_max_iter_;
        double cg_tol = cg_tol_;

        // solve using CG and normalize
        int cg_return = CG(iml_operator, new_vector, vector, iml_preconditioner,
                           cg_max_iter, cg_tol);

        if (cg_return != 0) {
            throw std::runtime_error("CG iteration did not converge");
        }

        new_vector.Normalize();

        // compute new eigenvalue
        operator_->Apply(new_vector, vector);
        double new_lambda = new_vector.Dot(vector);
        double d_lambda = std::abs(lambda - new_lambda);

        // report diagnostics
        std::cout << std::setw(9) << power_iter << " " << std::setw(24)
                  << std::setprecision(16) << new_lambda << " " << std::setw(12)
                  << std::setprecision(6) << d_lambda << " " << std::setw(9)
                  << cg_max_iter << " " << std::setw(12) << cg_tol << std::endl;

        // check for convergence
        if (d_lambda < power_tol_) {
            break;
        }

        // update eigenvalue and eigenstate estiamtes
        vector = new_vector;
        lambda = new_lambda;
    }

    // output table footer
    std::cout << "============================================================="
                 "========="
              << std::endl;
}

std::ostream &operator<<(std::ostream &os, const InvPowerIter &iterator) {
    os << "==InvPowerIter==" << std::endl
       << *(iterator.operator_) << std::endl
       << *(iterator.preconditioner_) << std::endl
       << "    shift          : " << iterator.shift_ << std::endl
       << "    cg_max_iter    : " << iterator.cg_max_iter_ << std::endl
       << "    cg_tol         : " << iterator.cg_tol_ << std::endl
       << "    power_max_iter : " << iterator.power_max_iter_ << std::endl
       << "    power_tol      : " << iterator.power_tol_ << std::endl
       << "==InvPowerIter==";

    return os;
}
