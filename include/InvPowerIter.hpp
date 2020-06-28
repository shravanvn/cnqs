#ifndef INV_POWER_ITER_HPP
#define INV_POWER_ITER_HPP

#include <iostream>
#include <memory>
#include <string>

#include "CnqsOperator.hpp"
#include "CnqsPreconditioner.hpp"
#include "CnqsVector.hpp"

/**
 * @brief Implements inverse power iteration algorithm
 *
 * Given an operator \f$H\f$ compute the eigenstate corresponding to eigenvalue
 * closes to \f$\mu\f$ using inverse power iteration
 * \f[
 * \psi^{(k + 1)} = \frac{(H - \mu I)^{-1} \psi^{(k)}}{\| (H - \mu I)^{-1}
 * \psi^{(k)} \|}, \quad \lambda^{(k + 1)} = \langle \psi^{(k + 1)}, H \psi^{(k
 * + 1)} \rangle
 * \f]
 * We perform the linear solve \f$(H - \mu I)^{-1} \psi^{(k)}\f$ is achieved
 * through CG iterations. A preconditioner \f$M\f$ for this iteration is also
 * needed. Iteration is stopped when
 * \f[
 * | \lambda^{(k + 1)} - \lambda^{(k)} | < \tau
 * \f]
 * for some tolerance \f$\tau\f$.
 */
class InvPowerIter {
public:
    /**
     * @brief Construct a new InvPowerIter object
     *
     * @param cnqs_operator Shared pointer to the operator \f$H\f$
     * @param shift Value of the shift \f$\mu\f$
     * @param preconditioner Shared pointer to the preconditioner \f$M\f$
     */
    InvPowerIter(
        const std::shared_ptr<const CnqsOperator> &cnqs_operator, double shift,
        const std::shared_ptr<const CnqsPreconditioner> &preconditioner)
        : operator_(cnqs_operator),
          shift_(shift),
          preconditioner_(preconditioner) {}

    /**
     * @brief Default destructor
     *
     */
    ~InvPowerIter() = default;

    /**
     * @brief Set the CG iteration parameters
     *
     * @param cg_max_iter Maximum number of CG iterations
     * @param cg_tol Tolerance for CG iterations
     */
    void SetCgIterParams(int cg_max_iter, double cg_tol) {
        cg_max_iter_ = cg_max_iter;
        cg_tol_ = cg_tol;
    }

    /**
     * @brief Set the power iteration parameters
     *
     * @param power_max_iter Maximum number of power iterations
     * @param power_tol Tolerance for power iterations
     */
    void SetPowerIterParams(int power_max_iter, double power_tol) {
        power_max_iter_ = power_max_iter;
        power_tol_ = power_tol;
    }

    /**
     * @brief Find the eigenstate with eigenvalue closes to \f$\mu\f$
     *
     * @param vector CnqsVector storing the result
     */
    void FindMinimalEigenState(CnqsVector &vector) const;

    /**
     * @brief Create a string representation of the InvPowerIter object
     *
     * @return C++ standard string with description
     */
    std::string Describe() const;

private:
    std::shared_ptr<const CnqsOperator> operator_;
    double shift_;
    std::shared_ptr<const CnqsPreconditioner> preconditioner_;
    int power_max_iter_;
    double power_tol_;
    int cg_max_iter_;
    double cg_tol_;
};

/**
 * @brief Print InvPowerIter objects to output streams (e.g. `std::cout`)
 *
 */
std::ostream &operator<<(std::ostream &os, const InvPowerIter &iterator);

#endif
