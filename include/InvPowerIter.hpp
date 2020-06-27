#ifndef INV_POWER_ITER_HPP
#define INV_POWER_ITER_HPP

#include <iostream>
#include <memory>

#include "CnqsOperator.hpp"
#include "CnqsPreconditioner.hpp"
#include "CnqsVector.hpp"

class InvPowerIter {
public:
    InvPowerIter(
        const std::shared_ptr<const CnqsOperator> &cnqs_operator, double shift,
        const std::shared_ptr<const CnqsPreconditioner> &preconditioner)
        : operator_(cnqs_operator),
          shift_(shift),
          preconditioner_(preconditioner) {}

    ~InvPowerIter() = default;

    void SetCgIterParams(int cg_max_iter, double cg_tol) {
        cg_max_iter_ = cg_max_iter;
        cg_tol_ = cg_tol;
    }

    void SetPowerIterParams(int power_max_iter, double power_tol) {
        power_max_iter_ = power_max_iter;
        power_tol_ = power_tol;
    }

    void FindMinimalEigenState(CnqsVector &vector) const;

    friend std::ostream &operator<<(std::ostream &os,
                                    const InvPowerIter &iterator);

private:
    std::shared_ptr<const CnqsOperator> operator_;
    double shift_;
    std::shared_ptr<const CnqsPreconditioner> preconditioner_;
    int power_max_iter_;
    double power_tol_;
    int cg_max_iter_;
    double cg_tol_;
};

#endif
