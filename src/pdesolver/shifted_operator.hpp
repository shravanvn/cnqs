#ifndef CNQS_PDESOLVER_SHIFTEDOPERATOR_HPP
#define CNQS_PDESOLVER_SHIFTEDOPERATOR_HPP

#include <Teuchos_RCP.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>

namespace cnqs {

namespace pdesolver {

class ShiftedOperator : public Tpetra::Operator<double, int, long> {
public:
    ShiftedOperator(
        const Teuchos::RCP<const Tpetra::Operator<double, int, long>> &A,
        double mu)
        : A_(A), mu_(mu) {}

    virtual ~ShiftedOperator() = default;

    Teuchos::RCP<const Tpetra::Map<int, long>> getDomainMap() const {
        return A_->getDomainMap();
    }

    Teuchos::RCP<const Tpetra::Map<int, long>> getRangeMap() const {
        return A_->getRangeMap();
    }

    void apply(const Tpetra::MultiVector<double, int, long> &X,
               Tpetra::MultiVector<double, int, long> &Y,
               Teuchos::ETransp mode = Teuchos::NO_TRANS, double alpha = 1.0,
               double beta = 0.0) const {
        // Y = alpha * A^mode * X + beta * Y
        A_->apply(X, Y, mode, alpha, beta);

        // Y = Y - mu * X
        Y.update(-mu_, X, 1.0);
    }

private:
    Teuchos::RCP<const Tpetra::Operator<double, int, long>> A_;
    double mu_;
};

}  // namespace pdesolver

}  // namespace cnqs

#endif
