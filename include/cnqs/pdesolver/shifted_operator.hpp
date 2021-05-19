#ifndef CNQS_PDESOLVER_SHIFTEDOPERATOR_HPP
#define CNQS_PDESOLVER_SHIFTEDOPERATOR_HPP

#include <Teuchos_RCP.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>

namespace cnqs {

namespace pdesolver {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class ShiftedOperator
    : public Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> {
public:
    typedef Scalar scalar_type;
    typedef LocalOrdinal local_ordinal_type;
    typedef GlobalOrdinal global_ordinal_type;
    typedef Node node_type;
    typedef Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> MV;
    typedef Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node> map_type;

private:
    typedef Tpetra::Import<LocalOrdinal, GlobalOrdinal, Node> import_type;

public:
    ShiftedOperator(const Teuchos::RCP<const Tpetra::Operator<
                        Scalar, LocalOrdinal, GlobalOrdinal, Node>> &A,
                    Scalar mu)
        : A_(A), mu_(mu) {}

    virtual ~ShiftedOperator() {}

    Teuchos::RCP<const map_type> getDomainMap() const {
        return A_->getDomainMap();
    }

    Teuchos::RCP<const map_type> getRangeMap() const {
        return A_->getRangeMap();
    }

    void apply(const MV &X, MV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
               Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
               Scalar beta = Teuchos::ScalarTraits<Scalar>::zero()) const {
        // Y = alpha * A^mode * X + beta * Y
        A_->apply(X, Y, mode, alpha, beta);
        // Y = Y - mu * X
        Y.update(-mu_, X, Teuchos::ScalarTraits<Scalar>::one());
    }

private:
    Teuchos::RCP<
        const Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
        A_;
    Scalar mu_;
};

}  // namespace pdesolver

}  // namespace cnqs

#endif
