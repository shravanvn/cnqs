#ifndef CNQS_SHIFTEDOPERATOR_HPP
#define CNQS_SHIFTEDOPERATOR_HPP

#include <Teuchos_RCP.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>

namespace Cnqs {

class ShiftedOperator : public Tpetra::Operator<> {
public:
    typedef Tpetra::Operator<>::scalar_type scalar_type;
    typedef Tpetra::Operator<>::local_ordinal_type local_ordinal_type;
    typedef Tpetra::Operator<>::global_ordinal_type global_ordinal_type;
    typedef Tpetra::Operator<>::node_type node_type;
    typedef Tpetra::MultiVector<scalar_type, local_ordinal_type,
                                global_ordinal_type, node_type>
        MV;
    typedef Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type>
        map_type;

private:
    typedef Tpetra::Import<local_ordinal_type, global_ordinal_type, node_type>
        import_type;

public:
    ShiftedOperator(
        const Teuchos::RCP<const Tpetra::Operator<
            scalar_type, local_ordinal_type, global_ordinal_type, node_type>>
            &A,
        scalar_type mu)
        : A_(A), mu_(mu) {}

    virtual ~ShiftedOperator() {}

    Teuchos::RCP<const map_type> getDomainMap() const {
        return A_->getDomainMap();
    }

    Teuchos::RCP<const map_type> getRangeMap() const {
        return A_->getRangeMap();
    }

    void
    apply(const MV &X, MV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
          scalar_type alpha = Teuchos::ScalarTraits<scalar_type>::one(),
          scalar_type beta = Teuchos::ScalarTraits<scalar_type>::zero()) const {
        // Y = alpha * A^mode * X + beta * Y
        A_->apply(X, Y, mode, alpha, beta);
        // Y = Y - mu * X
        Y.update(-mu_, X, Teuchos::ScalarTraits<scalar_type>::one());
    }

private:
    Teuchos::RCP<const Tpetra::Operator<scalar_type, local_ordinal_type,
                                        global_ordinal_type, node_type>>
        A_;
    scalar_type mu_;
};

} // namespace Cnqs

#endif
