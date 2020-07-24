#ifndef CNQS_FOURIERPROBLEM_HPP
#define CNQS_FOURIERPROBLEM_HPP

#include <memory>
#include <string>

#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Time.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>

#include "Cnqs_Network.hpp"
#include "Cnqs_Problem.hpp"

namespace Cnqs {

class FourierProblem : public Problem {
public:
    FourierProblem(const std::shared_ptr<const Cnqs::Network> &network,
                   int maxFreq,
                   const Teuchos::RCP<const Teuchos::Comm<int>> &comm);

    double runInversePowerIteration(int numPowerIter, double tolPowerIter,
                                    int numCgIter, double tolCgIter,
                                    const std::string &fileName) const;

    std::string description() const;

private:
    Teuchos::RCP<const Tpetra::Map<int, int>>
    constructMap(const Teuchos::RCP<Teuchos::Time> &timer) const;

    Teuchos::RCP<Tpetra::MultiVector<double, int, int>>
    constructInitialState(const Teuchos::RCP<const Tpetra::Map<int, int>> &map,
                          const Teuchos::RCP<Teuchos::Time> &timer) const;

    Teuchos::RCP<const Tpetra::CrsMatrix<double, int, int>>
    constructHamiltonian(const Teuchos::RCP<const Tpetra::Map<int, int>> &map,
                         const Teuchos::RCP<Teuchos::Time> &timer) const;

    Teuchos::RCP<const Tpetra::CrsMatrix<double, int, int>>
    constructPreconditioner(
        const Teuchos::RCP<const Tpetra::Map<int, int>> &map,
        const Teuchos::RCP<Teuchos::Time> &timer) const;

    std::shared_ptr<const Cnqs::Network> network_;
    int maxFreq_;
    std::vector<int> unfoldingFactors_;
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
};

} // namespace Cnqs

#endif
