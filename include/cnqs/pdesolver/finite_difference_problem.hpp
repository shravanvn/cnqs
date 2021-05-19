#ifndef CNQS_PDESOLVER_FINITEDIFFERENCEPROBLEM_HPP
#define CNQS_PDESOLVER_FINITEDIFFERENCEPROBLEM_HPP

#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "cnqs/pdesolver/hamiltonian.hpp"
#include "cnqs/pdesolver/problem.hpp"
#include "cnqs/pdesolver/shifted_operator.hpp"

namespace cnqs {

namespace pdesolver {

// =============================================================================
// Declarations
// =============================================================================

/// Basic finite-difference implementation of Problem
///
/// The space \f$[0, 2\pi]^d\f$ is discretized uniformly with \f$n\f$ grid
/// points per dimension. Then any state \f$\psi \in \mathcal{H}^1([0,
/// 2\pi]^d)\f$ that is \f$2\pi\f$-periodic along each of the dimensions is
/// discretized as a \f$d\f$-dimensional \f$n \times \cdots \times n\f$ tensor
/// \f$v\f$ with
///
/// \f[
///     v(i_0, \ldots, i_{d - 1}) \approx \psi(i_0 h, \ldots, i_{d - 1} h),
///     \quad 0 \leq i_k \leq n - 1, \quad 0 \leq k \leq d - 1, \quad h =
///     \frac{2 \pi}{n}
/// \f]
///
/// The derivative is approximated using a fourth-order five-point
/// center-difference stencil
///
/// \f[
///     f''(x) \approx \frac{-f(x - 2h) + 16 f(x - h) - 30 f(x) + 16 f(x + h)
///     - f(x + 2h)}{h^2}
/// \f]
///
/// Wrap-around is employed at the dimension edges to capture periodicy of the
/// state \f$\psi\f$ in the tensor \f$v\f$.
///
/// @note This class is built on top of Trilinos to support distributed
/// computing.
template <class Real, class LocalOrdinal, class GlobalOrdinal, class Node>
class FiniteDifferenceProblem : public Problem<Real, GlobalOrdinal> {
public:
    /// @brief Construct a FiniteDifferenceProblem given hamiltonian and
    /// discretization
    ///
    /// @param [in] hamiltonian Quantum rotor hamiltonian \f$(\mathcal{V},
    /// \mathcal{E})\f$, implemented in Hamiltonian.
    ///
    /// @param [in] laplacianFactor Prefactor \f$h\f$ of the Laplacian; must be
    /// non-negative.
    ///
    /// @param [in] numGridPoint Number of grid points per dimension, \f$n\f$.
    ///
    /// @param [in] comm Communicator.
    FiniteDifferenceProblem(
        const std::shared_ptr<const Hamiltonian<Real, GlobalOrdinal>>
            &hamiltonian,
        GlobalOrdinal numGridPoint,
        const Teuchos::RCP<const Teuchos::Comm<int>> &comm);

    Real runInversePowerIteration(int numPowerIter, Real tolPowerIter,
                                  int numCgIter, Real tolCgIter,
                                  const std::string &fileName) const;

private:
    Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
    constructMap(const Teuchos::RCP<Teuchos::Time> &timer) const;

    Teuchos::RCP<Tpetra::MultiVector<Real, LocalOrdinal, GlobalOrdinal, Node>>
    constructInitialState(
        const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
            &map,
        const Teuchos::RCP<Teuchos::Time> &timer) const;

    Teuchos::RCP<
        const Tpetra::CrsMatrix<Real, LocalOrdinal, GlobalOrdinal, Node>>
    constructHamiltonian(
        const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
            &map,
        const Teuchos::RCP<Teuchos::Time> &timer) const;

    std::shared_ptr<const Hamiltonian<Real, GlobalOrdinal>> hamiltonian_;
    GlobalOrdinal numGridPoint_;
    std::vector<GlobalOrdinal> unfoldingFactors_;
    std::vector<Real> theta_;
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
};

// =============================================================================
// Implementations
// =============================================================================

template <class Real, class LocalOrdinal, class GlobalOrdinal, class Node>
FiniteDifferenceProblem<Real, LocalOrdinal, GlobalOrdinal, Node>::
    FiniteDifferenceProblem(
        const std::shared_ptr<const Hamiltonian<Real, GlobalOrdinal>>
            &hamiltonian,
        GlobalOrdinal numGridPoint,
        const Teuchos::RCP<const Teuchos::Comm<int>> &comm)
    : hamiltonian_(hamiltonian),
      numGridPoint_(numGridPoint),
      unfoldingFactors_(
          std::vector<GlobalOrdinal>(hamiltonian->numRotor() + 1)),
      theta_(std::vector<Real>(numGridPoint)),
      comm_(comm) {
    TEUCHOS_TEST_FOR_EXCEPTION(numGridPoint_ < 5, std::domain_error,
                               "Need at least 5 grid points");

    const GlobalOrdinal numRotor = hamiltonian_->numRotor();
    unfoldingFactors_[0] = 1;
    for (GlobalOrdinal d = 0; d < numRotor; ++d) {
        unfoldingFactors_[d + 1] = numGridPoint_ * unfoldingFactors_[d];
    }

    Real dTheta = 8.0 * std::atan(1.0) / numGridPoint_;
    for (GlobalOrdinal n = 0; n < numGridPoint_; ++n) {
        theta_[n] = n * dTheta;
    }
}

template <class Real, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
FiniteDifferenceProblem<Real, LocalOrdinal, GlobalOrdinal, Node>::constructMap(
    const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    const GlobalOrdinal numRotor = hamiltonian_->numRotor();

    const int rank = comm_->getRank();
    const int size = comm_->getSize();

    const GlobalOrdinal globalIndexBase = 0;
    const GlobalOrdinal globalNumElem = unfoldingFactors_[numRotor];

    GlobalOrdinal localNumElem = (globalNumElem + size - 1) / size;
    if (rank == size - 1) {
        localNumElem = globalNumElem - rank * localNumElem;
    }

    auto map = Teuchos::rcp(new Tpetra::Map<LocalOrdinal, GlobalOrdinal>(
        globalNumElem, localNumElem, globalIndexBase, comm_));
    TEUCHOS_TEST_FOR_EXCEPTION(
        !map->isContiguous(), std::logic_error,
        "==cnqs::pdesolver::FiniteDifferenceProblem::ConstructMap== Could not "
        "construct a contiguous map");

    return map.getConst();
}

template <class Real, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Tpetra::MultiVector<Real, LocalOrdinal, GlobalOrdinal, Node>>
FiniteDifferenceProblem<Real, LocalOrdinal, GlobalOrdinal, Node>::
    constructInitialState(
        const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
            &map,
        const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    const GlobalOrdinal numRotor = hamiltonian_->numRotor();
    auto state = Teuchos::rcp(
        new Tpetra::MultiVector<Real, LocalOrdinal, GlobalOrdinal, Node>(map,
                                                                         1));

    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<Real> distribution(-1.0, 1.0);

    {
        const GlobalOrdinal numLocalRows = map->getNodeNumElements();

        state->sync_host();
        auto x_2d = state->getLocalViewHost();
        state->modify_host();

        for (GlobalOrdinal i = 0; i < numLocalRows; ++i) {
            x_2d(i, 0) = distribution(generator);
        }

        state->sync_device();
    }

    std::vector<Real> stateNorm(1);
    state->norm2(stateNorm);

    std::vector<Real> scaleFactor(1);
    scaleFactor[0] = 1.0 / stateNorm[0];
    state->scale(scaleFactor);

    return state;
}

template <class Real, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Tpetra::CrsMatrix<Real, LocalOrdinal, GlobalOrdinal, Node>>
FiniteDifferenceProblem<Real, LocalOrdinal, GlobalOrdinal, Node>::
    constructHamiltonian(
        const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
            &map,
        const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    // hamiltonian parameters
    const GlobalOrdinal numRotor = hamiltonian_->numRotor();
    const Real vertexWeight = hamiltonian_->vertexWeight();
    const std::vector<std::tuple<GlobalOrdinal, GlobalOrdinal, Real>>
        &edgeList = hamiltonian_->edgeList();

    // finite difference discretization parameters
    const Real h = theta_[1] - theta_[0];
    const Real fact = -0.5 * vertexWeight / (12.0 * h * h);

    // allocate memory for the Hamiltonian
    const GlobalOrdinal numEntryPerRow = 4 * numRotor + 1;
    auto hamiltonian = Teuchos::rcp(
        new Tpetra::CrsMatrix<Real, LocalOrdinal, GlobalOrdinal, Node>(
            map, numEntryPerRow, Tpetra::StaticProfile));

    // assemble the Hamiltonian, one row at a time
    const GlobalOrdinal numLocalRows = map->getNodeNumElements();
    for (GlobalOrdinal localRowId = 0; localRowId < numLocalRows;
         ++localRowId) {
        const GlobalOrdinal globalRowIdLin = map->getGlobalElement(localRowId);

        // unwrap linear index i -> dimensional index (i_0, ..., i_{d - 1})
        std::vector<GlobalOrdinal> globalRowIdDim(numRotor);
        globalRowIdDim[0] = globalRowIdLin;
        for (GlobalOrdinal d = 0; d < numRotor; ++d) {
            if (d < numRotor - 1) {
                globalRowIdDim[d + 1] = globalRowIdDim[d] / numGridPoint_;
            }
            globalRowIdDim[d] %= numGridPoint_;
        }

        // collect column indices and values for current row
        std::vector<GlobalOrdinal> currentRowColumnIndices(numEntryPerRow);
        std::vector<Real> currentRowValues(numEntryPerRow);

        currentRowColumnIndices[0] = globalRowIdLin;
        currentRowValues[0] = -30.0 * fact * numRotor;
        for (const auto &edge : edgeList) {
            const GlobalOrdinal j = std::get<0>(edge);
            const GlobalOrdinal k = std::get<1>(edge);
            const Real beta = std::get<2>(edge);

            currentRowValues[0] +=
                beta * (2.0 - 2.0 * std::cos(theta_[globalRowIdDim[j]] -
                                             theta_[globalRowIdDim[k]]));
        }

        GlobalOrdinal currentRowNonZeroCount = 1;

        for (GlobalOrdinal k = -2; k <= 2; ++k) {
            if (k != 0) {
                for (GlobalOrdinal d = 0; d < numRotor; ++d) {
                    // compute the global dimensional index of k-th neighbor
                    // along d-th dimension
                    std::vector<GlobalOrdinal> globalColumnIdDim(
                        globalRowIdDim);
                    globalColumnIdDim[d] += k;
                    if (globalColumnIdDim[d] < 0) {
                        globalColumnIdDim[d] += numGridPoint_;
                    } else if (globalColumnIdDim[d] >= numGridPoint_) {
                        globalColumnIdDim[d] -= numGridPoint_;
                    }

                    // convert the global dimensional index to global linear
                    // index
                    GlobalOrdinal globalColumnIdLin = 0;
                    for (GlobalOrdinal d = 0; d < numRotor; ++d) {
                        globalColumnIdLin +=
                            unfoldingFactors_[d] * globalColumnIdDim[d];
                    }
                    currentRowColumnIndices[currentRowNonZeroCount] =
                        globalColumnIdLin;

                    // compute the correct nonzero value of the matrix entry
                    if (std::abs(k) == 1) {
                        currentRowValues[currentRowNonZeroCount] = 16.0 * fact;
                    } else if (std::abs(k) == 2) {
                        currentRowValues[currentRowNonZeroCount] = -fact;
                    }

                    ++currentRowNonZeroCount;
                }
            }
        }

        TEUCHOS_TEST_FOR_EXCEPTION(
            currentRowNonZeroCount != numEntryPerRow, std::logic_error,
            "==cnqs::pdesolver::FiniteDifferenceProblem::ConstructOperators== "
            "Could not match number of non-zero entries in a row");

        hamiltonian->insertGlobalValues(globalRowIdLin, currentRowColumnIndices,
                                        currentRowValues);
    }

    hamiltonian->fillComplete();
    return hamiltonian.getConst();
}

template <class Real, class LocalOrdinal, class GlobalOrdinal, class Node>
Real FiniteDifferenceProblem<Real, LocalOrdinal, GlobalOrdinal, Node>::
    runInversePowerIteration(int maxPowerIter, Real tolPowerIter, int maxCgIter,
                             Real tolCgIter,
                             const std::string &fileName) const {
    // create timers
    Teuchos::RCP<Teuchos::Time> mapTime =
        Teuchos::TimeMonitor::getNewCounter("CNQS: Map Construction Time");
    Teuchos::RCP<Teuchos::Time> initialStateTime =
        Teuchos::TimeMonitor::getNewCounter(
            "CNQS: Initial State Construction Time");
    Teuchos::RCP<Teuchos::Time> hamiltonianTime =
        Teuchos::TimeMonitor::getNewCounter(
            "CNQS: Hamiltonian Construction Time");
    Teuchos::RCP<Teuchos::Time> powerIterationTime =
        Teuchos::TimeMonitor::getNewCounter(
            "CNQS: Inverse Power Iteration Time");

    // construct map
    auto map = constructMap(mapTime);

    // construct initial state
    auto x = constructInitialState(map, initialStateTime);

    // construct Hamiltonian
    auto H = constructHamiltonian(map, hamiltonianTime);

    // get lower bound for eigenvalues
    const Real mu = -4 * hamiltonian_->sumAbsEdgeWeights();

    std::vector<Real> lambda(1);
    {
        // create local timer
        Teuchos::TimeMonitor localTimer(*powerIterationTime);

        // compute initial guess for lambda
        auto y = Teuchos::rcp(
            new Tpetra::MultiVector<Real, LocalOrdinal, GlobalOrdinal, Node>(
                x->getMap(), 1));
        H->apply(*x, *y);
        x->dot(*y, lambda);

        if (comm_->getRank() == 0) {
            std::cout << std::scientific;
            std::cout << "====================================================="
                         "================="
                      << std::endl
                      << " inv_iter                   lambda     d_lambda   "
                         "cg_iter       cg_tol"
                      << std::endl
                      << "--------- ------------------------ ------------ "
                         "--------- ------------"
                      << std::endl;
            std::cout << std::setw(9) << 0 << " " << std::setw(24)
                      << std::setprecision(16) << lambda[0] << std::endl;
        }

        // construct shifted Hamiltonian
        auto A = Teuchos::rcp(
            new ShiftedOperator<Real, LocalOrdinal, GlobalOrdinal, Node>(H,
                                                                         mu));

        // create solver parameters
        auto params = Teuchos::rcp(new Teuchos::ParameterList());
        params->set("Maximum Iterations", maxCgIter);
        params->set("Convergence Tolerance", tolCgIter);

        for (int i = 1; i <= maxPowerIter; ++i) {
            // create linear problem
            auto z = Teuchos::rcp(
                new Tpetra::MultiVector<Real, LocalOrdinal, GlobalOrdinal,
                                        Node>(x->getMap(), 1));
            auto problem = Teuchos::rcp(
                new Belos::LinearProblem<
                    Real,
                    Tpetra::MultiVector<Real, LocalOrdinal, GlobalOrdinal,
                                        Node>,
                    Tpetra::Operator<Real, LocalOrdinal, GlobalOrdinal, Node>>(
                    A, z, x.getConst()));
            problem->setHermitian();
            problem->setProblem();

            // create CG solver
            Belos::PseudoBlockCGSolMgr<
                Real,
                Tpetra::MultiVector<Real, LocalOrdinal, GlobalOrdinal, Node>,
                Tpetra::Operator<Real, LocalOrdinal, GlobalOrdinal, Node>>
                solver(problem, params);

            // solve
            Belos::ReturnType status = solver.solve();

            TEUCHOS_TEST_FOR_EXCEPTION(
                status != Belos::ReturnType::Converged, std::runtime_error,
                "==cnqs::pdesolver::FiniteDifferenceProblem::"
                "runInversePowerIteration== CG iteration did not converge");

            // copy x = solution
            x = problem->getLHS();

            // normalize x = x / norm2(x)
            std::vector<Real> xNorm(1);
            x->norm2(xNorm);

            std::vector<Real> scaleFactor(1);
            scaleFactor[0] = 1.0 / xNorm[0];
            x->scale(scaleFactor);

            // compute new estimate for lambda
            std::vector<Real> lambdaNew(1);

            H->apply(*x, *y);
            x->dot(*y, lambdaNew);
            const Real dLambda = std::abs(lambda[0] - lambdaNew[0]);
            lambda[0] = lambdaNew[0];

            if (comm_->getRank() == 0) {
                std::cout << std::setw(9) << i << " " << std::setw(24)
                          << std::setprecision(16) << lambda[0] << " "
                          << std::setw(12) << std::setprecision(6) << dLambda
                          << " " << std::setw(9) << solver.getNumIters() << " "
                          << std::setw(12) << solver.achievedTol() << std::endl;
            }

            // check for convergence
            if (dLambda < tolPowerIter) {
                break;
            }
        }

        const Real h = theta_[1] - theta_[0];
        x->scale(1.0 / (h * h));

        if (comm_->getRank() == 0) {
            std::cout << "====================================================="
                         "================="
                      << std::endl;
        }
    }

    // save estimated state
    if (fileName.compare("") != 0) {
        Tpetra::MatrixMarket::Writer<
            Tpetra::CrsMatrix<Real, LocalOrdinal, GlobalOrdinal, Node>>::
            writeDenseFile(fileName, *x, "low_eigen_state",
                           "Estimated lowest energy eigenstate of Hamiltonian");
    }

    // get time summary
    Teuchos::TimeMonitor::summarize();

    return lambda[0];
}

}  // namespace pdesolver

}  // namespace cnqs

#endif
