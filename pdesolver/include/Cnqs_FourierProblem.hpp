#ifndef CNQS_FOURIERPROBLEM_HPP
#define CNQS_FOURIERPROBLEM_HPP

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
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "Cnqs_Network.hpp"
#include "Cnqs_Problem.hpp"
#include "Cnqs_ShiftedOperator.hpp"

namespace Cnqs {

/**
 * @brief Fourier-series based implementation of Problem
 *
 * Since the state \f$\psi \in \mathcal{H}^1([0, 2\pi]^d)\f$ is assumed to be
 * \f$2\pi\f$-periodic along each of the dimensions, it can be represented in
 * terms of a Fourier series, with the Fourier coefficients
 * \f$\hat{\psi}(\omega)\f$ defined for \f$\omega \in \mathbb{Z}^d\f$. The
 * Hamiltonian for these coefficients is given by
 *
 * \f[
 *     \hat{H} \hat{\psi}(\omega) = \frac{1}{2} \| \omega \|^2
 *     \hat{\psi}(\omega) + \sum_{(j, k) \in \mathcal{E}} \beta_{jk}
 *     [\hat{\psi}(\omega) - \hat{\psi}(\omega + e_j - e_k) - \hat{\psi}(\omega
 *     - e_j + e_k)]
 * \f]
 *
 * where \f$e_j\f$ is the \f$j\f$-th standard basis of \f$\mathbb{R}^d\f$. We
 * truncate this lattice by restricting \f$-\omega_\text{max} \leq \omega_j \leq
 * \omega_\text{max}\f$ (in other words, setting \f$\hat{\psi}(\omega) = 0\f$
 * for \f$\omega\f$ outside this limit). This leads to a \f$d\f$ dimensional
 * \f$(2 \omega_\text{max} + 1) \times \cdots \times (2 \omega_\text{max} +
 * 1)\f$ tensor with
 *
 * \f[
 *     v(i_0, \ldots, i_{d - 1}) = \hat{\psi}(i_0 - \omega_\text{max}, \ldots,
 *     i_{d - 1} - \omega_\text{max}), \quad 0 \leq i_k \leq 2
 *     \omega_\text{max}, \quad 0 \leq k \leq d - 1
 * \f]
 *
 * @note This class is built on top of Trilinos to support distributed
 * computing.
 */
template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class FourierProblem : public Problem<Scalar, GlobalOrdinal> {
public:
    /**
     * @brief Construct a FourierProblem given network and cutoff frequency
     *
     * @param [in] network Quantum rotor network \f$(\mathcal{V},
     * \mathcal{E})\f$, implemented in Network.
     * @param [in] maxFreq Cutoff frequency, \f$\omega_\text{max}\f$.
     * @param [in] comm Communicator.
     */
    FourierProblem(
        const std::shared_ptr<const Network<Scalar, GlobalOrdinal>> &network,
        GlobalOrdinal maxFreq,
        const Teuchos::RCP<const Teuchos::Comm<int>> &comm);

    /**
     * @brief Run inverse power iteration
     *
     * Computing the lowest-energy eigenpair of the Hamiltonian \f$\hat{H}\f$
     * can be achieved using inverse power iteration. Given a lower-bound
     * \f$\mu\f$ of the eigenvalues of this operator, one starts at an arbitrary
     * state \f$\hat{\psi}_0\f$ and iterates
     *
     * \f[
     *     \hat{\psi}_{k + 1} = \frac{(\hat{H} - \mu I)^{-1} \hat{\psi}_k}{\|
     *     (\hat{H} - \mu I)^{-1} \hat{\psi}_k \|}, \quad \lambda_{k + 1} =
     *     \langle \hat{\psi}_{k + 1}, \hat{H} \hat{\psi}_{k + 1} \rangle
     * \f]
     *
     * This iteration is stopped when
     *
     * -    Iteration counter \f$k\f$ reaches a maximal value
     *      \f$k_\text{power}\f$
     * -    The difference between the eigenvalue estimates drop below a
     *      threshold: \f$| \lambda_{k + 1} - \lambda_k | < \tau_\text{power}\f$
     *
     * To solve the linear system in the power iteration, we employ the CG
     * iterative solver with maxinum number of iterations \f$k_\text{CG}\f$ and
     * tolerance \f$\tau_\text{CG}\f$. To accelerate the CG iterations, we
     * empoly a preconditioner defined as
     *
     * \f[
     *     \hat{M} \hat{\psi}(\omega) = \sqrt{\frac{1}{2} \| \omega \|^2 +
     *     \sum_{(j, k) \in \mathcal{E}} \beta_{jk} - \mu} \;\;
     *     \hat{\psi}(\omega)
     * \f]
     *
     * and solve the linear system
     *
     * \f[
     *     \hat{M}^{-1} (\hat{H} - \mu I) \hat{M}^{-1} \hat{M}
     *     \hat{\phi}_{k + 1} = \hat{M}^{-1} \hat{\psi}_k
     * \f]
     *
     * @param [in] maxPowerIter Maximum number of power iterations,
     * \f$k_\text{power}\f$.
     * @param [in] tolPowerIter Tolerance at which to stop power iteration,
     * \f$\tau_\text{power}\f$.
     * @param [in] maxCgIter Number of CG iterations to run per power iteration,
     * \f$k_\text{CG}\f$.
     * @param [in] tolCgIter Tolerance at which to stop CG iteration,
     * \f$\tau_\text{CG}\f$.
     * @param [in] fileName File where to store the estimated eigenstate. If set
     * to the empty string `""`, then the eigenstate is not saved.
     * @return Estimated smallest eigenvalue of the Hamiltonian.
     */
    Scalar runInversePowerIteration(int numPowerIter, Scalar tolPowerIter,
                                    int numCgIter, Scalar tolCgIter,
                                    const std::string &fileName) const;

    nlohmann::json description() const;

    /**
     * @brief Print FourierProblem object to output streams
     *
     * @param [in,out] os Output stream
     * @param [in] problem Quantum rotor network Hamiltonian eigenproblem
     * discretized in Fourier domain
     * @return Output stream
     */
    friend std::ostream &operator<<(std::ostream &os,
                                    const FourierProblem &problem) {
        os << problem.description().dump(4);
        return os;
    }

private:
    Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
    constructMap(const Teuchos::RCP<Teuchos::Time> &timer) const;

    Teuchos::RCP<Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
    constructInitialState(
        const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
            &map,
        const Teuchos::RCP<Teuchos::Time> &timer) const;

    Teuchos::RCP<
        const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
    constructHamiltonian(
        const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
            &map,
        const Teuchos::RCP<Teuchos::Time> &timer) const;

    Teuchos::RCP<
        const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
    constructPreconditioner(
        const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
            &map,
        const Teuchos::RCP<Teuchos::Time> &timer) const;

    std::shared_ptr<const Network<Scalar, GlobalOrdinal>> network_;
    GlobalOrdinal maxFreq_;
    std::vector<GlobalOrdinal> unfoldingFactors_;
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
};

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
FourierProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>::FourierProblem(
    const std::shared_ptr<const Network<Scalar, GlobalOrdinal>> &network,
    GlobalOrdinal maxFreq, const Teuchos::RCP<const Teuchos::Comm<int>> &comm)
    : network_(network),
      maxFreq_(maxFreq),
      unfoldingFactors_(std::vector<GlobalOrdinal>(network->numRotor() + 1)),
      comm_(comm) {
    TEUCHOS_TEST_FOR_EXCEPTION(maxFreq_ < 1, std::domain_error,
                               "==Cnqs::FourierProblem::FourierProblem== Need "
                               "maximum frequency of at least one");

    const GlobalOrdinal numRotor = network_->numRotor();
    unfoldingFactors_[0] = 1;
    for (GlobalOrdinal d = 0; d < numRotor; ++d) {
        unfoldingFactors_[d + 1] = (2 * maxFreq_ + 1) * unfoldingFactors_[d];
    }
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
FourierProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>::constructMap(
    const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    const GlobalOrdinal numRotor = network_->numRotor();

    const int rank = comm_->getRank();
    const int size = comm_->getSize();

    const GlobalOrdinal globalIndexBase = 0;
    const GlobalOrdinal globalNumElem = unfoldingFactors_[numRotor];

    GlobalOrdinal localNumElem = (globalNumElem + size - 1) / size;
    if (rank == size - 1) {
        localNumElem = globalNumElem - rank * localNumElem;
    }

    auto map = Teuchos::rcp(new Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>(
        globalNumElem, localNumElem, globalIndexBase, comm_));
    TEUCHOS_TEST_FOR_EXCEPTION(
        !map->isContiguous(), std::logic_error,
        "==Cnqs::FourierProblem::ConstructMap== Could not "
        "construct a contiguous map");

    return map.getConst();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
FourierProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    constructInitialState(
        const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
            &map,
        const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    const GlobalOrdinal numRotor = network_->numRotor();
    auto state = Teuchos::rcp(
        new Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>(map,
                                                                           1));
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<Scalar> distribution(-1.0, 1.0);

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

    std::vector<Scalar> stateNorm(1);
    state->norm2(stateNorm);

    std::vector<Scalar> scaleFactor(1);
    scaleFactor[0] = 1.0 / stateNorm[0];
    state->scale(scaleFactor);

    return state;
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
FourierProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>::constructHamiltonian(
    const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
        &map,
    const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    // network parameters
    const GlobalOrdinal numRotor = network_->numRotor();
    const std::vector<std::tuple<GlobalOrdinal, GlobalOrdinal, Scalar>>
        &edgeList = network_->edgeList();
    const GlobalOrdinal numEdge = edgeList.size();

    Scalar betaSum = 0.0;
    for (const auto &edge : edgeList) {
        const Scalar beta = std::get<2>(edge);
        betaSum += beta;
    }

    // allocate memory for the Hamiltonian
    const GlobalOrdinal numEntryPerRow = 2 * numEdge + 1;
    auto hamiltonian = Teuchos::rcp(
        new Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(
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
                globalRowIdDim[d + 1] = globalRowIdDim[d] / (2 * maxFreq_ + 1);
            }
            globalRowIdDim[d] %= 2 * maxFreq_ + 1;
        }

        // collect column indices and values for current row
        std::vector<GlobalOrdinal> currentRowColumnIndices(numEntryPerRow);
        std::vector<Scalar> currentRowValues(numEntryPerRow);

        currentRowColumnIndices[0] = globalRowIdLin;
        currentRowValues[0] = 0.0;
        for (GlobalOrdinal d = 0; d < numRotor; ++d) {
            currentRowValues[0] += std::pow(globalRowIdDim[d] - maxFreq_, 2.0);
        }
        currentRowValues[0] *= 0.5;
        currentRowValues[0] += betaSum;

        GlobalOrdinal currentRowNonZeroCount = 1;

        for (GlobalOrdinal e = 0; e < numEdge; ++e) {
            const GlobalOrdinal j = std::get<0>(edgeList[e]);
            const GlobalOrdinal k = std::get<1>(edgeList[e]);
            const Scalar beta = std::get<2>(edgeList[e]);

            for (GlobalOrdinal f = -1; f <= 1; ++f) {
                if (f != 0) {
                    std::vector<GlobalOrdinal> globalColumnIdDim(
                        globalRowIdDim);
                    globalColumnIdDim[j] += (f == -1) ? 1 : -1;
                    globalColumnIdDim[k] -= (f == -1) ? 1 : -1;

                    if ((globalColumnIdDim[j] >= 0) &&
                        (globalColumnIdDim[j] <= 2 * maxFreq_) &&
                        (globalColumnIdDim[k] >= 0) &&
                        (globalColumnIdDim[k] <= 2 * maxFreq_)) {
                        GlobalOrdinal globalColumnIdLin = 0;
                        for (GlobalOrdinal d = 0; d < numRotor; ++d) {
                            globalColumnIdLin +=
                                unfoldingFactors_[d] * globalColumnIdDim[d];
                        }

                        currentRowColumnIndices[currentRowNonZeroCount] =
                            globalColumnIdLin;
                        currentRowValues[currentRowNonZeroCount] = -beta;

                        ++currentRowNonZeroCount;
                    }
                }
            }
        }

        hamiltonian->insertGlobalValues(globalRowIdLin, currentRowColumnIndices,
                                        currentRowValues);
    }

    hamiltonian->fillComplete();
    return hamiltonian.getConst();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
FourierProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    constructPreconditioner(
        const Teuchos::RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
            &map,
        const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    // network parameters
    const GlobalOrdinal numRotor = network_->numRotor();
    const Scalar eigValLowerBound = network_->eigValLowerBound();

    Scalar betaSum = 0.0;
    for (const auto &edge : network_->edgeList()) {
        const Scalar beta = std::get<2>(edge);
        betaSum += beta;
    }

    // allocate memory for the preconditioner
    auto preconditioner = Teuchos::rcp(
        new Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(
            map, 1, Tpetra::StaticProfile));

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
                globalRowIdDim[d + 1] = globalRowIdDim[d] / (2 * maxFreq_ + 1);
            }
            globalRowIdDim[d] %= 2 * maxFreq_ + 1;
        }

        // collect column indices and values for current row
        std::vector<GlobalOrdinal> currentRowColumnIndices(1);
        std::vector<Scalar> currentRowValues(1);

        currentRowColumnIndices[0] = globalRowIdLin;
        currentRowValues[0] = 0.0;
        for (GlobalOrdinal d = 0; d < numRotor; ++d) {
            currentRowValues[0] += std::pow(globalRowIdDim[d] - maxFreq_, 2.0);
        }
        currentRowValues[0] = std::sqrt(
            1.0 / (0.5 * currentRowValues[0] + betaSum - eigValLowerBound));

        preconditioner->insertGlobalValues(
            globalRowIdLin, currentRowColumnIndices, currentRowValues);
    }

    preconditioner->fillComplete();
    return preconditioner.getConst();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Scalar FourierProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    runInversePowerIteration(int maxPowerIter, Scalar tolPowerIter,
                             int maxCgIter, Scalar tolCgIter,
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
    Teuchos::RCP<Teuchos::Time> preconditionerTime =
        Teuchos::TimeMonitor::getNewCounter(
            "CNQS: Preconditioner Construction Time");
    Teuchos::RCP<Teuchos::Time> powerIterationTime =
        Teuchos::TimeMonitor::getNewCounter(
            "CNQS: Inverse Power Iteration Time");

    auto map = constructMap(mapTime);
    auto x = constructInitialState(map, initialStateTime);
    auto H = constructHamiltonian(map, hamiltonianTime);
    auto M = constructPreconditioner(map, preconditionerTime);

    std::vector<Scalar> lambda(1);
    {
        // create local timer
        Teuchos::TimeMonitor localTimer(*powerIterationTime);

        // compute initial guess for lambda
        auto y = Teuchos::rcp(
            new Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>(
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
            new ShiftedOperator<Scalar, LocalOrdinal, GlobalOrdinal, Node>(
                H, network_->eigValLowerBound()));

        // create solver parameters
        auto params = Teuchos::rcp(new Teuchos::ParameterList());
        params->set("Maximum Iterations", maxCgIter);
        params->set("Convergence Tolerance", tolCgIter);

        for (int i = 1; i <= maxPowerIter; ++i) {
            // create linear problem
            auto z = Teuchos::rcp(
                new Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal,
                                        Node>(x->getMap(), 1));
            auto problem = Teuchos::rcp(
                new Belos::LinearProblem<
                    Scalar,
                    Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal,
                                        Node>,
                    Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal,
                                     Node>>(A, z, x.getConst()));
            problem->setLeftPrec(M);
            problem->setRightPrec(M);
            problem->setHermitian();
            problem->setProblem();

            // create CG solver
            Belos::PseudoBlockCGSolMgr<
                Scalar,
                Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>,
                Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
                solver(problem, params);

            // solve
            Belos::ReturnType status = solver.solve();

            TEUCHOS_TEST_FOR_EXCEPTION(
                status != Belos::ReturnType::Converged, std::runtime_error,
                "==Cnqs::FourierProblem::runInversePowerIteration== CG "
                "iteration "
                "did not converge");

            // copy x = solution
            x = problem->getLHS();

            // normalize x = x / norm2(x)
            std::vector<Scalar> xNorm(1);
            x->norm2(xNorm);

            std::vector<Scalar> scaleFactor(1);
            scaleFactor[0] = 1.0 / xNorm[0];
            x->scale(scaleFactor);

            // compute new estimate for lambda
            std::vector<Scalar> lambdaNew(1);

            H->apply(*x, *y);
            x->dot(*y, lambdaNew);
            const Scalar dLambda = std::abs(lambda[0] - lambdaNew[0]);
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

        if (comm_->getRank() == 0) {
            std::cout << "====================================================="
                         "================="
                      << std::endl;
        }
    }

    // save estimated state
    if (fileName.compare("") != 0) {
        Tpetra::MatrixMarket::Writer<
            Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>::
            writeDenseFile(fileName, *x, "low_eigen_state",
                           "Estimated lowest energy eigenstate of Hamiltonian");
    }

    // get time summary
    Teuchos::TimeMonitor::summarize();

    return lambda[0];
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
nlohmann::json
FourierProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>::description() const {
    nlohmann::json description;

    description["name"] = "fourier_problem";
    description["network"] = network_->description();
    description["max_freq"] = maxFreq_;

    return description;
}

}  // namespace Cnqs

#endif
