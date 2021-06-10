#include "cnqs/pdesolver/spectral_difference_problem.hpp"

#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <tuple>

#include "shifted_operator.hpp"

cnqs::pdesolver::SpectralDifferenceProblem::SpectralDifferenceProblem(
    const std::shared_ptr<const Hamiltonian> &hamiltonian, long maxFreq,
    const Teuchos::RCP<const Teuchos::Comm<int>> &comm)
    : hamiltonian_(hamiltonian),
      maxFreq_(maxFreq),
      unfoldingFactors_(std::vector<long>(hamiltonian->numRotor() + 1)),
      comm_(comm) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        maxFreq_ < 1, std::domain_error,
        "SpectralDifferenceProblem== Need maximum frequency of at least one");

    const long numRotor = hamiltonian_->numRotor();
    unfoldingFactors_[0] = 1;
    for (long d = 0; d < numRotor; ++d) {
        unfoldingFactors_[d + 1] = (2 * maxFreq_ + 1) * unfoldingFactors_[d];
    }
}

Teuchos::RCP<const Tpetra::Map<int, long>>
cnqs::pdesolver::SpectralDifferenceProblem::constructMap(
    const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    const long numRotor = hamiltonian_->numRotor();

    const int rank = comm_->getRank();
    const int size = comm_->getSize();

    const long globalIndexBase = 0;
    const long globalNumElem = unfoldingFactors_[numRotor];

    long localNumElem = (globalNumElem + size - 1) / size;
    if (rank == size - 1) {
        localNumElem = globalNumElem - rank * localNumElem;
    }

    auto map = Teuchos::rcp(new Tpetra::Map<int, long>(
        globalNumElem, localNumElem, globalIndexBase, comm_));
    TEUCHOS_TEST_FOR_EXCEPTION(!map->isContiguous(), std::logic_error,
                               "Could not construct a contiguous map");

    return map.getConst();
}

Teuchos::RCP<Tpetra::MultiVector<double, int, long>>
cnqs::pdesolver::SpectralDifferenceProblem::constructInitialState(
    const Teuchos::RCP<const Tpetra::Map<int, long>> &map,
    const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    const long numRotor = hamiltonian_->numRotor();
    auto state =
        Teuchos::rcp(new Tpetra::MultiVector<double, int, long>(map, 1));
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    {
        const long numLocalRows = map->getNodeNumElements();

        state->sync_host();
        auto x_2d = state->getLocalViewHost();
        state->modify_host();

        for (long i = 0; i < numLocalRows; ++i) {
            x_2d(i, 0) = distribution(generator);
        }

        state->sync_device();
    }

    std::vector<double> stateNorm(1);
    state->norm2(stateNorm);

    std::vector<double> scaleFactor(1);
    scaleFactor[0] = 1.0 / stateNorm[0];
    state->scale(scaleFactor);

    return state;
}

Teuchos::RCP<const Tpetra::CrsMatrix<double, int, long>>
cnqs::pdesolver::SpectralDifferenceProblem::constructHamiltonian(
    const Teuchos::RCP<const Tpetra::Map<int, long>> &map,
    const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    // hamiltonian parameters
    const long numRotor = hamiltonian_->numRotor();
    const double vertexWeight = hamiltonian_->vertexWeight();
    const std::vector<std::tuple<long, long, double>> &edgeList =
        hamiltonian_->edgeList();
    const long numEdge = edgeList.size();
    const double sumWeights = hamiltonian_->sumEdgeWeights();

    // allocate memory for the Hamiltonian
    const long numEntryPerRow = 2 * numEdge + 1;
    auto hamiltonian = Teuchos::rcp(new Tpetra::CrsMatrix<double, int, long>(
        map, numEntryPerRow, Tpetra::StaticProfile));

    // assemble the Hamiltonian, one row at a time
    const long numLocalRows = map->getNodeNumElements();
    for (long localRowId = 0; localRowId < numLocalRows; ++localRowId) {
        const long globalRowIdLin = map->getGlobalElement(localRowId);

        // unwrap linear index i -> dimensional index (i_0, ..., i_{d - 1})
        std::vector<long> globalRowIdDim(numRotor);
        globalRowIdDim[0] = globalRowIdLin;
        for (long d = 0; d < numRotor; ++d) {
            if (d < numRotor - 1) {
                globalRowIdDim[d + 1] = globalRowIdDim[d] / (2 * maxFreq_ + 1);
            }
            globalRowIdDim[d] %= 2 * maxFreq_ + 1;
        }

        // collect column indices and values for current row
        std::vector<long> currentRowColumnIndices(numEntryPerRow);
        std::vector<double> currentRowValues(numEntryPerRow);

        currentRowColumnIndices[0] = globalRowIdLin;
        currentRowValues[0] = 0.0;
        for (long d = 0; d < numRotor; ++d) {
            currentRowValues[0] += std::pow(globalRowIdDim[d] - maxFreq_, 2.0);
        }
        currentRowValues[0] *= 0.5 * vertexWeight;
        currentRowValues[0] += 2.0 * sumWeights;

        long currentRowNonZeroCount = 1;

        for (long e = 0; e < numEdge; ++e) {
            const long j = std::get<0>(edgeList[e]);
            const long k = std::get<1>(edgeList[e]);
            const double beta = std::get<2>(edgeList[e]);

            for (long f = -1; f <= 1; ++f) {
                if (f != 0) {
                    std::vector<long> globalColumnIdDim(globalRowIdDim);
                    globalColumnIdDim[j] += (f == -1) ? 1 : -1;
                    globalColumnIdDim[k] -= (f == -1) ? 1 : -1;

                    if ((globalColumnIdDim[j] >= 0) &&
                        (globalColumnIdDim[j] <= 2 * maxFreq_) &&
                        (globalColumnIdDim[k] >= 0) &&
                        (globalColumnIdDim[k] <= 2 * maxFreq_)) {
                        long globalColumnIdLin = 0;
                        for (long d = 0; d < numRotor; ++d) {
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

Teuchos::RCP<const Tpetra::CrsMatrix<double, int, long>>
cnqs::pdesolver::SpectralDifferenceProblem::constructPreconditioner(
    const Teuchos::RCP<const Tpetra::Map<int, long>> &map,
    const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    // hamiltonian parameters
    const long numRotor = hamiltonian_->numRotor();
    const double vertexWeight = hamiltonian_->vertexWeight();
    const double sumWeights = hamiltonian_->sumEdgeWeights();
    const double mu = -4 * hamiltonian_->sumAbsEdgeWeights();

    // allocate memory for the preconditioner
    auto preconditioner = Teuchos::rcp(new Tpetra::CrsMatrix<double, int, long>(
        map, 1, Tpetra::StaticProfile));

    // assemble the Hamiltonian, one row at a time
    const long numLocalRows = map->getNodeNumElements();
    for (long localRowId = 0; localRowId < numLocalRows; ++localRowId) {
        const long globalRowIdLin = map->getGlobalElement(localRowId);

        // unwrap linear index i -> dimensional index (i_0, ..., i_{d - 1})
        std::vector<long> globalRowIdDim(numRotor);
        globalRowIdDim[0] = globalRowIdLin;
        for (long d = 0; d < numRotor; ++d) {
            if (d < numRotor - 1) {
                globalRowIdDim[d + 1] = globalRowIdDim[d] / (2 * maxFreq_ + 1);
            }
            globalRowIdDim[d] %= 2 * maxFreq_ + 1;
        }

        // collect column indices and values for current row
        std::vector<long> currentRowColumnIndices(1);
        std::vector<double> currentRowValues(1);

        currentRowColumnIndices[0] = globalRowIdLin;
        currentRowValues[0] = 0.0;
        for (long d = 0; d < numRotor; ++d) {
            currentRowValues[0] += std::pow(globalRowIdDim[d] - maxFreq_, 2.0);
        }
        currentRowValues[0] =
            1 / std::sqrt(vertexWeight * currentRowValues[0] / 2 +
                          2 * sumWeights - mu);

        preconditioner->insertGlobalValues(
            globalRowIdLin, currentRowColumnIndices, currentRowValues);
    }

    preconditioner->fillComplete();
    return preconditioner.getConst();
}

double cnqs::pdesolver::SpectralDifferenceProblem::runInversePowerIteration(
    long maxPowerIter, double tolPowerIter, long maxCgIter, double tolCgIter,
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
    auto mu = -4 * hamiltonian_->sumAbsEdgeWeights();

    std::vector<double> lambda(1);
    {
        // create local timer
        Teuchos::TimeMonitor localTimer(*powerIterationTime);

        // compute initial guess for lambda
        auto y = Teuchos::rcp(
            new Tpetra::MultiVector<double, int, long>(x->getMap(), 1));
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
        auto A = Teuchos::rcp(new ShiftedOperator(H, mu));

        // create solver parameters
        auto params = Teuchos::rcp(new Teuchos::ParameterList());
        params->set("Maximum Iterations", static_cast<int>(maxCgIter));
        params->set("Convergence Tolerance", tolCgIter);

        for (long i = 1; i <= maxPowerIter; ++i) {
            // create linear problem
            auto z = Teuchos::rcp(
                new Tpetra::MultiVector<double, int, long>(x->getMap(), 1));
            auto problem = Teuchos::rcp(
                new Belos::LinearProblem<double,
                                         Tpetra::MultiVector<double, int, long>,
                                         Tpetra::Operator<double, int, long>>(
                    A, z, x.getConst()));
            problem->setLeftPrec(M);
            problem->setRightPrec(M);
            problem->setHermitian();
            problem->setProblem();

            // create CG solver
            Belos::PseudoBlockCGSolMgr<double,
                                       Tpetra::MultiVector<double, int, long>,
                                       Tpetra::Operator<double, int, long>>
                solver(problem, params);

            // solve
            Belos::ReturnType status = solver.solve();

            TEUCHOS_TEST_FOR_EXCEPTION(
                status != Belos::ReturnType::Converged, std::runtime_error,
                "runInversePowerIteration== CG iteration did not converge");

            // copy x = solution
            x = problem->getLHS();

            // normalize x = x / norm2(x)
            std::vector<double> xNorm(1);
            x->norm2(xNorm);

            std::vector<double> scaleFactor(1);
            scaleFactor[0] = 1.0 / xNorm[0];
            x->scale(scaleFactor);

            // compute new estimate for lambda
            std::vector<double> lambdaNew(1);

            H->apply(*x, *y);
            x->dot(*y, lambdaNew);
            const double dLambda = std::abs(lambda[0] - lambdaNew[0]);
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
        Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<double, int, long>>::
            writeDenseFile(fileName, *x, "low_eigen_state",
                           "Estimated lowest energy eigenstate of Hamiltonian");
    }

    // get time summary
    Teuchos::TimeMonitor::summarize();

    return lambda[0];
}
