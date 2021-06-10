#include "cnqs/pdesolver/finite_difference_problem.hpp"

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

cnqs::pdesolver::FiniteDifferenceProblem::FiniteDifferenceProblem(
    const std::shared_ptr<const Hamiltonian> &hamiltonian, long numGridPoint,
    const Teuchos::RCP<const Teuchos::Comm<int>> &comm)
    : hamiltonian_(hamiltonian),
      numGridPoint_(numGridPoint),
      unfoldingFactors_(std::vector<long>(hamiltonian->numRotor() + 1)),
      theta_(std::vector<double>(numGridPoint)),
      comm_(comm) {
    TEUCHOS_TEST_FOR_EXCEPTION(numGridPoint_ < 5, std::domain_error,
                               "Need at least 5 grid points");

    const long numRotor = hamiltonian_->numRotor();
    unfoldingFactors_[0] = 1;
    for (long d = 0; d < numRotor; ++d) {
        unfoldingFactors_[d + 1] = numGridPoint_ * unfoldingFactors_[d];
    }

    double dTheta = 8.0 * std::atan(1.0) / numGridPoint_;
    for (long n = 0; n < numGridPoint_; ++n) {
        theta_[n] = n * dTheta;
    }
}

Teuchos::RCP<const Tpetra::Map<int, long>>
cnqs::pdesolver::FiniteDifferenceProblem::constructMap(
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
cnqs::pdesolver::FiniteDifferenceProblem::constructInitialState(
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
cnqs::pdesolver::FiniteDifferenceProblem::constructHamiltonian(
    const Teuchos::RCP<const Tpetra::Map<int, long>> &map,
    const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    // hamiltonian parameters
    const long numRotor = hamiltonian_->numRotor();
    const double vertexWeight = hamiltonian_->vertexWeight();
    const std::vector<std::tuple<long, long, double>> &edgeList =
        hamiltonian_->edgeList();

    // finite difference discretization parameters
    const double h = theta_[1] - theta_[0];
    const double fact = -0.5 * vertexWeight / (12.0 * h * h);

    // allocate memory for the Hamiltonian
    const long numEntryPerRow = 4 * numRotor + 1;
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
                globalRowIdDim[d + 1] = globalRowIdDim[d] / numGridPoint_;
            }
            globalRowIdDim[d] %= numGridPoint_;
        }

        // collect column indices and values for current row
        std::vector<long> currentRowColumnIndices(numEntryPerRow);
        std::vector<double> currentRowValues(numEntryPerRow);

        currentRowColumnIndices[0] = globalRowIdLin;
        currentRowValues[0] = -30.0 * fact * numRotor;
        for (const auto &edge : edgeList) {
            const long j = std::get<0>(edge);
            const long k = std::get<1>(edge);
            const double beta = std::get<2>(edge);

            currentRowValues[0] +=
                beta * (2.0 - 2.0 * std::cos(theta_[globalRowIdDim[j]] -
                                             theta_[globalRowIdDim[k]]));
        }

        long currentRowNonZeroCount = 1;

        for (long k = -2; k <= 2; ++k) {
            if (k != 0) {
                for (long d = 0; d < numRotor; ++d) {
                    // compute the global dimensional index of k-th neighbor
                    // along d-th dimension
                    std::vector<long> globalColumnIdDim(globalRowIdDim);
                    globalColumnIdDim[d] += k;
                    if (globalColumnIdDim[d] < 0) {
                        globalColumnIdDim[d] += numGridPoint_;
                    } else if (globalColumnIdDim[d] >= numGridPoint_) {
                        globalColumnIdDim[d] -= numGridPoint_;
                    }

                    // convert the global dimensional index to global linear
                    // index
                    long globalColumnIdLin = 0;
                    for (long d = 0; d < numRotor; ++d) {
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
            "Could not match number of non-zero entries in a row");

        hamiltonian->insertGlobalValues(globalRowIdLin, currentRowColumnIndices,
                                        currentRowValues);
    }

    hamiltonian->fillComplete();
    return hamiltonian.getConst();
}

double cnqs::pdesolver::FiniteDifferenceProblem::runInversePowerIteration(
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
    const double mu = -4 * hamiltonian_->sumAbsEdgeWeights();

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
            problem->setHermitian();
            problem->setProblem();

            // create CG solver
            Belos::PseudoBlockCGSolMgr<double,
                                       Tpetra::MultiVector<double, int, long>,
                                       Tpetra::Operator<double, int, long>>
                solver(problem, params);

            // solve
            Belos::ReturnType status = solver.solve();

            TEUCHOS_TEST_FOR_EXCEPTION(status != Belos::ReturnType::Converged,
                                       std::runtime_error,
                                       "CG iteration did not converge");

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

        const double h = theta_[1] - theta_[0];
        x->scale(1.0 / (h * h));

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
