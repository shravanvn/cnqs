#include "Cnqs_BasicProblem.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include "Cnqs_ShiftedOperator.hpp"

Cnqs::BasicProblem::BasicProblem(
    const std::shared_ptr<const Cnqs::Network> &network, int numGridPoint,
    const Teuchos::RCP<const Teuchos::Comm<int>> &comm)
    : network_(network), numGridPoint_(numGridPoint),
      unfoldingFactors_(std::vector<int>(network->numRotor() + 1)),
      theta_(std::vector<double>(numGridPoint)), comm_(comm) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        numGridPoint_ < 5, std::domain_error,
        "==Cnqs::BasicProblem::BasicProblem== Need at least 5 grid points");

    const int numRotor = network_->numRotor();
    unfoldingFactors_[0] = 1;
    for (int d = 0; d < numRotor; ++d) {
        unfoldingFactors_[d + 1] = numGridPoint_ * unfoldingFactors_[d];
    }

    double dTheta = 8.0 * std::atan(1.0) / numGridPoint_;
    for (int n = 0; n < numGridPoint_; ++n) {
        theta_[n] = n * dTheta;
    }
}

Teuchos::RCP<const Tpetra::Map<int, int>> Cnqs::BasicProblem::constructMap(
    const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    const int numRotor = network_->numRotor();

    const int rank = comm_->getRank();
    const int size = comm_->getSize();

    const int globalIndexBase = 0;
    const int globalNumElem = unfoldingFactors_[numRotor];

    int localNumElem = (globalNumElem + size - 1) / size;
    if (rank == size - 1) {
        localNumElem = globalNumElem - rank * localNumElem;
    }

    auto map = Teuchos::rcp(new Tpetra::Map<int, int>(
        globalNumElem, localNumElem, globalIndexBase, comm_));
    TEUCHOS_TEST_FOR_EXCEPTION(!map->isContiguous(), std::logic_error,
                               "==Cnqs::BasicProblem::ConstructMap== Could not "
                               "construct a contiguous map");

    return map.getConst();
}

Teuchos::RCP<Tpetra::MultiVector<double, int, int>>
Cnqs::BasicProblem::constructInitialState(
    const Teuchos::RCP<const Tpetra::Map<int, int>> &map,
    const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    const int numRotor = network_->numRotor();
    auto state =
        Teuchos::rcp(new Tpetra::MultiVector<double, int, int>(map, 1));

    {
        const int numLocalRows = map->getNodeNumElements();

        state->sync_host();
        auto x_2d = state->getLocalViewHost();
        state->modify_host();

        for (int i = 0; i < numLocalRows; ++i) {
            int linearIndex = map->getGlobalElement(i);

            double entry = 1.0;
            for (int d = 0; d < numRotor; ++d) {
                const int i_d = linearIndex % numGridPoint_;
                entry *= std::cos(theta_[i_d]);
                linearIndex /= numGridPoint_;
            }

            x_2d(i, 0) = entry;
        }

        using memory_space =
            Tpetra::MultiVector<double, int, int>::device_type::memory_space;
        state->sync<memory_space>();
    }

    std::vector<double> stateNorm(1);
    state->norm2(stateNorm);

    std::vector<double> scaleFactor(1);
    scaleFactor[0] = 1.0 / stateNorm[0];
    state->scale(scaleFactor);

    return state;
}

Teuchos::RCP<const Tpetra::CrsMatrix<double, int, int>>
Cnqs::BasicProblem::constructHamiltonian(
    const Teuchos::RCP<const Tpetra::Map<int, int>> &map,
    const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    // network parameters
    const int numRotor = network_->numRotor();
    const std::vector<std::tuple<int, int, double>> &edgeList =
        network_->edgeList();

    // finite difference discretization parameters
    const double h = theta_[1] - theta_[0];
    const double fact = -0.5 / (12.0 * h * h);

    // allocate memory for the Hamiltonian
    const int numEntryPerRow = 4 * numRotor + 1;
    auto hamiltonian = Teuchos::rcp(new Tpetra::CrsMatrix<double, int, int>(
        map, numEntryPerRow, Tpetra::StaticProfile));

    // assemble the Hamiltonian, one row at a time
    const int numLocalRows = map->getNodeNumElements();
    for (int localRowId = 0; localRowId < numLocalRows; ++localRowId) {
        const int globalRowIdLin = map->getGlobalElement(localRowId);

        // unwrap linear index i -> dimensional index (i_0, ..., i_{d - 1})
        std::vector<int> globalRowIdDim(numRotor);
        globalRowIdDim[0] = globalRowIdLin;
        for (int d = 0; d < numRotor; ++d) {
            if (d < numRotor - 1) {
                globalRowIdDim[d + 1] = globalRowIdDim[d] / numGridPoint_;
            }
            globalRowIdDim[d] %= numGridPoint_;
        }

        // collect column indices and values for current row
        std::vector<int> currentRowColumnIndices(numEntryPerRow);
        std::vector<double> currentRowValues(numEntryPerRow);

        currentRowColumnIndices[0] = globalRowIdLin;
        currentRowValues[0] = -30.0 * fact * numRotor;
        for (const auto &edge : edgeList) {
            const int j = std::get<0>(edge);
            const int k = std::get<1>(edge);
            const double g = std::get<2>(edge);

            currentRowValues[0] += -g * std::cos(theta_[globalRowIdDim[j]] -
                                                 theta_[globalRowIdDim[k]]);
        }

        int currentRowNonZeroCount = 1;

        for (int k = -2; k <= 2; ++k) {
            if (k != 0) {
                for (int d = 0; d < numRotor; ++d) {
                    // compute the global dimensional index of k-th neighbor
                    // along d-th dimension
                    std::vector<int> globalColumnIdDim(globalRowIdDim);
                    globalColumnIdDim[d] += k;
                    if (globalColumnIdDim[d] < 0) {
                        globalColumnIdDim[d] += numGridPoint_;
                    } else if (globalColumnIdDim[d] >= numGridPoint_) {
                        globalColumnIdDim[d] -= numGridPoint_;
                    }

                    // convert the global dimensional index to global linear
                    // index
                    int globalColumnIdLin = 0;
                    for (int d = 0; d < numRotor; ++d) {
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
            "==Cnqs::BasicProblem::ConstructOperators== Could not match number "
            "of non-zero entries in a row");

        hamiltonian->insertGlobalValues(globalRowIdLin, currentRowColumnIndices,
                                        currentRowValues);
    }

    hamiltonian->fillComplete();
    return hamiltonian.getConst();
}

double Cnqs::BasicProblem::runInversePowerIteration(
    int maxPowerIter, double tolPowerIter, int maxCgIter, double tolCgIter,
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

    std::vector<double> lambda(1);
    {
        // create local timer
        Teuchos::TimeMonitor localTimer(*powerIterationTime);

        // compute initial guess for lambda
        auto y = Teuchos::rcp(
            new Tpetra::MultiVector<double, int, int>(x->getMap(), 1));
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
            new Cnqs::ShiftedOperator(H, network_->eigValLowerBound()));

        // create solver parameters
        auto params = Teuchos::rcp(new Teuchos::ParameterList());
        params->set("Maximum Iterations", maxCgIter);
        params->set("Convergence Tolerance", tolCgIter);

        for (int i = 1; i <= maxPowerIter; ++i) {
            // create linear problem
            auto z = Teuchos::rcp(
                new Tpetra::MultiVector<double, int, int>(x->getMap(), 1));
            auto problem = Teuchos::rcp(
                new Belos::LinearProblem<double,
                                         Tpetra::MultiVector<double, int, int>,
                                         Tpetra::Operator<double, int, int>>(
                    A, z, x.getConst()));
            problem->setHermitian();
            problem->setProblem();

            // create CG solver
            Belos::PseudoBlockCGSolMgr<double,
                                       Tpetra::MultiVector<double, int, int>,
                                       Tpetra::Operator<double, int, int>>
                solver(problem, params);

            // solve
            Belos::ReturnType status = solver.solve();

            TEUCHOS_TEST_FOR_EXCEPTION(
                status != Belos::ReturnType::Converged, std::runtime_error,
                "==Cnqs::BasicProblem::runInversePowerIteration== CG iteration "
                "did not converge");

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
        Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<double, int, int>>::
            writeDenseFile(fileName, *x, "low_eigen_state",
                           "Estimated lowest energy eigenstate of Hamiltonian");
    }

    // get time summary
    Teuchos::TimeMonitor::summarize();

    return lambda[0];
}

nlohmann::json Cnqs::BasicProblem::description() const {
    nlohmann::json description;

    description["name"] = "basic_problem";
    description["network"] = network_->description();
    description["num_grid_point"] = numGridPoint_;

    return description;
}
