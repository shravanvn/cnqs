#include "Cnqs_BasicProblem.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <Teuchos_TimeMonitor.hpp>
#ifndef NDEBUG
#include <MatrixMarket_Tpetra.hpp>
#endif

#include "Utils.hpp"

Cnqs::BasicProblem::BasicProblem(
    const std::shared_ptr<const Cnqs::Network> &network, int numGridPoint,
    const Teuchos::RCP<const Teuchos::Comm<int>> &comm)
    : network_(network), numGridPoint_(numGridPoint),
      unfoldingFactors_(std::vector<int>(network->numRotor() + 1)),
      theta_(std::vector<double>(numGridPoint)), comm_(comm) {
    if (numGridPoint_ < 5) {
        throw std::domain_error(
            "==Cnqs::BasicProblem::BasicProblem== Need at least 5 grid points");
    }

    const int numRotor = network_->numRotor();
    unfoldingFactors_[0] = 1;
    for (int d = 0; d < numRotor; ++d) {
        unfoldingFactors_[d + 1] = numGridPoint_ * unfoldingFactors_[d];
    }

    double dTheta = 8.0 * std::atan(1.0) / numGridPoint_;
    for (int n = 0; n < numGridPoint_; ++n) {
        theta_[n] = n * dTheta;
    }

    // create timers
    Teuchos::RCP<Teuchos::Time> mapTime =
        Teuchos::TimeMonitor::getNewCounter("Map Construction Time");
    Teuchos::RCP<Teuchos::Time> initialStateTime =
        Teuchos::TimeMonitor::getNewCounter("Initial State Construction Time");
    Teuchos::RCP<Teuchos::Time> hamiltonianTime =
        Teuchos::TimeMonitor::getNewCounter("Hamiltonian Construction Time");

    // construct map
    auto map = constructMap(mapTime);

    // construct initial state
    auto initialState = constructInitialState(map, initialStateTime);
#ifndef NDEBUG
    Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<double, int, int>>::
        writeDenseFile("cnqs_basic_problem_initial_state.mm", *initialState,
                       "initial_state", "Initial state for CNQS basic problem");
#endif

    // construct Hamiltonian
    auto hamiltonian = constructHamiltonian(map, hamiltonianTime);
#ifndef NDEBUG
    Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<double, int, int>>::
        writeSparseFile("cnqs_basic_problem_hamiltonian.mm", *hamiltonian,
                        "operator",
                        "Hamiltonian operator for CNQS basic problem");
#endif

    // get time summary
    Teuchos::TimeMonitor::summarize();
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

Teuchos::RCP<const Tpetra::Vector<double, int, int>>
Cnqs::BasicProblem::constructInitialState(
    const Teuchos::RCP<const Tpetra::Map<int, int>> &map,
    const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor localTimer(*timer);

    const int numRotor = network_->numRotor();
    auto state = Teuchos::rcp(new Tpetra::Vector<double, int, int>(map));

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
            Tpetra::Vector<double, int, int>::device_type::memory_space;
        state->sync<memory_space>();
    }

    return state.getConst();
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
            if (k == 0) {
                continue;
            }

            for (int d = 0; d < numRotor; ++d) {
                // compute the global dimensional index of k-th neighbor along
                // d-th dimension
                std::vector<int> globalColumnIdDim(globalRowIdDim);
                globalColumnIdDim[d] += k;
                if (globalColumnIdDim[d] < 0) {
                    globalColumnIdDim[d] += numGridPoint_;
                } else if (globalColumnIdDim[d] >= numGridPoint_) {
                    globalColumnIdDim[d] -= numGridPoint_;
                }

                // convert the global dimensional index to global linear index
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
                } else {
                    throw std::logic_error(
                        "==Cnqs::BasicProblem::ConstructOperators== Index k "
                        "should not reach a value of 0");
                }

                ++currentRowNonZeroCount;
            }
        }

        if (currentRowNonZeroCount != numEntryPerRow) {
            throw std::logic_error(
                "==Cnqs::BasicProblem::ConstructOperators== Could not match "
                "number of non-zero entries in a row");
        }

        hamiltonian->insertGlobalValues(globalRowIdLin, currentRowColumnIndices,
                                        currentRowValues);
    }

    hamiltonian->fillComplete();
    return hamiltonian.getConst();
}

double Cnqs::BasicProblem::runInversePowerIteration(
    int numPowerIter, double tolPowerIter, int numCgIter, double tolCgIter,
    const std::string &fileName) const {
    return 0.0;
}

std::string Cnqs::BasicProblem::description() const {
    std::string description;
    description += "{\n";
    description +=
        "    \"network\": " + padString(network_->description()) + ",\n";
    description +=
        "    \"num_grid_point\": " + std::to_string(numGridPoint_) + "\n";
    description += "}";

    return description;
}
