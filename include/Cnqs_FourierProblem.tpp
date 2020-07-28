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

    LocalOrdinal localNumElem = (globalNumElem + size - 1) / size;
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
    const Scalar value = std::pow(4.0 * std::atan(1.0), numRotor);

    {
        const LocalOrdinal numLocalRows = map->getNodeNumElements();

        state->sync_host();
        auto x_2d = state->getLocalViewHost();
        state->modify_host();

        for (LocalOrdinal i = 0; i < numLocalRows; ++i) {
            GlobalOrdinal linearIndex = map->getGlobalElement(i);

            bool flag = true;
            for (GlobalOrdinal d = 0; d < numRotor; ++d) {
                const GlobalOrdinal i_d =
                    linearIndex % (2 * maxFreq_ + 1) - maxFreq_;
                if (std::abs(i_d) != 1) {
                    flag = false;
                    break;
                }
                linearIndex /= (2 * maxFreq_ + 1);
            }

            x_2d(i, 0) = flag ? value : 0.0;
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

    // allocate memory for the Hamiltonian
    const GlobalOrdinal numEntryPerRow = 2 * numEdge + 1;
    auto hamiltonian = Teuchos::rcp(
        new Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(
            map, numEntryPerRow, Tpetra::StaticProfile));

    // assemble the Hamiltonian, one row at a time
    const LocalOrdinal numLocalRows = map->getNodeNumElements();
    for (LocalOrdinal localRowId = 0; localRowId < numLocalRows; ++localRowId) {
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

        GlobalOrdinal currentRowNonZeroCount = 1;

        for (GlobalOrdinal e = 0; e < numEdge; ++e) {
            const GlobalOrdinal j = std::get<0>(edgeList[e]);
            const GlobalOrdinal k = std::get<1>(edgeList[e]);
            const Scalar g = std::get<2>(edgeList[e]);

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
                        currentRowValues[currentRowNonZeroCount] = -0.5 * g;

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

    // allocate memory for the preconditioner
    auto preconditioner = Teuchos::rcp(
        new Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(
            map, 1, Tpetra::StaticProfile));

    // assemble the Hamiltonian, one row at a time
    const LocalOrdinal numLocalRows = map->getNodeNumElements();
    for (LocalOrdinal localRowId = 0; localRowId < numLocalRows; ++localRowId) {
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
        currentRowValues[0] =
            std::sqrt(1.0 / (0.5 * currentRowValues[0] - eigValLowerBound));

        preconditioner->insertGlobalValues(
            globalRowIdLin, currentRowColumnIndices, currentRowValues);
    }

    preconditioner->fillComplete();
    return preconditioner.getConst();
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Scalar FourierProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
    runInversePowerIteration(GlobalOrdinal maxPowerIter, Scalar tolPowerIter,
                             GlobalOrdinal maxCgIter, Scalar tolCgIter,
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

        for (GlobalOrdinal i = 1; i <= maxPowerIter; ++i) {
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
