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
    const std::shared_ptr<const Hamiltonian> &hamiltonian, long num_grid_point,
    const Teuchos::RCP<const Teuchos::Comm<int>> &comm)
    : hamiltonian_(hamiltonian),
      num_grid_point_(num_grid_point),
      unfolding_factors_(std::vector<long>(hamiltonian->NumRotor() + 1)),
      theta_(std::vector<double>(num_grid_point)),
      comm_(comm) {
    TEUCHOS_TEST_FOR_EXCEPTION(num_grid_point_ < 5, std::domain_error,
                               "Need at least 5 grid points");

    const long num_rotor = hamiltonian_->NumRotor();
    unfolding_factors_[0] = 1;
    for (long d = 0; d < num_rotor; ++d) {
        unfolding_factors_[d + 1] = num_grid_point_ * unfolding_factors_[d];
    }

    double d_theta = 8.0 * std::atan(1.0) / num_grid_point_;
    for (long n = 0; n < num_grid_point_; ++n) {
        theta_[n] = n * d_theta;
    }
}

Teuchos::RCP<const Tpetra::Map<int, long>>
cnqs::pdesolver::FiniteDifferenceProblem::ConstructMap(
    const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor local_timer(*timer);

    const long num_rotor = hamiltonian_->NumRotor();

    const int rank = comm_->getRank();
    const int size = comm_->getSize();

    const long global_index_base = 0;
    const long global_num_element = unfolding_factors_[num_rotor];

    long local_num_element = (global_num_element + size - 1) / size;
    if (rank == size - 1) {
        local_num_element = global_num_element - rank * local_num_element;
    }

    auto map = Teuchos::rcp(new Tpetra::Map<int, long>(
        global_num_element, local_num_element, global_index_base, comm_));
    TEUCHOS_TEST_FOR_EXCEPTION(!map->isContiguous(), std::logic_error,
                               "Could not construct a contiguous map");

    return map.getConst();
}

Teuchos::RCP<Tpetra::MultiVector<double, int, long>>
cnqs::pdesolver::FiniteDifferenceProblem::ConstructInitialState(
    const Teuchos::RCP<const Tpetra::Map<int, long>> &map,
    const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor local_timer(*timer);

    const long num_rotor = hamiltonian_->NumRotor();
    auto state =
        Teuchos::rcp(new Tpetra::MultiVector<double, int, long>(map, 1));

    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    {
        const long num_local_rows = map->getNodeNumElements();

        state->sync_host();
        auto x_2d = state->getLocalViewHost();
        state->modify_host();

        for (long i = 0; i < num_local_rows; ++i) {
            x_2d(i, 0) = distribution(generator);
        }

        state->sync_device();
    }

    std::vector<double> state_norm(1);
    state->norm2(state_norm);

    std::vector<double> scale_factor(1);
    scale_factor[0] = 1.0 / state_norm[0];
    state->scale(scale_factor);

    return state;
}

Teuchos::RCP<const Tpetra::CrsMatrix<double, int, long>>
cnqs::pdesolver::FiniteDifferenceProblem::ConstructHamiltonian(
    const Teuchos::RCP<const Tpetra::Map<int, long>> &map,
    const Teuchos::RCP<Teuchos::Time> &timer) const {
    // create local timer
    Teuchos::TimeMonitor local_timer(*timer);

    // hamiltonian parameters
    const long num_rotor = hamiltonian_->NumRotor();
    const double vertex_weight = hamiltonian_->VertexWeight();
    const std::vector<std::tuple<long, long, double>> &edge_list =
        hamiltonian_->EdgeList();

    // finite difference discretization parameters
    const double d_theta = theta_[1] - theta_[0];
    const double fact = -0.5 * vertex_weight / (12.0 * d_theta * d_theta);

    // allocate memory for the Hamiltonian
    const long num_entry_per_row = 4 * num_rotor + 1;
    auto hamiltonian = Teuchos::rcp(new Tpetra::CrsMatrix<double, int, long>(
        map, num_entry_per_row, Tpetra::StaticProfile));

    // assemble the Hamiltonian, one row at a time
    const long num_local_rows = map->getNodeNumElements();
    for (long local_row_id = 0; local_row_id < num_local_rows; ++local_row_id) {
        const long global_row_id_linear = map->getGlobalElement(local_row_id);

        // unwrap linear index i -> dimensional index (i_0, ..., i_{d - 1})
        std::vector<long> global_row_id_cartesian(num_rotor);
        global_row_id_cartesian[0] = global_row_id_linear;
        for (long d = 0; d < num_rotor; ++d) {
            if (d < num_rotor - 1) {
                global_row_id_cartesian[d + 1] =
                    global_row_id_cartesian[d] / num_grid_point_;
            }
            global_row_id_cartesian[d] %= num_grid_point_;
        }

        // collect column indices and values for current row
        std::vector<long> current_row_column_indices(num_entry_per_row);
        std::vector<double> current_row_values(num_entry_per_row);

        current_row_column_indices[0] = global_row_id_linear;
        current_row_values[0] = -30.0 * fact * num_rotor;
        for (const auto &edge : edge_list) {
            const long j = std::get<0>(edge);
            const long k = std::get<1>(edge);
            const double beta = std::get<2>(edge);

            current_row_values[0] +=
                beta *
                (2.0 - 2.0 * std::cos(theta_[global_row_id_cartesian[j]] -
                                      theta_[global_row_id_cartesian[k]]));
        }

        long current_row_non_zero_count = 1;

        for (long k = -2; k <= 2; ++k) {
            if (k != 0) {
                for (long d = 0; d < num_rotor; ++d) {
                    // compute the global dimensional index of k-th neighbor
                    // along d-th dimension
                    std::vector<long> global_column_id_cartesian(
                        global_row_id_cartesian);
                    global_column_id_cartesian[d] += k;
                    if (global_column_id_cartesian[d] < 0) {
                        global_column_id_cartesian[d] += num_grid_point_;
                    } else if (global_column_id_cartesian[d] >=
                               num_grid_point_) {
                        global_column_id_cartesian[d] -= num_grid_point_;
                    }

                    // convert the global dimensional index to global linear
                    // index
                    long global_column_id_linear = 0;
                    for (long d = 0; d < num_rotor; ++d) {
                        global_column_id_linear +=
                            unfolding_factors_[d] *
                            global_column_id_cartesian[d];
                    }
                    current_row_column_indices[current_row_non_zero_count] =
                        global_column_id_linear;

                    // compute the correct nonzero value of the matrix entry
                    if (std::abs(k) == 1) {
                        current_row_values[current_row_non_zero_count] =
                            16.0 * fact;
                    } else if (std::abs(k) == 2) {
                        current_row_values[current_row_non_zero_count] = -fact;
                    }

                    ++current_row_non_zero_count;
                }
            }
        }

        TEUCHOS_TEST_FOR_EXCEPTION(
            current_row_non_zero_count != num_entry_per_row, std::logic_error,
            "Could not match number of non-zero entries in a row");

        hamiltonian->insertGlobalValues(global_row_id_linear,
                                        current_row_column_indices,
                                        current_row_values);
    }

    hamiltonian->fillComplete();
    return hamiltonian.getConst();
}

double cnqs::pdesolver::FiniteDifferenceProblem::RunInversePowerIteration(
    long num_power_iter, double tol_power_iter, long num_cg_iter,
    double tol_cg_iter, const std::string &file_name) const {
    // create timers
    Teuchos::RCP<Teuchos::Time> map_time =
        Teuchos::TimeMonitor::getNewCounter("CNQS: Map Construction Time");
    Teuchos::RCP<Teuchos::Time> initial_state_time =
        Teuchos::TimeMonitor::getNewCounter(
            "CNQS: Initial State Construction Time");
    Teuchos::RCP<Teuchos::Time> hamiltonian_time =
        Teuchos::TimeMonitor::getNewCounter(
            "CNQS: Hamiltonian Construction Time");
    Teuchos::RCP<Teuchos::Time> power_iteration_time =
        Teuchos::TimeMonitor::getNewCounter(
            "CNQS: Inverse Power Iteration Time");

    // construct map
    auto map = ConstructMap(map_time);

    // construct initial state
    auto x = ConstructInitialState(map, initial_state_time);

    // construct Hamiltonian
    auto H = ConstructHamiltonian(map, hamiltonian_time);

    // get lower bound for eigenvalues
    const double mu = -4 * hamiltonian_->SumAbsEdgeWeights();

    std::vector<double> lambda(1);
    {
        // create local timer
        Teuchos::TimeMonitor local_timer(*power_iteration_time);

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
        params->set("Maximum Iterations", static_cast<int>(num_cg_iter));
        params->set("Convergence Tolerance", tol_cg_iter);

        for (long i = 1; i <= num_power_iter; ++i) {
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
            std::vector<double> x_norm(1);
            x->norm2(x_norm);

            std::vector<double> scale_factor(1);
            scale_factor[0] = 1.0 / x_norm[0];
            x->scale(scale_factor);

            // compute new estimate for lambda
            std::vector<double> lambda_new(1);

            H->apply(*x, *y);
            x->dot(*y, lambda_new);
            const double d_lambda = std::abs(lambda[0] - lambda_new[0]);
            lambda[0] = lambda_new[0];

            if (comm_->getRank() == 0) {
                std::cout << std::setw(9) << i << " " << std::setw(24)
                          << std::setprecision(16) << lambda[0] << " "
                          << std::setw(12) << std::setprecision(6) << d_lambda
                          << " " << std::setw(9) << solver.getNumIters() << " "
                          << std::setw(12) << solver.achievedTol() << std::endl;
            }

            // check for convergence
            if (d_lambda < tol_power_iter) {
                break;
            }
        }

        const double d_theta = theta_[1] - theta_[0];
        x->scale(1.0 / (d_theta * d_theta));

        if (comm_->getRank() == 0) {
            std::cout << "====================================================="
                         "================="
                      << std::endl;
        }
    }

    // save estimated state
    if (file_name.compare("") != 0) {
        Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<double, int, long>>::
            writeDenseFile(file_name, *x, "low_eigen_state",
                           "Estimated lowest energy eigenstate of Hamiltonian");
    }

    // get time summary
    Teuchos::TimeMonitor::summarize();

    return lambda[0];
}
