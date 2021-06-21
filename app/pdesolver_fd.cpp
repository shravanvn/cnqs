#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "cnqs/pdesolver/finite_difference_problem.hpp"
#include "cnqs/pdesolver/hamiltonian.hpp"

int main(int argc, char **argv) {
    int exit_code = 1;

    Tpetra::ScopeGuard tpetra_scope(&argc, &argv);
    {
        std::string hamiltonian_file_name = "hamiltonian.yaml";
        long num_grid_point = 128;
        long num_power_iter = 10000;
        double tol_power_iter = 1.0e-15;
        long num_cg_iter = 10000;
        double tol_cg_iter = 1.0e-15;
        std::string ground_state_file_name = "";

        auto cmd_parser = Teuchos::CommandLineProcessor(false, true, true);
        cmd_parser.setOption(
            "hamiltonian-file-name", &hamiltonian_file_name,
            "File containing YAML description of rotor hamiltonian", true);
        cmd_parser.setOption(
            "num-grid-point", &num_grid_point,
            "Number of grid points per dimension for domain discretization",
            true);
        cmd_parser.setOption(
            "num-power-iter", &num_power_iter,
            "Maximum number of steps in inverse power iteration");
        cmd_parser.setOption("tol-power-iter", &tol_power_iter,
                             "Tolerance for inverse power iteration");
        cmd_parser.setOption(
            "num-cg-iter", &num_cg_iter,
            "Maximum number of CG iterations per inverse power iteration step");
        cmd_parser.setOption("tol-cg-iter", &tol_cg_iter,
                             "Tolerance of CG iteration");
        cmd_parser.setOption(
            "ground-state-file-name", &ground_state_file_name,
            "Name of the file where ground state will be saved");

        const auto status = cmd_parser.parse(argc, argv);

        if (status == Teuchos::CommandLineProcessor::EParseCommandLineReturn::
                          PARSE_SUCCESSFUL) {
            const auto comm = Tpetra::getDefaultComm();
            const auto hamiltonian =
                std::make_shared<cnqs::pdesolver::Hamiltonian>(
                    hamiltonian_file_name);
            cnqs::pdesolver::FiniteDifferenceProblem problem(
                hamiltonian, num_grid_point, comm);
            problem.RunInversePowerIteration(num_power_iter, tol_power_iter,
                                             num_cg_iter, tol_cg_iter,
                                             ground_state_file_name);

            exit_code = 0;
        } else if (status == Teuchos::CommandLineProcessor::
                                 EParseCommandLineReturn::PARSE_HELP_PRINTED) {
            exit_code = 0;
        } else {
            std::cerr << "ERROR: Could not parse command line options"
                      << std::endl;
            std::cerr << "Run " << argv[0]
                      << " --help to get an overview of avaialble options"
                      << std::endl;
        }
    }

    return exit_code;
}
