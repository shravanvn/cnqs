#include <Teuchos_Comm.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_MultiVector.hpp>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "cnqs/pdesolver/finite_difference_problem.hpp"
#include "cnqs/pdesolver/hamiltonian.hpp"

int main(int argc, char **argv) {
    using Scalar = Tpetra::MultiVector<>::scalar_type;
    using LocalOrdinal = Tpetra::MultiVector<>::local_ordinal_type;
    using GlobalOrdinal = Tpetra::MultiVector<>::global_ordinal_type;
    using Node = Tpetra::MultiVector<>::node_type;

    int exitCode = 1;

    Tpetra::ScopeGuard tpetraScope(&argc, &argv);
    {
        std::string hamiltonianFileName = "hamiltonian.yaml";
        GlobalOrdinal numGridPoint = 128;
        GlobalOrdinal maxPowerIter = 10000;
        Scalar tolPowerIter = 1.0e-15;
        GlobalOrdinal maxCgIter = 10000;
        Scalar tolCgIter = 1.0e-15;
        std::string groundStateFileName = "";

        auto cmdParser = Teuchos::CommandLineProcessor(false, true, true);
        cmdParser.setOption(
            "hamiltonian-file-name", &hamiltonianFileName,
            "File containing YAML description of rotor hamiltonian", true);
        cmdParser.setOption(
            "num-grid-point", &numGridPoint,
            "Number of grid points per dimension for domain discretization",
            true);
        cmdParser.setOption(
            "max-power-iter", &maxPowerIter,
            "Maximum number of steps in inverse power iteration");
        cmdParser.setOption("tol-power-iter", &tolPowerIter,
                            "Tolerance for inverse power iteration");
        cmdParser.setOption(
            "max-cg-iter", &maxCgIter,
            "Maximum number of CG iterations per inverse power iteration step");
        cmdParser.setOption("tol-cg-iter", &tolCgIter,
                            "Tolerance of CG iteration");
        cmdParser.setOption(
            "ground-state-file-name", &groundStateFileName,
            "Name of the file where ground state will be saved");

        const auto status = cmdParser.parse(argc, argv);

        if (status == Teuchos::CommandLineProcessor::EParseCommandLineReturn::
                          PARSE_SUCCESSFUL) {
            const auto comm = Tpetra::getDefaultComm();
            const auto hamiltonian = std::make_shared<
                cnqs::pdesolver::Hamiltonian<Scalar, GlobalOrdinal>>(
                hamiltonianFileName);
            cnqs::pdesolver::FiniteDifferenceProblem<Scalar, LocalOrdinal,
                                                     GlobalOrdinal, Node>
                problem(hamiltonian, numGridPoint, comm);
            problem.runInversePowerIteration(maxPowerIter, tolPowerIter,
                                             maxCgIter, tolCgIter,
                                             groundStateFileName);

            exitCode = 0;
        } else if (status == Teuchos::CommandLineProcessor::
                                 EParseCommandLineReturn::PARSE_HELP_PRINTED) {
            exitCode = 0;
        } else {
            std::cerr << "ERROR: Could not parse command line options"
                      << std::endl;
            std::cerr << "Run " << argv[0]
                      << " --help to get an overview of avaialble options"
                      << std::endl;
        }
    }

    return exitCode;
}
