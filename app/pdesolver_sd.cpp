#include <Teuchos_CommandLineProcessor.hpp>
#include <Tpetra_Core.hpp>
#include <iostream>
#include <memory>
#include <string>

#include "cnqs/pdesolver/hamiltonian.hpp"
#include "cnqs/pdesolver/spectral_difference_problem.hpp"

int main(int argc, char **argv) {
    int exitCode = 1;

    Tpetra::ScopeGuard tpetraScope(&argc, &argv);
    {
        std::string hamiltonianFileName = "hamiltonian.yaml";
        long maxFreq = 32;
        long maxPowerIter = 10000;
        double tolPowerIter = 1.0e-15;
        long maxCgIter = 10000;
        double tolCgIter = 1.0e-15;
        std::string groundStateFileName = "";

        auto cmdParser = Teuchos::CommandLineProcessor(false, true, true);
        cmdParser.setOption(
            "hamiltonian-file-name", &hamiltonianFileName,
            "File containing YAML description of rotor hamiltonian", true);
        cmdParser.setOption("max-frequency", &maxFreq,
                            "Frequency cutoff in Fourier expansions", true);
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
            const auto hamiltonian =
                std::make_shared<cnqs::pdesolver::Hamiltonian>(
                    hamiltonianFileName);
            cnqs::pdesolver::SpectralDifferenceProblem problem(hamiltonian,
                                                               maxFreq, comm);
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
