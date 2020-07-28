#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_MultiVector.hpp>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "Cnqs_BasicProblem.hpp"
#include "Cnqs_Network.hpp"

void printHelp(const std::string &programName) {
    std::cout << "USAGE:" << std::endl
              << "    " << programName
              << " <networkFileName> <numGridPoint> [<maxPowerIter> "
                 "<tolPowerIter> <maxCgIter> <tolCgIter> [<fileName>]]"
              << std::endl;
    std::cout << "DEFAULTS:" << std::endl
              << "    maxPowerIter : 100" << std::endl
              << "    tolPowerIter : 1.0e-06" << std::endl
              << "    maxCgIter    : 1000" << std::endl
              << "    tolCgIter    : 1.0e-06" << std::endl
              << "    fileName     : \"\"" << std::endl;
}

int main(int argc, char **argv) {
    using Scalar = Tpetra::MultiVector<>::scalar_type;
    using LocalOrdinal = Tpetra::MultiVector<>::local_ordinal_type;
    using GlobalOrdinal = Tpetra::MultiVector<>::global_ordinal_type;
    using Node = Tpetra::MultiVector<>::node_type;

    Tpetra::ScopeGuard tpetraScope(&argc, &argv);
    {
        const auto comm = Tpetra::getDefaultComm();

        // parse command line arguments
        if ((argc == 2) && (std::string(argv[1]).compare("-h") == 0)) {
            if (comm->getRank() == 0) {
                printHelp(argv[0]);
            }
        } else if ((argc < 3) || (argc > 3 && argc < 7) || (argc > 8)) {
            if (comm->getRank() == 0) {
                std::cout << "ERROR: Incorrect number of command line arguments"
                          << std::endl
                          << "       Run \"" << argv[0] << " -h\" for help"
                          << std::endl;
            }
        } else {
            // argc = 3, 7 or 8

            auto network =
                std::make_shared<Cnqs::Network<Scalar, GlobalOrdinal>>(argv[1]);
            GlobalOrdinal numGridPoint = std::atoi(argv[2]);
            GlobalOrdinal maxPowerIter = 100;
            Scalar tolPowerIter = 1.0e-06;
            GlobalOrdinal maxCgIter = 1000;
            Scalar tolCgIter = 1.0e-06;
            std::string fileName("");

            if (argc > 3) {
                maxPowerIter = std::atoi(argv[3]);
                tolPowerIter = std::atof(argv[4]);
                maxCgIter = std::atoi(argv[5]);
                tolCgIter = std::atof(argv[6]);
            }

            if (argc > 7) {
                fileName = std::string(argv[7]);
            }

            // output full command for sanity check
            if (comm->getRank() == 0) {
                std::cout << "==main== Begin run configuration" << std::endl;
                std::cout << "network: " << *network << std::endl;
                std::cout << "num_grid_point: " << numGridPoint << std::endl;
                std::cout << "max_power_iter: " << maxPowerIter << std::endl;
                std::cout << "tol_power_iter: " << tolPowerIter << std::endl;
                std::cout << "max_cg_iter: " << maxCgIter << std::endl;
                std::cout << "tol_cg_iter: " << tolCgIter << std::endl;
                std::cout << "file_name: " << fileName << std::endl;
                std::cout << "==main== End run configuration" << std::endl;
            }

            // create Fourier problem
            Cnqs::BasicProblem<Scalar, LocalOrdinal, GlobalOrdinal, Node>
                problem(network, numGridPoint, comm);

            // solve Fourier problem
            problem.runInversePowerIteration(maxPowerIter, tolPowerIter,
                                             maxCgIter, tolCgIter, fileName);
        }
    }

    return 0;
}
