#include <iostream>
#include <memory>

#include <Tpetra_Core.hpp>

#include "Cnqs_FourierProblem.hpp"
#include "Cnqs_Network.hpp"

int main(int argc, char **argv) {
    Tpetra::ScopeGuard tpetraScope(&argc, &argv);
    {
        const int num_rotor = 2;
        const std::vector<std::tuple<int, int, double>> edges{{0, 1, 1.0}};
        const auto network = std::make_shared<Cnqs::Network>(num_rotor, edges);

        const int maxFreq = 8;
        const auto comm = Tpetra::getDefaultComm();
        const Cnqs::FourierProblem problem(network, maxFreq, comm);

        if (comm->getRank() == 0) {
            std::cout << problem << std::endl;
        }

        const int maxPowerIter = 100;
        const double tolPowerIter = 1.0e-06;
        const int maxCgIter = 1000;
        const double tolCgIter = 1.0e-06;
        const std::string fileName = "cnqs_fourier_problem_state.mm";

        comm->barrier();
        const double lambda = problem.runInversePowerIteration(
            maxPowerIter, tolPowerIter, maxCgIter, tolCgIter, fileName);

        comm->barrier();
        if (comm->getRank() == 0) {
            std::cout << "Estimated smallest eigenvalue: " << std::scientific
                      << lambda << std::endl;
        }
    }

    return 0;
}
