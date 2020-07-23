#include <iostream>
#include <memory>

#include <Tpetra_Core.hpp>

#include "Cnqs_BasicProblem.hpp"
#include "Cnqs_Network.hpp"

int main(int argc, char **argv) {

    Tpetra::ScopeGuard tpetraScope(&argc, &argv);
    {
        int num_rotor = 2;
        std::vector<std::tuple<int, int, double>> edges{{0, 1, 1.0}};
        auto network = std::make_shared<Cnqs::Network>(num_rotor, edges);

        int numGridPoint = 32;
        auto comm = Tpetra::getDefaultComm();
        Cnqs::BasicProblem problem(network, numGridPoint, comm);

        if (comm->getRank() == 0) {
            std::cout << problem << std::endl;
        }
    }

    return 0;
}
