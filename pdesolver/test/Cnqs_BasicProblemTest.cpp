#include <Tpetra_Core.hpp>
#include <Tpetra_MultiVector.hpp>
#include <iostream>
#include <memory>

#include "Cnqs_BasicProblem.hpp"
#include "Cnqs_Network.hpp"

int main(int argc, char **argv) {
    using Real = Tpetra::MultiVector<>::scalar_type;
    using LocalOrdinal = Tpetra::MultiVector<>::local_ordinal_type;
    using GlobalOrdinal = Tpetra::MultiVector<>::global_ordinal_type;
    using NodeType = Tpetra::MultiVector<>::node_type;

    Tpetra::ScopeGuard tpetraScope(&argc, &argv);
    {
        const GlobalOrdinal num_rotor = 2;
        const std::vector<std::tuple<GlobalOrdinal, GlobalOrdinal, Real>>
            edges{{0, 1, 1.0}};
        const auto network =
            std::make_shared<Cnqs::Network<Real, GlobalOrdinal>>(num_rotor,
                                                                   edges);
        const Real laplaceFactor = 1.0;
        const GlobalOrdinal numGridPoint = 256;
        const auto comm = Tpetra::getDefaultComm();
        const Cnqs::BasicProblem<Real, LocalOrdinal, GlobalOrdinal, NodeType>
            problem(network, laplaceFactor, numGridPoint, comm);

        const GlobalOrdinal maxPowerIter = 100;
        const Real tolPowerIter = 1.0e-06;
        const GlobalOrdinal maxCgIter = 1000;
        const Real tolCgIter = 1.0e-06;
        const std::string fileName = "cnqs_basic_problem_state.mm";

        comm->barrier();
        const Real lambda = problem.runInversePowerIteration(
            maxPowerIter, tolPowerIter, maxCgIter, tolCgIter, fileName);

        comm->barrier();
        if (comm->getRank() == 0) {
            std::cout << "Estimated smallest eigenvalue: " << std::scientific
                      << lambda << std::endl;
        }
    }

    return 0;
}
