#include <iostream>
#include <tuple>
#include <vector>

#include "Cnqs_Network.hpp"

int main() {
    {
        int num_rotor = 4;
        std::vector<std::tuple<int, int, double>> edges{
            {0, 1, 0.5}, {1, 2, -1.5}, {2, 3, -2.5}, {3, 0, 1.5}};
        Cnqs::Network<double, int> network(num_rotor, edges);

        std::cout << network << std::endl;

        std::cout << "Lower bound for Hamiltonian eigenvalue: "
                  << std::scientific << network.eigValLowerBound() << std::endl;
    }

    {
        Cnqs::Network<double, int> network("network.json");

        std::cout << network << std::endl;

        std::cout << "Lower bound for Hamiltonian eigenvalue: "
                  << std::scientific << network.eigValLowerBound() << std::endl;
    }

    return 0;
}
