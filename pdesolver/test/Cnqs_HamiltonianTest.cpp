#include <iostream>
#include <tuple>
#include <vector>

#include "Cnqs_Hamiltonian.hpp"

int main() {
    {
        int num_rotor = 4;
        double vertex_weight = 5.0;
        std::vector<std::tuple<int, int, double>> edges{
            {0, 1, 0.5}, {1, 2, -1.5}, {2, 3, -2.5}, {3, 0, 1.5}};

        Cnqs::Hamiltonian<double, int> hamiltonian(num_rotor, vertex_weight,
                                                   edges);

        std::cout << "Sum of weights: " << std::scientific
                  << hamiltonian.sumEdgeWeights() << std::endl;
        std::cout << "Sum of absolute values of weights: " << std::scientific
                  << hamiltonian.sumAbsEdgeWeights() << std::endl;
    }

    {
        Cnqs::Hamiltonian<double, int> hamiltonian("hamiltonian.yaml");

        std::cout << "Sum of weights: " << std::scientific
                  << hamiltonian.sumEdgeWeights() << std::endl;
        std::cout << "Sum of absolute values of weights: " << std::scientific
                  << hamiltonian.sumAbsEdgeWeights() << std::endl;
    }

    return 0;
}
