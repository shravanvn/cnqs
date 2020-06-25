#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "cnqs_hamiltonian_direct.hpp"

int main(int argc, char **argv) {
    // validate number of command line arguments
    if (argc == 2 && std::string(argv[1]).compare("--help") == 0) {
        std::cout << "USAGE: " << argv[0]
                  << " <d> <n> <g> <J> <cg_max_it> <cg_tol> <power_max_it> "
                     "<power_tol> <file_name>"
                  << std::endl;

        return 1;
    }

    if (argc != 10) {
        std::cout << "ERROR: Incorrect number of command line arguments"
                  << std::endl
                  << "       Run '" << argv[0]
                  << " --help' for the exact syntax" << std::endl;
        return 1;
    }

    // create Hamiltonian
    int d = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    double g = std::atof(argv[3]);
    double J = std::atof(argv[4]);

    std::vector<std::tuple<int, int>> edges(d - 1);
    for (int i = 1; i <= d - 1; ++i) {
        edges[i - 1] = std::make_tuple(0, i);
    }

    std::cout << "Hamiltonian parameters:" << std::endl
              << "        d : " << d << std::endl
              << "        n : " << n << std::endl
              << "        g : " << g << std::endl
              << "        J : " << J << std::endl
              << "    edges : ";
    for (const std::tuple<int, int> &edge : edges) {
        std::cout << "(" << std::get<0>(edge) << ", " << std::get<1>(edge)
                  << "), ";
    }
    std::cout << std::endl;

    CnqsHamiltonianDirect hamiltonian(d, n, edges, g, J);

    // setup parameters for inverse power iteration
    int cg_max_it = std::atoi(argv[5]);
    double cg_tol = std::atof(argv[6]);
    int power_max_it = std::atoi(argv[7]);
    double power_tol = std::atof(argv[8]);

    // run inverse power iteration
    std::string file_name = argv[9];
    hamiltonian.inverse_power_iteration(cg_max_it, cg_tol, power_max_it,
                                        power_tol, file_name);

    return 0;
}
