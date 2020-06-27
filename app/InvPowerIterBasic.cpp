#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "CnqsBasicOperator.hpp"
#include "CnqsTrivialPreconditioner.hpp"
#include "CnqsVector.hpp"
#include "InvPowerIter.hpp"

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "ERROR: Please specify a simulation configuration"
                  << std::endl;
        std::cout << "USAGE: " << argv[0] << "<simulation_parameters.txt>"
                  << std::endl;
        return 1;
    }

    // simulation parameters
    int d = 0;
    int n = 0;
    std::vector<std::tuple<int, int>> edges;
    double g = 0.0;
    double J = 0.0;
    int cg_max_iter = 0;
    double cg_tol = 0.0;
    int power_max_iter = 0;
    double power_tol = 0.0;
    std::string file_name = "";

    // parse parameters
    {
        std::ifstream file(argv[1]);
        std::string line;

        std::getline(file, line);
        file >> d;
        std::getline(file, line);

        std::getline(file, line);
        file >> n;
        std::getline(file, line);

        int num_edges = 0;

        std::getline(file, line);
        file >> num_edges;
        std::getline(file, line);

        std::getline(file, line);
        edges.reserve(num_edges);
        for (int i = 0; i < num_edges; ++i) {
            int j = 0;
            int k = 0;

            file >> j >> k;
            std::getline(file, line);

            edges.emplace_back(std::make_tuple(j, k));
        }

        std::getline(file, line);
        file >> g;
        std::getline(file, line);

        std::getline(file, line);
        file >> J;
        std::getline(file, line);

        std::getline(file, line);
        file >> cg_max_iter;
        std::getline(file, line);

        std::getline(file, line);
        file >> cg_tol;
        std::getline(file, line);

        std::getline(file, line);
        file >> power_max_iter;
        std::getline(file, line);

        std::getline(file, line);
        file >> power_tol;
        std::getline(file, line);

        std::getline(file, line);
        file >> file_name;
        std::getline(file, line);

        file.close();
    }

    // dump the simulation parameters to screen
    {
        std::cout << "d: " << d << std::endl;
        std::cout << "n: " << n << std::endl;
        std::cout << "edges: " << std::flush;
        for (const auto &edge : edges) {
            std::cout << "(" << std::get<0>(edge) << ", " << std::get<1>(edge)
                      << "), " << std::flush;
        }
        std::cout << std::endl;
        std::cout << "g: " << g << std::endl;
        std::cout << "J: " << J << std::endl;
        std::cout << "cg_max_iter: " << cg_max_iter << std::endl;
        std::cout << "cg_tol: " << cg_tol << std::endl;
        std::cout << "power_max_iter: " << power_max_iter << std::endl;
        std::cout << "power_tol: " << power_tol << std::endl;
        std::cout << "file_name: " << file_name << std::endl;
    }

    // run main program
    {
        // allocate memory for state
        int num_element = 1;
        for (int i = 0; i < d; ++i) {
            num_element *= n;
        }

        CnqsVector cnqs_vector(num_element);

        // construct hamiltonian
        auto cnqs_operator =
            std::make_shared<const CnqsBasicOperator>(d, n, edges, g, J);

        // construct preconditioner
        auto cnqs_preconditioner =
            std::make_shared<const CnqsTrivialPrecondtioner>();

        // construct inverse power iterator
        InvPowerIter iterator(cnqs_operator, cnqs_operator->EigValLowerBound(),
                              cnqs_preconditioner);
        iterator.SetCgIterParams(cg_max_iter, cg_tol);
        iterator.SetPowerIterParams(power_max_iter, power_tol);

        std::cout << iterator << std::endl;

        // compute minimal eigenvalue state
        iterator.FindMinimalEigenState(cnqs_vector);

        // save state, if requested
        if (file_name.compare("NULL") != 0) {
            cnqs_vector.Save(file_name);
        }
    }

    return 0;
}
