#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "CnqsFourierOperator.hpp"
#include "CnqsFourierPreconditioner.hpp"
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
    int num_rotor = 0;
    int max_freq = 0;
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
        file >> num_rotor;
        std::getline(file, line);

        std::getline(file, line);
        file >> max_freq;
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
        std::cout << "num_rotor: " << num_rotor << std::endl;
        std::cout << "max_freq: " << max_freq << std::endl;
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
        for (int i = 0; i < num_rotor; ++i) {
            num_element *= (2 * max_freq + 1);
        }

        CnqsVector cnqs_vector(num_element);

        // construct hamiltonian
        auto cnqs_operator = std::make_shared<const CnqsFourierOperator>(
            num_rotor, max_freq, edges, g, J);
        double eig_val_lower_bound = cnqs_operator->EigValLowerBound();

        // construct preconditioner
        auto cnqs_preconditioner =
            std::make_shared<const CnqsFourierPreconditioner>(
                num_rotor, max_freq, g, J, eig_val_lower_bound);

        // construct inverse power iterator
        InvPowerIter iterator(cnqs_operator, eig_val_lower_bound,
                              cnqs_preconditioner);
        iterator.SetCgIterParams(cg_max_iter, cg_tol);
        iterator.SetPowerIterParams(power_max_iter, power_tol);

        std::cout << iterator << std::endl;

        // compute minimal eigenvalue state, record execution time
        auto start_time = std::chrono::high_resolution_clock::now();
        iterator.FindMinimalEigenState(cnqs_vector);
        auto stop_time = std::chrono::high_resolution_clock::now();

        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::microseconds>(stop_time -
                                                                  start_time);
        std::cout << "Elapsed time: " << elapsed_time.count() << " us"
                  << std::endl;

        // save state, if requested
        if (file_name.compare("NULL") != 0) {
            cnqs_vector.Save(file_name);
        }
    }

    return 0;
}
