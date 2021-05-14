#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "cnqs/config.hpp"
#include "cnqs/nqs.hpp"
#include "cnqs/optimization.hpp"
#include "cnqs/sampler.hpp"

int main(int argc, char **argv) {
    if (argc == 2 && (std::string(argv[1]).compare("-h") == 0 ||
                      std::string(argv[1]).compare("--help") == 0)) {
        std::cout << "USAGE: " << argv[0] << " <config.yaml> [<output_prefix>]"
                  << std::endl;
        return 0;
    }

    if (argc == 2 || argc == 3) {
        // start time
        auto start_time = std::chrono::high_resolution_clock::now();

        // construct config
        cnqs::Config config(argv[1]);

        // overwrite autogenerated time-stamp output_prefix if third command
        // line argument is supplied
        if (argc == 3) {
            config.output_prefix = argv[2];
        }

        // output configuration
        config.Output();

        // construct random number generator
        std::mt19937 rng(0);

        // construct initial NQS and randomly initialize
        cnqs::Nqs nqs(config);
        nqs.RandInit(rng);

        // open output log file
        std::string output_file_name = config.output_prefix + "_output.csv";

        std::ofstream output_file(output_file_name);
        if (!output_file.is_open()) {
            throw std::runtime_error("Could not open output log file");
        }

        // write header
        output_file << "step,visible_bias_norm,hidden_bias_norm,"
                    << "acceptance_rate,energy_avg,energy_std,gradient_norm"
                    << std::endl;

        // main loop
        for (int t = 0; t < config.gradient_descent_num_steps; ++t) {
            int num_vars = nqs.NumVars();
            double visible_bias_norm = nqs.VisibleBiasNorm();
            double hidden_bias_norm = nqs.HiddenBiasNorm();

            double local_energy_avg;
            double local_energy_std;
            std::vector<double> log_psi_gradient_avg(num_vars);
            std::vector<double> log_psi_gradient_outer_avg(num_vars * num_vars);
            std::vector<double> local_energy_log_psi_gradient_avg(num_vars);
            double acceptance_rate;
            cnqs::MetropolisSampler(
                config, nqs, local_energy_avg, local_energy_std,
                log_psi_gradient_avg, log_psi_gradient_outer_avg,
                local_energy_log_psi_gradient_avg, acceptance_rate, rng);

            double gradient_norm;
            cnqs::StochasticReconfiguration(
                config, nqs, local_energy_avg, log_psi_gradient_avg,
                log_psi_gradient_outer_avg, local_energy_log_psi_gradient_avg,
                gradient_norm);

            output_file << t << "," << std::scientific << visible_bias_norm
                        << "," << hidden_bias_norm << "," << acceptance_rate
                        << "," << local_energy_avg << "," << local_energy_std
                        << "," << gradient_norm << std::defaultfloat
                        << std::endl;
        }

        output_file.close();

        // stop time
        auto stop_time = std::chrono::high_resolution_clock::now();

        // report elapsed time
        std::chrono::duration<double> elapsed_time = stop_time - start_time;
        std::cout << "Elapsed time = " << elapsed_time.count() << " seconds"
                  << std::endl;

        return 0;
    }

    std::cerr
        << "ERROR: Could not understand command line arguments (use --help)"
        << std::endl;
    return 1;
}
