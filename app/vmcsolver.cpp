#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>

#include "cnqs/vmcsolver/config.hpp"
#include "cnqs/vmcsolver/nqs.hpp"
#include "cnqs/vmcsolver/optimization.hpp"
#include "cnqs/vmcsolver/sampler.hpp"

int main(int argc, char **argv) {
    if (argc == 2 && (std::string(argv[1]).compare("-h") == 0 ||
                      std::string(argv[1]).compare("--help") == 0)) {
        std::cout << "USAGE: " << argv[0] << " <config.yaml> [<output_prefix>]"
                  << std::endl;
        return 0;
    }

    if (argc == 2) {
        // start time
        auto start_time = std::chrono::high_resolution_clock::now();

        // construct config and save to file
        cnqs::vmcsolver::Config config(argv[1]);
        config.Output();

        // construct random number generator
        std::mt19937 rng(0);

        // construct initial NQS and randomly initialize
        cnqs::vmcsolver::Nqs nqs(config);
        nqs.RandInit(rng);

        const int num_vars = nqs.NumVars();

        // open output log file
        std::string output_file_name = config.output_prefix + "output.csv";

        std::ofstream output_file(output_file_name);
        if (!output_file.is_open()) {
            throw std::runtime_error("Could not open output log file");
        }

        // write header
        output_file << "step,visible_bias_norm,hidden_bias_norm,"
                    << "acceptance_rate,energy_avg,energy_std,gradient_norm"
                    << std::endl;

        // main loop
        for (int step = 0; step <= config.gradient_descent_num_steps; ++step) {
            if (config.output_model && (config.output_frequency > 0) &&
                (step % config.output_frequency == 0)) {
                nqs.Output(config.output_prefix + "model/step_" +
                           std::to_string(step) + ".txt");
            }

            double visible_bias_norm = nqs.VisibleBiasNorm();
            double hidden_bias_norm = nqs.HiddenBiasNorm();

            double local_energy_avg;
            double local_energy_std;
            std::vector<double> log_psi_gradient_avg(num_vars);
            std::vector<double> log_psi_gradient_outer_avg(num_vars * num_vars);
            std::vector<double> local_energy_log_psi_gradient_avg(num_vars);
            double acceptance_rate;
            cnqs::vmcsolver::MetropolisSampler(
                step, config, nqs, local_energy_avg, local_energy_std,
                log_psi_gradient_avg, log_psi_gradient_outer_avg,
                local_energy_log_psi_gradient_avg, acceptance_rate, rng);

            double gradient_norm = -std::numeric_limits<double>::infinity();
            if (step < config.gradient_descent_num_steps) {
                cnqs::vmcsolver::StochasticReconfiguration(
                    config, nqs, local_energy_avg, log_psi_gradient_avg,
                    log_psi_gradient_outer_avg,
                    local_energy_log_psi_gradient_avg, gradient_norm);
            }

            output_file << step << "," << std::scientific << visible_bias_norm
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
