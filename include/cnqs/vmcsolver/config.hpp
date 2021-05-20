#ifndef CNQS_VMCSOLVER_CONFIG_HPP
#define CNQS_VMCSOLVER_CONFIG_HPP

#include <string>
#include <vector>

namespace cnqs {

namespace vmcsolver {

/// Simulation configuration
class Config {
public:
    Config(const std::string &file_name);

    void UpdateOutputPrefix(const std::string &new_output_prefix);

    void CreateDirs() const;

    void Output() const;

    int hamiltonian_num_rotor;
    int hamiltonian_vertex_weight;
    int hamiltonian_num_edges;
    std::vector<int> hamiltonian_edge_vertex_0;
    std::vector<int> hamiltonian_edge_vertex_1;
    std::vector<double> hamiltonian_edge_weights;

    int rbm_num_hidden;

    int metropolis_num_steps;
    int metropolis_warm_steps;
    int metropolis_cherry_pick;
    double metropolis_bump_size;
    bool metropolis_bump_single;
    bool metropolis_save_samples;

    int gradient_descent_num_steps;
    double gradient_descent_learning_rate;

    double stochastic_reconfig_regularization;

    std::string output_prefix;
};

}  // namespace vmcsolver

}  // namespace cnqs

#endif
