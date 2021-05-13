#include "cnqs/config.hpp"

#include <yaml-cpp/yaml.h>

#include <ctime>
#include <fstream>
#include <iostream>

cnqs::Config::Config(const std::string &file_name) {
    YAML::Node config = YAML::LoadFile(file_name);

    {
        YAML::Node hamiltonian = config["hamiltonian"];

        if (hamiltonian.Type() == YAML::NodeType::Scalar) {
            std::string hamiltonian_file = hamiltonian.as<std::string>();
            hamiltonian = YAML::LoadFile(hamiltonian_file);
        }

        hamiltonian_num_rotor = hamiltonian["num_rotor"].as<int>();
        hamiltonian_vertex_weight = hamiltonian["vertex_weight"].as<double>();

        hamiltonian_num_edges = hamiltonian["edges"].size();

        hamiltonian_edge_vertex_0.resize(hamiltonian_num_edges);
        hamiltonian_edge_vertex_1.resize(hamiltonian_num_edges);
        hamiltonian_edge_weights.resize(hamiltonian_num_edges);
        for (int i = 0; i < hamiltonian_num_edges; ++i) {
            hamiltonian_edge_vertex_0[i] =
                hamiltonian["edges"][i]["j"].as<int>();
            hamiltonian_edge_vertex_1[i] =
                hamiltonian["edges"][i]["k"].as<int>();
            hamiltonian_edge_weights[i] =
                hamiltonian["edges"][i]["beta"].as<double>();
        }
    }

    rbm_num_hidden = config["rbm"]["num_hidden"].as<int>();

    {
        YAML::Node metropolis = config["metropolis"];

        metropolis_num_steps = metropolis["num_steps"].as<int>();
        metropolis_warm_steps = metropolis["warm_steps"].as<int>();
        metropolis_cherry_pick = metropolis["cherry_pick"].as<int>();
        metropolis_bump_size = metropolis["bump_size"].as<double>();
        metropolis_bump_single = metropolis["bump_single"].as<bool>();
    }

    {
        YAML::Node gradient_descent = config["gradient_descent"];

        gradient_descent_num_steps = gradient_descent["num_steps"].as<int>();
        gradient_descent_learning_rate = gradient_descent["lr"].as<double>();
    }

    stochastic_reconfig_regularization =
        config["stoch_reconfig"]["sr_reg"].as<double>();

    std::time_t time = std::time(nullptr);
    char time_string[20];
    std::strftime(time_string, 20, "%Y-%m-%d_%H-%M-%S", std::localtime(&time));
    output_prefix = time_string;
}

void cnqs::Config::Output() const {
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "hamiltonian";
    out << YAML::Value;
    {
        out << YAML::BeginMap;
        out << YAML::Key << "num_rotor";
        out << YAML::Value << hamiltonian_num_rotor;
        out << YAML::Key << "vertex_weight";
        out << YAML::Value << hamiltonian_vertex_weight;
        out << YAML::Key << "edges";
        out << YAML::Value;
        {
            out << YAML::BeginSeq;
            for (int i = 0; i < hamiltonian_num_edges; ++i) {
                out << YAML::BeginMap;
                out << YAML::Key << "j";
                out << YAML::Value << hamiltonian_edge_vertex_0[i];
                out << YAML::Key << "k";
                out << YAML::Value << hamiltonian_edge_vertex_1[i];
                out << YAML::Key << "beta";
                out << YAML::Value << hamiltonian_edge_weights[i];
                out << YAML::EndMap;
            }
            out << YAML::EndSeq;
        }
        out << YAML::EndMap;
    }
    out << YAML::Key << "rbm";
    out << YAML::Value;
    {
        out << YAML::BeginMap;
        out << YAML::Key << "num_hidden";
        out << YAML::Value << rbm_num_hidden;
        out << YAML::EndMap;
    }
    out << YAML::Key << "metropolis";
    out << YAML::Value;
    {
        out << YAML::BeginMap;
        out << YAML::Key << "num_steps";
        out << YAML::Value << metropolis_num_steps;
        out << YAML::Key << "warm_steps";
        out << YAML::Value << metropolis_warm_steps;
        out << YAML::Key << "cherry_pick";
        out << YAML::Value << metropolis_cherry_pick;
        out << YAML::Key << "bump_size";
        out << YAML::Value << metropolis_bump_size;
        out << YAML::Key << "bump_single";
        out << YAML::Value << metropolis_bump_single;
        out << YAML::EndMap;
    }
    out << YAML::Key << "gradient_descent";
    out << YAML::Value;
    {
        out << YAML::BeginMap;
        out << YAML::Key << "num_steps";
        out << YAML::Value << gradient_descent_num_steps;
        out << YAML::Key << "lr";
        out << YAML::Value << gradient_descent_learning_rate;
        out << YAML::EndMap;
    }
    out << YAML::Key << "stoch_reconfig";
    out << YAML::Value;
    {
        out << YAML::BeginMap;
        out << YAML::Key << "sr_reg";
        out << YAML::Value << stochastic_reconfig_regularization;
        out << YAML::EndMap;
    }
    out << YAML::Key << "output_prefix";
    out << YAML::Value << output_prefix;
    out << YAML::EndMap;

    std::ofstream output_file(output_prefix + "_config.yaml");
    if (!output_file.is_open()) {
        throw std::runtime_error("Could not open file to write configuration");
    }
    output_file << out.c_str() << std::endl;
}
