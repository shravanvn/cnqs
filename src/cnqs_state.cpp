#include "cnqs_state.hpp"

void CnqsState::save(const std::string &file_name, unsigned long id) const {
    std::string data_set_name = "state_" + std::to_string(id);
    vec_.save(
        arma::hdf5_name(file_name, data_set_name, arma::hdf5_opts::replace));
}

void CnqsState::load(const std::string &file_name, unsigned long id) {
    std::string data_set_name = "state_" + std::to_string(id);
    vec_.load(arma::hdf5_name(file_name, data_set_name));
}
