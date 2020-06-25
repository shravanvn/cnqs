#include "gtest/gtest.h"

#include <tuple>
#include <vector>

#include "cnqs_hamiltonian.hpp"
#include "cnqs_state.hpp"

TEST(cnqs_hamiltonian, constructor_params) {
    int d = 2;
    int n = 128;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;

    CnqsHamiltonian hamiltonian(d, n, edges, g, J);
}

TEST(cnqs_hamiltonian, initialize_state) {
    int d = 2;
    int n = 128;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;

    CnqsHamiltonian hamiltonian(d, n, edges, g, J);

    CnqsState state = hamiltonian.initialize_state();
    state.save("cnqs_hamiltonian_initial_state.h5", 0);
}

TEST(cnqs_hamiltonian, operator_apply_no_edge) {
    int d = 2;
    int n = 128;
    std::vector<std::tuple<int, int>> edges{};
    double g = 1.0;
    double J = 1.0;

    CnqsHamiltonian hamiltonian(d, n, edges, g, J);

    CnqsState state = hamiltonian.initialize_state();
    state.save("cnqs_hamiltonian_operator_apply_no_edge.h5", 0);

    CnqsState new_state = hamiltonian * state;
    new_state.save("cnqs_hamiltonian_operator_apply_no_edge.h5", 1);
}

TEST(cnqs_hamiltonian, operator_apply_one_edge) {
    int d = 2;
    int n = 128;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;

    CnqsHamiltonian hamiltonian(d, n, edges, g, J);

    CnqsState state = hamiltonian.initialize_state();
    state.save("cnqs_hamiltonian_operator_apply_one_edge.h5", 0);

    CnqsState new_state = hamiltonian * state;
    new_state.save("cnqs_hamiltonian_operator_apply_one_edge.h5", 1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
