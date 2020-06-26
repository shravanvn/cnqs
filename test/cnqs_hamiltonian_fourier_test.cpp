#include "gtest/gtest.h"

#include <tuple>
#include <vector>

#include "cnqs_hamiltonian_fourier.hpp"
#include "cnqs_state.hpp"

TEST(cnqs_hamiltonian_fourier, constructor_params) {
    int d = 2;
    int n = 17;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;

    CnqsHamiltonianFourier hamiltonian(d, n, edges, g, J);
}

TEST(cnqs_hamiltonian_fourier, initialize_state) {
    int d = 2;
    int n = 17;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;

    CnqsHamiltonianFourier hamiltonian(d, n, edges, g, J);

    CnqsState state = hamiltonian.initialize_state();
    state.save("cnqs_hamiltonian_fourier_initial_state.txt");
}

TEST(cnqs_hamiltonian_fourier, operator_apply_no_edge) {
    int d = 2;
    int n = 17;
    std::vector<std::tuple<int, int>> edges{};
    double g = 1.0;
    double J = 1.0;

    CnqsHamiltonianFourier hamiltonian(d, n, edges, g, J);

    CnqsState state = hamiltonian.initialize_state();
    state.save("cnqs_hamiltonian_fourier_operator_apply_no_edge_state_0.txt");

    CnqsState new_state = hamiltonian * state;
    new_state.save(
        "cnqs_hamiltonian_fourier_operator_apply_no_edge_state_1.txt");
}

TEST(cnqs_hamiltonian_fourier, operator_apply_one_edge) {
    int d = 2;
    int n = 17;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;

    CnqsHamiltonianFourier hamiltonian(d, n, edges, g, J);

    CnqsState state = hamiltonian.initialize_state();
    state.save("cnqs_hamiltonian_fourier_operator_apply_one_edge_state_0.txt");

    CnqsState new_state = hamiltonian * state;
    new_state.save(
        "cnqs_hamiltonian_fourier_operator_apply_one_edge_state_1.txt");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
