#include "gtest/gtest.h"

#include <tuple>
#include <vector>

#include "cnqs_hamiltonian.hpp"
#include "cnqs_state.hpp"

TEST(cnqs_hamiltonian, constructor_params) {
    unsigned long d = 2;
    unsigned long n = 128;
    std::vector<const std::tuple<unsigned long, unsigned long>> edges{{1, 2}};
    double g = 1.0;
    double J = 1.0;

    CnqsHamiltonian hamiltonian(d, n, edges, g, J);
}

TEST(cnqs_hamiltonian, initialize_state) {
    unsigned long d = 2;
    unsigned long n = 128;
    std::vector<const std::tuple<unsigned long, unsigned long>> edges{{1, 2}};
    double g = 1.0;
    double J = 1.0;

    CnqsHamiltonian hamiltonian(d, n, edges, g, J);

    CnqsState state = hamiltonian.initialize_state();
    state.save("cnqs_hamiltonian_initial_state.h5");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
