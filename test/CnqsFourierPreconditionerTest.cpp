#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

#include "CnqsFourierOperator.hpp"
#include "CnqsFourierPreconditioner.hpp"
#include "gtest/gtest.h"

TEST(CnqsFourierPreconditioner, ConstructFromParameters) {
    int num_rotor = 2;
    int max_freq = 8;
    double g = 1.0;
    double J = 1.0;

    CnqsFourierPreconditioner preconditioner(num_rotor, 2 * max_freq + 1, g, J,
                                             -1.0);

    std::cout << preconditioner << std::endl;
}

TEST(CnqsFourierPreconditioner, Solve) {
    const double PI = 4.0 * std::atan(1.0);

    int num_rotor = 2;
    int max_freq = 8;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;
    int num_element = (2 * max_freq + 1) * (2 * max_freq + 1);

    CnqsFourierOperator cnqs_operator(num_rotor, max_freq, edges, g, J);
    CnqsFourierPreconditioner cnqs_preconditioner(
        num_rotor, 2 * max_freq + 1, g, J, cnqs_operator.EigValLowerBound());

    CnqsVector cnqs_vector_0(num_element);
    cnqs_operator.ConstructInitialState(cnqs_vector_0);

    CnqsVector cnqs_vector_1(num_element);
    cnqs_preconditioner.Solve(cnqs_vector_0, cnqs_vector_1);

    cnqs_vector_0.Save("cnqs_fourier_preconditioner_solve_state_0.txt");
    cnqs_vector_1.Save("cnqs_fourier_preconditioner_solve_state_1.txt");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
