#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

#include "CnqsFourierOperator.hpp"
#include "CnqsFourierPreconditioner.hpp"
#include "gtest/gtest.h"

TEST(CnqsFourierPreconditioner, ConstructFromParameters) {
    int d = 2;
    int n = 17;
    double g = 1.0;
    double J = 1.0;

    CnqsFourierPreconditioner preconditioner(d, n, g, J, -1.0);

    std::cout << preconditioner << std::endl;
}

TEST(CnqsFourierPreconditioner, Solve) {
    const double PI = 4.0 * std::atan(1.0);

    int d = 2;
    int n = 17;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;
    int num_element = n * n;

    CnqsFourierOperator cnqs_operator(d, n, edges, g, J);
    CnqsFourierPreconditioner cnqs_preconditioner(
        d, n, g, J, cnqs_operator.EigValLowerBound());

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
