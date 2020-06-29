#include <iostream>
#include <tuple>
#include <vector>

#include "CnqsFourierOperator.hpp"
#include "gtest/gtest.h"

TEST(CnqsFourierOperator, ConstructFromParameters) {
    int num_rotor = 2;
    int max_freq = 8;
    std::vector<std::tuple<int, int>> edges{{1, 0}};
    double g = 1.0;
    double J = 1.0;

    CnqsFourierOperator cnqs_operator(num_rotor, max_freq, edges, g, J);

    std::cout << cnqs_operator << std::endl;
}

TEST(CnqsFourierOperator, ConstructInitialState) {
    int num_rotor = 2;
    int max_freq = 8;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;
    int num_element = (2 * max_freq + 1) * (2 * max_freq + 1);

    CnqsFourierOperator cnqs_operator(num_rotor, max_freq, edges, g, J);
    CnqsVector cnqs_vector(num_element);

    cnqs_operator.ConstructInitialState(cnqs_vector);

    cnqs_vector.Save("cnqs_fourier_operator_initial_state.txt");
}

TEST(CnqsFourierOperator, Apply) {
    int num_rotor = 2;
    int max_freq = 8;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;
    int num_element = (2 * max_freq + 1) * (2 * max_freq + 1);

    CnqsFourierOperator cnqs_operator(num_rotor, max_freq, edges, g, J);

    CnqsVector cnqs_vector_0(num_element);
    CnqsVector cnqs_vector_1(num_element);

    cnqs_operator.ConstructInitialState(cnqs_vector_0);
    cnqs_operator.Apply(cnqs_vector_0, cnqs_vector_1);

    cnqs_vector_0.Save("cnqs_fourier_operator_apply_state_0.txt");
    cnqs_vector_1.Save("cnqs_fourier_operator_apply_state_1.txt");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
