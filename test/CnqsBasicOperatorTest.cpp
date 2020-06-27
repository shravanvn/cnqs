#include <iostream>
#include <tuple>
#include <vector>

#include "CnqsBasicOperator.hpp"
#include "gtest/gtest.h"

TEST(CnqsBasicOperator, ConstructFromParameters) {
    int d = 2;
    int n = 128;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;

    CnqsBasicOperator cnqs_operator(d, n, edges, g, J);

    std::cout << cnqs_operator << std::endl;
}

TEST(CnqsBasicOperator, ConstructInitialState) {
    int d = 2;
    int n = 128;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;
    int num_element = n * n;

    CnqsBasicOperator cnqs_operator(d, n, edges, g, J);
    CnqsVector cnqs_vector(num_element);

    cnqs_operator.ConstructInitialState(cnqs_vector);

    cnqs_vector.Save("cnqs_basic_operator_initial_state.txt");
}

TEST(CnqsBasicOperator, Apply) {
    int d = 2;
    int n = 128;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;
    int num_element = n * n;

    CnqsBasicOperator cnqs_operator(d, n, edges, g, J);

    CnqsVector cnqs_vector_0(num_element);
    CnqsVector cnqs_vector_1(num_element);

    cnqs_operator.ConstructInitialState(cnqs_vector_0);
    cnqs_operator.Apply(cnqs_vector_0, cnqs_vector_1);

    cnqs_vector_0.Save("cnqs_basic_operator_apply_state_0.txt");
    cnqs_vector_1.Save("cnqs_basic_operator_apply_state_1.txt");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
