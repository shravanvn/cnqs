#include <iostream>
#include <tuple>
#include <vector>

#include "CnqsBasicOperator.hpp"
#include "gtest/gtest.h"

TEST(CnqsBasicOperator, ConstructFromParameters) {
    int num_rotor = 2;
    int num_grid_point = 128;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;

    CnqsBasicOperator cnqs_operator(num_rotor, num_grid_point, edges, g, J);

    std::cout << cnqs_operator << std::endl;
}

TEST(CnqsBasicOperator, ConstructInitialState) {
    int num_rotor = 2;
    int num_grid_point = 128;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;
    int num_element = num_grid_point * num_grid_point;

    CnqsBasicOperator cnqs_operator(num_rotor, num_grid_point, edges, g, J);
    CnqsVector cnqs_vector(num_element);

    cnqs_operator.ConstructInitialState(cnqs_vector);

    cnqs_vector.Save("cnqs_basic_operator_initial_state.txt");
}

TEST(CnqsBasicOperator, Apply) {
    int num_rotor = 2;
    int num_grid_point = 128;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;
    int num_element = num_grid_point * num_grid_point;

    CnqsBasicOperator cnqs_operator(num_rotor, num_grid_point, edges, g, J);

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
