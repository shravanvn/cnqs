#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "CnqsBasicOperator.hpp"
#include "CnqsTrivialPreconditioner.hpp"
#include "InvPowerIter.hpp"
#include "gtest/gtest.h"

TEST(InvPowerIter, ConstructFromParameters) {
    int num_rotor_ = 2;
    int num_grid_point = 128;
    std::vector<std::tuple<int, int>> edges{{1, 0}};
    double g = 1.0;
    double J = 1.0;

    auto cnqs_operator = std::make_shared<const CnqsBasicOperator>(
        num_rotor_, num_grid_point, edges, g, J);

    auto cnqs_preconditioner =
        std::make_shared<const CnqsTrivialPreconditioner>();

    InvPowerIter iterator(cnqs_operator, 0.0, cnqs_preconditioner);
    iterator.SetCgIterParams(1000, 1.0e-06);
    iterator.SetPowerIterParams(100, 1.0e-06);

    std::cout << iterator << std::endl;
}

TEST(InvPowerIter, FindMinimalEigenState) {
    int num_rotor_ = 2;
    int num_grid_point = 32;
    std::vector<std::tuple<int, int>> edges{{0, 1}};
    double g = 1.0;
    double J = 1.0;
    int num_element = num_grid_point * num_grid_point;

    auto cnqs_operator = std::make_shared<const CnqsBasicOperator>(
        num_rotor_, num_grid_point, edges, g, J);

    auto cnqs_preconditioner =
        std::make_shared<const CnqsTrivialPreconditioner>();

    InvPowerIter iterator(cnqs_operator, cnqs_operator->EigValLowerBound(),
                          cnqs_preconditioner);
    iterator.SetCgIterParams(1000, 1.0e-02);
    iterator.SetPowerIterParams(100, 1.0e-02);

    CnqsVector vector(num_element);
    cnqs_operator->ConstructInitialState(vector);

    iterator.FindMinimalEigenState(vector);

    vector.Save("inv_power_iter_minimal_eigen_state.txt");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
