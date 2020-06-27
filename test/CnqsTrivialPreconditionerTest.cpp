#include <cmath>
#include <iostream>

#include "CnqsTrivialPreconditioner.hpp"
#include "gtest/gtest.h"

TEST(CnqsBasicOperator, ConstructFromParameters) {
    CnqsTrivialPrecondtioner preconditioner;

    std::cout << preconditioner << std::endl;
}

TEST(CnqsBasicOperator, Solve) {
    const double PI = 4.0 * std::atan(1.0);

    CnqsTrivialPrecondtioner preconditioner;

    CnqsVector cnqs_vector_0(100);
    for (int i = 0; i < 100; ++i) {
        cnqs_vector_0(i) = std::cos(0.02 * PI * i);
    }

    CnqsVector cnqs_vector_1;

    preconditioner.Solve(cnqs_vector_0, cnqs_vector_1);

    for (int i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(cnqs_vector_0(i), cnqs_vector_1(i));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
