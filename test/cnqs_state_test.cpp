#include "gtest/gtest.h"

#include <armadillo>
#include <cmath>
#include <iostream>

#include "cnqs_state.hpp"

TEST(cnqs_state, size_constructor) {
    CnqsState vec(100);

    for (unsigned long i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec(i), 0.0);
    }
}

TEST(cnqs_state, data_constructor) {
    arma::Col<double> data(100);
    for (unsigned long i = 0; i < 100; ++i) {
        data(i) = i;
    }

    CnqsState vec(data);

    for (unsigned long i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec(i), i);
    }
}

TEST(cnqs_state, vector_assignment) {
    arma::Col<double> data(100);
    for (unsigned long i = 0; i < 100; ++i) {
        data(i) = i;
    }

    CnqsState vec1(data);
    CnqsState vec2 = vec1;

    for (unsigned long i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec1(i), vec2(i));
    }
}

TEST(cnqs_state, scalar_assignment) {
    CnqsState vec(100);
    vec = 1.0;

    for (unsigned long i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec(i), 1.0);
    }
}

TEST(cnqs_state, vector_addition) {
    arma::Col<double> data1(100);
    for (unsigned long i = 0; i < 100; ++i) {
        data1(i) = i;
    }
    CnqsState vec1(data1);

    arma::Col<double> data2(100);
    for (unsigned long i = 0; i < 100; ++i) {
        data2(i) = 100.0 - i;
    }
    CnqsState vec2(data2);

    CnqsState vec = vec1 + vec2;

    for (unsigned long i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec(i), 100.0);
    }
}

TEST(cnqs_state, vector_subtraction) {
    arma::Col<double> data1(100);
    for (unsigned long i = 0; i < 100; ++i) {
        data1(i) = i;
    }
    CnqsState vec1(data1);

    arma::Col<double> data2(100);
    for (unsigned long i = 0; i < 100; ++i) {
        data2(i) = 100.0 - i;
    }
    CnqsState vec2(data2);

    CnqsState vec = vec1 - vec2;

    for (unsigned long i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec(i), 2.0 * i - 100);
    }
}

TEST(cnqs_state, save_and_load) {
    const double PI = 4.0 * std::atan(1.0);

    arma::Col<double> data(100);
    for (unsigned long i = 0; i < 100; ++i) {
        data(i) = std::cos(0.02 * PI * i);
    }

    CnqsState vec1(data);
    vec1.save("cnqs_state.h5");

    CnqsState vec2;
    vec2.load("cnqs_state.h5");

    for (unsigned long i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec2(i), std::cos(0.02 * PI * i));
    }
}

TEST(cnqs_state, scalar_product) {
    const double PI = 4.0 * std::atan(1.0);

    arma::Col<double> data(100);
    for (unsigned long i = 0; i < 100; ++i) {
        data(i) = std::sin(0.02 * PI * i);
    }

    CnqsState vec1(data);
    CnqsState vec2 = 2.0 * vec1;

    for (unsigned long i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec2(i), 2.0 * std::sin(0.02 * PI * i));
    }
}

TEST(cnqs_state, dot_product) {
    CnqsState vec1(100);
    vec1 = 1.0;

    CnqsState vec2(100);
    vec2 = 2.0;

    ASSERT_DOUBLE_EQ(dot(vec1, vec2), 200.0);
}

TEST(cnqs_state, vector_norm) {
    CnqsState vec(100);
    vec = 4.0;

    ASSERT_DOUBLE_EQ(norm(vec), 40.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
