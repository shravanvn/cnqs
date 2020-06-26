#include "gtest/gtest.h"

#include <cmath>
#include <iostream>
#include <vector>

#include "CnqsVector.hpp"

static const double PI = 4.0 * std::atan(1.0);

TEST(CnqsVector, ConstructFromSize) {
    CnqsVector vec(100);

    ASSERT_EQ(vec.Size(), 100);

    for (int i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec(i), 0.0);
    }
}

TEST(CnqsVector, ConstructFromData) {
    std::vector<double> data(100);
    for (int i = 0; i < 100; ++i) {
        data[i] = i;
    }

    CnqsVector vec(data);

    ASSERT_EQ(vec.Size(), 100);

    for (int i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec(i), i);
    }
}

TEST(CnqsVector, ScalarAssignment) {
    CnqsVector vec(100);

    vec = 1.0;

    for (int i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec(i), 1.0);
    }
}

TEST(CnqsVector, VectorAdditionInPlace) {
    std::vector<double> data1(100);
    std::vector<double> data2(100);

    for (int i = 0; i < 100; ++i) {
        data1[i] = i;
        data2[i] = 100 - i;
    }

    CnqsVector vec1(data1);
    CnqsVector vec2(data2);

    vec1 += vec2;

    for (int i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec1(i), 100.0);
    }
}

TEST(CnqsVector, VectorSubtractionInPlace) {
    std::vector<double> data1(100);
    std::vector<double> data2(100);

    for (int i = 0; i < 100; ++i) {
        data1[i] = 100 - i;
        data2[i] = i;
    }

    CnqsVector vec1(data1);
    CnqsVector vec2(data2);

    vec1 -= vec2;

    for (int i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec1(i), 100.0 - 2.0 * i);
    }
}

TEST(CnqsVector, ScalarMultiplicationInPlace) {
    std::vector<double> data(100);

    for (int i = 0; i < 100; ++i) {
        data[i] = i * (100.0 - i);
    }

    CnqsVector vec(data);

    vec *= 2.0;

    for (int i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec(i), 2.0 * i * (100.0 - i));
    }
}

TEST(CnqsVector, ScalarDivisionInPlace) {
    std::vector<double> data(100);

    for (int i = 0; i < 100; ++i) {
        data[i] = i * (100.0 - i);
    }

    CnqsVector vec(data);

    vec /= 3.0;

    for (int i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec(i), i * (100.0 - i) / 3.0);
    }
}

TEST(CnqsVector, VectorAddition) {
    std::vector<double> data1(100);
    std::vector<double> data2(100);

    for (int i = 0; i < 100; ++i) {
        data1[i] = i;
        data2[i] = 100.0 - i;
    }

    CnqsVector vec1(data1);
    CnqsVector vec2(data2);

    CnqsVector vec = vec1 + vec2;

    for (int i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec(i), 100.0);
    }
}

TEST(CnqsVector, VectorSubtraction) {
    std::vector<double> data1(100);
    std::vector<double> data2(100);

    for (int i = 0; i < 100; ++i) {
        data1[i] = i;
        data2[i] = 100.0 - i;
    }

    CnqsVector vec1(data1);
    CnqsVector vec2(data2);

    CnqsVector vec = vec1 - vec2;

    for (int i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec(i), 2.0 * i - 100);
    }
}

TEST(CnqsVector, ScalarMultiplicationOnRight) {
    std::vector<double> data1(100);

    for (int i = 0; i < 100; ++i) {
        data1[i] = i;
    }

    CnqsVector vec1(data1);

    CnqsVector vec = vec1 * 2.0;

    for (int i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec(i), 2.0 * i);
    }
}

TEST(CnqsVector, ScalarDivisionOnRight) {
    std::vector<double> data1(100);

    for (int i = 0; i < 100; ++i) {
        data1[i] = i;
    }

    CnqsVector vec1(data1);

    CnqsVector vec = vec1 / 2.0;

    for (int i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec(i), 0.5 * i);
    }
}

TEST(CnqsVector, ScalarMultiplicationOnLeft) {
    std::vector<double> data1(100);

    for (int i = 0; i < 100; ++i) {
        data1[i] = i;
    }

    CnqsVector vec1(data1);

    CnqsVector vec = 3.0 * vec1;

    for (int i = 0; i < 100; ++i) {
        ASSERT_DOUBLE_EQ(vec(i), 3.0 * i);
    }
}

TEST(CnqsVector, Save) {
    std::vector<double> data(100);

    for (int i = 0; i < 100; ++i) {
        data[i] = std::cos(0.02 * PI * i);
    }

    CnqsVector vec(data);

    vec.Save("cnqs_vector.txt");
}

TEST(CnqsVector, Dot) {
    CnqsVector vec1(100);
    CnqsVector vec2(100);

    vec1 = 1.0;
    vec2 = 2.0;

    ASSERT_DOUBLE_EQ(vec1.Dot(vec2), 200.0);
}

TEST(CnqsVector, Norm) {
    CnqsVector vec(100);

    vec = 4.0;

    ASSERT_DOUBLE_EQ(vec.Norm(), 40.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
