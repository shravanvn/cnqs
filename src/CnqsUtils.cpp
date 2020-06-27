#include "CnqsUtils.hpp"

#include <stdexcept>

int IntPow(int n, int d) {
    if (d < 0) {
        throw std::domain_error(
            "==IntPow== Only intended for non-negative integer exponents");
    }

    if (d == 0) {
        return 1;
    }

    if (d == 1) {
        return n;
    }

    return IntPow(n, d / 2) * IntPow(n, d - d / 2);
}
