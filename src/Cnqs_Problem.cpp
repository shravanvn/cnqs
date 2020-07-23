#include "Cnqs_Problem.hpp"

std::ostream &operator<<(std::ostream &os, const Cnqs::Problem &problem) {
    os << problem.description();
    return os;
}
