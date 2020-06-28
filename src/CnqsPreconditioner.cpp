#include "CnqsPreconditioner.hpp"

std::ostream &operator<<(std::ostream &os,
                         const CnqsPreconditioner &cnqs_preconditioner) {
    os << cnqs_preconditioner.Describe() << std::flush;
    return os;
}
