#include "CnqsPreconditioner.hpp"

std::ostream &operator<<(std::ostream &os,
                         const CnqsPreconditioner &cnqs_preconditioner) {
    std::string description = "";
    cnqs_preconditioner.Describe(description);
    os << description << std::flush;

    return os;
}
