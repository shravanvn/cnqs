#include "CnqsPreconditioner.hpp"

std::ostream &operator<<(std::ostream &os,
                         const CnqsPreconditioner &preconditioner) {
    os << "==CnqsPreconditioner==" << std::endl
       << "    name : " << preconditioner.name_ << std::endl
       << "==CnqsPreconditioner==" << std::flush;

    return os;
}
