#ifndef CNQS_TRIVIAL_PRECONDITIONER_HPP
#define CNQS_TRIVIAL_PRECONDITIONER_HPP

#include "cnqs_state.hpp"

class CnqsTrivialPreconditioner {
  public:
    CnqsTrivialPreconditioner() = default;

    ~CnqsTrivialPreconditioner() = default;

    const CnqsState &solve(const CnqsState &state) const { return state; }

    const CnqsState &trans_solve(const CnqsState &state) const { return state; }
};

#endif
