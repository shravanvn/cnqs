#ifndef CNQS_PRECONDITIONER_TRIVIAL_HPP
#define CNQS_PRECONDITIONER_TRIVIAL_HPP

#include "cnqs_preconditioner.hpp"
#include "cnqs_state.hpp"

class CnqsPreconditionerTrivial : public CnqsPreconditioner {
  public:
    CnqsPreconditionerTrivial() = default;

    ~CnqsPreconditionerTrivial() = default;

    const CnqsState solve(const CnqsState &state) const { return state; }

    const CnqsState trans_solve(const CnqsState &state) const { return state; }
};

#endif
