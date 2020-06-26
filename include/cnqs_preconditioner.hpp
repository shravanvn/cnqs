#ifndef CNQS_PRECONDITIONER_HPP
#define CNQS_PRECONDITIONER_HPP

#include "cnqs_state.hpp"

class CnqsPreconditioner {
  public:
    CnqsPreconditioner() = default;

    virtual ~CnqsPreconditioner() = default;

    virtual const CnqsState solve(const CnqsState &state) const = 0;

    virtual const CnqsState trans_solve(const CnqsState &state) const = 0;
};

#endif
