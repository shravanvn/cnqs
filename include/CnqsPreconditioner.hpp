#ifndef CNQS_PRECONDITIONER_HPP
#define CNQS_PRECONDITIONER_HPP

#include <iostream>
#include <string>

#include "CnqsVector.hpp"

class CnqsPreconditioner {
public:
    CnqsPreconditioner(const std::string &name) : name_(name) {}

    virtual ~CnqsPreconditioner() = default;

    virtual void Solve(const CnqsVector &input_state,
                       CnqsVector &output_state) const = 0;

    friend std::ostream &operator<<(std::ostream &os,
                                    const CnqsPreconditioner &preconditioner);

private:
    std::string name_;
};

#endif
