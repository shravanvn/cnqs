#ifndef CNQS_PRECONDITIONER_HPP
#define CNQS_PRECONDITIONER_HPP

#include <iostream>
#include <string>

#include "CnqsVector.hpp"

/**
 * @brief Preconditioner pure abstract class for CnqsOperator objects
 *
 * This class lays out the framework for implementing preconditioners
 * corresponding to CnqsOperator objects.
 */
class CnqsPreconditioner {
public:
    /**
     * @brief Default destructor
     *
     */
    virtual ~CnqsPreconditioner() = default;

    /**
     * @brief Test if preconditioner can be applied on a CnqsVector
     *
     * @attention This is a pure virtual member function that will be overloaded
     * by concrete subclasses of CnqsPreconditioner.
     *
     * @param vector CnqsVector object that will be operated on
     */
    virtual void TestCompatibility(const CnqsVector &vector) const = 0;

    /**
     * @brief Apply the preconditioner to a CnqsVector
     *
     * @attention This is a pure virtual member function that will be overloaded
     * by concrete subclasses of CnqsPreconditioner.
     *
     * @param input_vector Input Vector
     * @param output_vector Output Vector
     */
    virtual void Solve(const CnqsVector &input_vector,
                       CnqsVector &output_vector) const = 0;

    /**
     * @brief Create a string representation of the CnqsPreconditioner object
     *
     * @attention This is a pure virtual member function that will be overloaded
     * by concrete subclasses of CnqsPreconditioner.
     *
     * @return C++ standard string containing the description
     */
    virtual std::string Describe() const = 0;
};

/**
 * @brief Print CnqsPreconditioner objects to output streams (e.g. `std::cout`)
 *
 */
std::ostream &operator<<(std::ostream &os,
                         const CnqsPreconditioner &cnqs_preconditioner);

#endif
