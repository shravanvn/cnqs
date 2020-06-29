#ifndef CNQS_PRECONDITIONER_HPP
#define CNQS_PRECONDITIONER_HPP

#include <iostream>
#include <string>

#include "CnqsVector.hpp"

/**
 * @brief Pure abstract class implementing Hermitian preconditioners on
 * CnqsVector objects
 *
 * This class lays out the framework for implementing a preconditioner \f$M\f$
 * on CnqsVector \f$v\f$. These preconditions are usually associated with a
 * CnqsOperator \f$H\f$.
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
     * Checks if \f$M^{-1} v\f$ can be computed by ensuring the dimensionalities
     * of \f$M\f$ and \f$v\f$ are compatible.
     *
     * @param [in] cnqs_vector CnqsVector vector \f$v\f$
     *
     * @throw std::length_error If \f$M^{-1} v\f$ cannot be computed
     *
     * @attention This is a pure virtual member function that must be overloaded
     * by concrete subclasses of CnqsPreconditioner.
     */
    virtual void TestCompatibility(const CnqsVector &vector) const = 0;

    /**
     * @brief Apply the preconditioner to a CnqsVector
     *
     * Compute \f$w = M^{-1} v\f$.
     *
     * @param input_vector CnqsVector \f$v\f$
     * @param output_vector CnqsVector \f$w\f$
     *
     * @attention This is a pure virtual member function that must be overloaded
     * by concrete subclasses of CnqsPreconditioner.
     */
    virtual void Solve(const CnqsVector &input_vector,
                       CnqsVector &output_vector) const = 0;

    /**
     * @brief Create a string representation of the CnqsPreconditioner object
     *
     * @return `std::string` with description
     *
     * @attention This is a pure virtual member function that must be overloaded
     * by concrete subclasses of CnqsPreconditioner.
     */
    virtual std::string Describe() const = 0;
};

/**
 * @brief Print CnqsPreconditioner objects to output streams (e.g. `std::cout`)
 *
 * This implementation uses the Describe() method of the CnqsPreconditioner
 * class.
 */
std::ostream &operator<<(std::ostream &os,
                         const CnqsPreconditioner &cnqs_preconditioner);

#endif
