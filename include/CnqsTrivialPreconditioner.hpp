#ifndef CNQS_TRIVIAL_PRECONDITIONER_HPP
#define CNQS_TRIVIAL_PRECONDITIONER_HPP

#include <string>

#include "CnqsPreconditioner.hpp"
#include "CnqsVector.hpp"

/**
 * @brief Trivial preconditioner for CnqsVector objects
 *
 * A CnqsTrivialPreconditioner \f$M\f$ acts on a CnqsVector \f$v\f$ as \f$M^{-1}
 * v = v\f$.
 *
 */
class CnqsTrivialPreconditioner : public CnqsPreconditioner {
public:
    /**
     * @brief Default constructor
     *
     */
    CnqsTrivialPreconditioner() = default;

    /**
     * @brief Default destructor
     *
     */
    ~CnqsTrivialPreconditioner() = default;

    /**
     * @brief Test if preconditioner can be applied on a CnqsVector
     *
     * This function does nothing - all CnqsVector objects \f$v\f$ are
     * compatible with CnqsTrivialPreconditioner objects.
     *
     * @param [in] cnqs_vector CnqsVector vector \f$v\f$
     */
    void TestCompatibility(const CnqsVector &vector) const;

    /**
     * @brief Apply the preconditioner to a CnqsVector
     *
     * This function simply copies the input \f$v\f$ to the output \f$w\f$
     *
     * @param input_vector CnqsVector \f$v\f$
     * @param output_vector CnqsVector \f$w\f$
     */
    void Solve(const CnqsVector &input_vector, CnqsVector &output_vector) const;

    /**
     * @brief Create a string representation of the CnqsTrivialPreconditioner
     * object
     *
     * @return `std::string` with description
     */
    std::string Describe() const;
};

#endif
