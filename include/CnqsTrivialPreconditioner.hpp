#ifndef CNQS_TRIVIAL_PRECONDITIONER_HPP
#define CNQS_TRIVIAL_PRECONDITIONER_HPP

#include <string>

#include "CnqsPreconditioner.hpp"
#include "CnqsVector.hpp"

/**
 * @brief Trivial preconditioner for CnqsOperator objects
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
     * @param vector CnqsVector object that will be operated on
     */
    void TestCompatibility(const CnqsVector &vector) const;

    /**
     * @brief Apply the preconditioner to a CnqsVector
     *
     * This function simply copies the input vector to the output
     *
     * @param input_vector Input Vector
     * @param output_vector Output Vector
     */
    void Solve(const CnqsVector &input_vector, CnqsVector &output_vector) const;

    /**
     * @brief Create a string representation of the CnqsTrivialPreconditioner
     * object
     *
     * @return C++ standard string containing the description
     */
    std::string Describe() const;
};

#endif
