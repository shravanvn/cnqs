#ifndef CNQS_OPERATOR_HPP
#define CNQS_OPERATOR_HPP

#include <ostream>
#include <string>

#include "CnqsVector.hpp"

/**
 * @brief Abstract class implementing Hermitian operators on CnqsVector objects
 *
 * This class lays out the framework for implementing a Hermitian operator
 * \f$H\f$ on CnqsVector object \f$v\f$.
 */
class CnqsOperator {
public:
    /**
     * @brief Default destructor
     *
     */
    virtual ~CnqsOperator() = default;

    /**
     * @brief Test if operator can be applied on a CnqsVector
     *
     * Checks if \f$H v\f$ can be computed by ensuring the dimensionalities of
     * \f$H\f$ and \f$v\f$ are compatible.
     *
     * @param [in] cnqs_vector CnqsVector vector \f$v\f$
     *
     * @throw std::length_error If \f$H v\f$ cannot be computed
     *
     * @attention This is a pure virtual member function that must be overloaded
     * by concrete subclasses of CnqsOperator.
     */
    virtual void TestCompatibility(const CnqsVector &cnqs_vector) const = 0;

    /**
     * @brief Set up the initial condition
     *
     * Sets the CnqsVector some appropriate initial state \f$v\f$ (e.g. in
     * implementing inverse power iteration, this could be the eigenstate of
     * some part \f$L\f$ of the full operator \f$H\f$ corresponding to the
     * smallest energy state).
     *
     * @param [out] cnqs_vector Will be set to \f$v\f$ defined as above
     *
     * @attention This is a pure virtual member function that must be overloaded
     * by concrete subclasses of CnqsOperator.
     */
    virtual void ConstructInitialState(CnqsVector &cnqs_vector) const = 0;

    /**
     * @brief Apply the operator to a CnqsVector
     *
     * Compute \f$w = H v\f$.
     *
     * @param [in] input_vector CnqsVector \f$v\f$
     * @param [out] output_vector CnqsVector \f$w\f$
     *
     * @attention This is a pure virtual member function that must be overloaded
     * by concrete subclasses of CnqsOperator.
     */

    virtual void Apply(const CnqsVector &input_vector,
                       CnqsVector &output_vector) const = 0;

    /**
     * @brief Apply a shifted version of the operator to a CnqsVector
     *
     * Compute \f$w = (H - \mu I) v\f$ where \f$I\f$ is the identity operator.
     * This is implemented in terms of the Apply() member function.
     *
     * @param input_vector CnqsVector \f$v\f$
     * @param shift Shift \f$\mu\f$
     * @param output_vector CnqsVector \f$w\f$
     */
    void ShiftedApply(const CnqsVector &input_vector, double shift,
                      CnqsVector &output_vector) const;

    /**
     * @brief Lower bound on the smallest eigenvalue of the Hamiltonian
     *
     * Returns an estimate \f$\mu\f$ such that \f$\lambda_\text{min} \geq \mu\f$
     * for the smallest eigenvalue of \f$H\f$.
     *
     * @return Estimate \f$\mu\f$
     *
     * @attention This is a pure virtual member function that must be overloaded
     * by concrete subclasses of CnqsOperator.
     */

    virtual double EigValLowerBound() const = 0;

    /**
     * @brief Create a string representation of the CnqsOperator object
     *
     * @return `std::string` with description
     *
     * @attention This is a pure virtual member function that must be overloaded
     * by concrete subclasses of CnqsOperator.
     */
    virtual std::string Describe() const = 0;
};

/**
 * @brief Print CnqsOperator objects to output streams (e.g. `std::cout`)
 *
 * This implementation uses the Describe() method of the CnqsOperator class.
 *
 */
std::ostream &operator<<(std::ostream &os, const CnqsOperator &cnqs_operator);

#endif
