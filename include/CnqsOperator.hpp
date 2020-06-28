#ifndef CNQS_OPERATOR_HPP
#define CNQS_OPERATOR_HPP

#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "CnqsVector.hpp"

/**
 * @brief Hamiltonian abstract class for CnqsVector objects
 *
 * This class lays out the framework for implementing the discretized
 * application of the Hamiltonian operator
 * \f[
 * H = -\frac{g J}{2} \sum_{j = 0}^{d - 1} \frac{\partial^2}{\partial
 * \theta_j^2} - J \sum_{(j, k) \in E} \cos(\theta_j - \theta_k)
 * \f]
 * on a function \f$\psi \in H^1([0, 2\pi]^d)\f$ that is \f$2\pi\f$-periodic in
 * each dimension. Here \f$E\f$ is the edge set of some graph with vertices
 * labeled \f$\{0, 1, 2, \ldots, d - 1\}\f$. The parameters \f$g\f$ and \f$J\f$
 * must satisfy \f$g J > 0\f$ to establish a lower bound \f$\lambda_\text{min}
 * \geq - J | E |\f$ on the minimal eigenvalue of this operator.
 */
class CnqsOperator {
public:
    /**
     * @brief Construct a new CnqsOperator object
     *
     * @param d Dimensionality (equivalently, number of quantum rotors). Must be
     * at least 2.
     * @param n Discretizing parameter per dimension. Must be at least 2.
     * @param edges Edge set specifying the connectivity between rotors. Each
     * edge is identified by a tuple \f$(j, k)\f$ of the boundary nodes, with
     * \f$0 \leq j, k \leq d - 1\f$.
     * @param g Parameter
     * @param J Parameter. Note that \f$gJ\f$ must be positive.
     */
    CnqsOperator(int d, int n, const std::vector<std::tuple<int, int>> &edges,
                 double g, double J);

    /**
     * @brief Default destructor
     *
     */
    virtual ~CnqsOperator() = default;

    /**
     * @brief Test if operator can be applied on a CnqsVector
     *
     * Running `cnqs_operator.TestCompatibility(cnqs_vector)` will raise an
     * exception if the `cnqs_operator` cannot act on the `cnqs_vector`.
     *
     * @param cnqs_vector CnqsVector object that will be operated on
     */
    void TestCompatibility(const CnqsVector &cnqs_vector) const {
        if (cnqs_vector.Size() != num_element_[d_]) {
            throw std::length_error(
                "==CnqsOperator== Vector length is not compatible with "
                "operator");
        }
    }

    /**
     * @brief Construct initial CnqsVector object corresponding to the operator
     *
     * @attention This is a pure virtual member function that must be overloaded
     * by concrete subclasses of CnqsOperator.
     *
     * @param cnqs_vector The CnqsVector object to be updated with new values
     */
    virtual void ConstructInitialState(CnqsVector &cnqs_vector) const = 0;

    /**
     * @brief Apply the operator to a CnqsVector
     *
     * @attention This is a pure virtual member function that must be overloaded
     * by concrete subclasses of CnqsOperator.
     *
     * @param input_vector Input vector
     * @param output_vector Output vector
     */
    virtual void Apply(const CnqsVector &input_vector,
                       CnqsVector &output_vector) const = 0;

    /**
     * @brief Lower bound on the smallest eigenvalue of the Hamiltonian
     *
     * As long as \f$gJ > 0\f$ this can be derived to be \f$-J | E |\f$.
     *
     * @return Lower bound
     */
    double EigValLowerBound() const { return -J_ * edges_.size(); }

    /**
     * @brief Apply a shifted version of the operator to a CnqsVector
     *
     * Applies the operator \f$H - \mu I\f$ for constant \f$mu\f$
     *
     * @param input_vector Input vector
     * @param shift Shift \f$\mu\f$
     * @param output_vector Output vector
     */
    void ShiftedApply(const CnqsVector &input_vector, double shift,
                      CnqsVector &output_vector) const;

    /**
     * @brief Create a string representation of the CnqsOperator object
     *
     * @attention This is a pure virtual member function that must be overloaded
     * by concrete subclasses of CnqsOperator.
     *
     * @return C++ standard string with description
     */
    virtual std::string Describe() const = 0;

protected:
    /**
     * @brief Dimensionality/Number of rotors
     *
     */
    int d_;

    /**
     * @brief Discretization parameter per dimension
     *
     */
    int n_;

    /**
     * @brief Edge set
     *
     * Each edge \f$(j, k)\f$ is identified by the nodes it connects. The node
     * numbers are in the set \f$\{0, 1, 2, \ldots, d - 1\}\f$.
     */
    std::vector<std::tuple<int, int>> edges_;

    /**
     * @brief Parameter
     *
     */
    double g_;

    /**
     * @brief Parameter
     *
     */
    double J_;

    /**
     * @brief Sizes of different unfoldings of the \f$d\f$-dimensional operator
     *
     * This is a array \f$s\f$ of length \f$d + 1\f$ with the \f$j\f$-th entry
     * \f$n^j\f$ for \f$j = 0, 1, \ldots, d - 1\f$. This can be used to
     * unroll column-major indices of a \f$d\f$-dimensional \f$n \times n \times
     * \cdots \times n\f$ vector:
     * \f[
     * (i_0, i_1, \ldots, i_{d - 1}) \mapsto s(0) \cdot i_0 + s(1) \cdot i_1 +
     * \cdots + s(d - 1) \cdot i_{d - 1}
     * \f]
     * and
     * \f[
     * (i_{<j}, i_j, i_{>j}) \mapsto s(0) \cdot i_{<j} + s(j) \cdot i_j + s(j +
     * 1) \cdot i_{>j}
     * \f]
     */
    std::vector<int> num_element_;
};

/**
 * @brief Print CnqsOperator objects to output streams (e.g. `std::cout`)
 *
 */
std::ostream &operator<<(std::ostream &os, const CnqsOperator &cnqs_operator);

#endif
