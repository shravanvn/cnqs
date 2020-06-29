#ifndef CNQS_BASIC_OPERATOR_HPP
#define CNQS_BASIC_OPERATOR_HPP

#include <string>
#include <tuple>
#include <vector>

#include "CnqsOperator.hpp"

/**
 * @brief CNQS Hamiltonian implemented using simple finite difference
 *
 * This class implements a discretizization of CNQS Hamiltonian given by
 *
 * \f[
 *     (H \psi)(\theta) = -\frac{g J}{2} \sum_{j = 0}^{d - 1} \frac{\partial^2
 *     \psi}{\partial \theta_j^2}(\theta) - J \sum_{(j, k) \in E} \cos(\theta_j
 *     - \theta_k) \psi(\theta)
 * \f]
 *
 * using a five-point finite different stencil to approximate the second
 * derivatives. For an one dimensional function \f$f(x)\f$ this stencil is given
 * by
 *
 * \f[
 *     f''(x) \approx \frac{-f(x + 2h) + 16 f(x + h) - 30 f(x) + 16
 *     f(x - h) - f(x - 2h)}{12 h^2}
 * \f]
 *
 * and "wrap-around" is used to compute the derivative at boundary points.
 *
 * The corresponding CnqsVector object should be a discretization of a
 * \f$d\f$-dimensional function \f$\psi\f$ defined on \f$[0, 2\pi]^d\f$ and
 * \f$2\pi\f$-periodic in each of the dimensions. Using \f$n\f$ grid points per
 * dimension to discretize this function will lead to a \f$d\f$ dimensional
 * tensor \f$v(i_0, i_1, \ldots, i_{d - 1})\f$ of size \f$n \times n \times
 * \cdots \times n\f$ with
 *
 * \f[
 *     v(i_0, i_1, \ldots, i_{d - 1}) \approx \psi(i_0 h, i_1 h, \ldots,
 *     i_{d - 1} h), \quad h = \frac{2 \pi}{n}
 * \f]
 *
 * The CnqsVector object representing \f$v\f$ is assumed to used column-major
 * indexing scheme: the \f$d\f$-dimensional index \f$(i_0, i_1, \ldots,
 * i_{d - 1})\f$ corresponds to linear index
 *
 * \f[
 *     i = i_0 + n i_1 + \cdots + n^{d - 1} i_d
 * \f]
 */
class CnqsBasicOperator : public CnqsOperator {
public:
    /**
     * @brief Construct a new CnqsBasicOperator object
     *
     * @param [in] num_rotor Number of quantum rotors \f$d\f$. At least two
     * rotors is needed to construct the operator
     *
     * @param [in] num_grid_point Number of grid points per dimension \f$n\f$.
     * At least five grid points is required to construct the operator
     *
     * @param [in] edges Edge set specifying connections between the quantum
     * rotors \f$E\f$. This is a `std::vector` of `std::tuple<int, int>`; each
     * tuple contains the the node numbers the edge connects. The node
     * identifiers should be in the set \f$\{0, 1, \ldots, d - 1\f$}
     *
     * @param [in] g Parameter \f$g\f$
     *
     * @param [in] J Parameter \f$J\f$. To ensure a finite lower bound of for
     * the spectram of the Hamiltonian, the product \f$g J\f$ is required to be
     * positive
     *
     * @throw std::domain_error If the input arguments are illegal as described
     * above
     */
    CnqsBasicOperator(int num_rotor, int num_grid_point,
                      const std::vector<std::tuple<int, int>> &edges, double g,
                      double J);

    /**
     * @brief Default destructor
     *
     */
    ~CnqsBasicOperator() = default;

    /**
     * @brief Test if operator can be applied on a CnqsVector
     *
     * Checks if \f$H v\f$ can be computed by ensuring the number of elements in
     * the \f$v\f$ is \f$n^d\f$.
     *
     * @param [in] cnqs_vector CnqsVector vector \f$v\f$
     *
     * @throw std::length_error If \f$H v\f$ cannot be computed
     */
    void TestCompatibility(const CnqsVector &cnqs_vector) const;

    /**
     * @brief Set up the initial condition
     *
     * Sets the CnqsVector to the discretization of smallest energy eigenstate
     * of the Laplacian
     *
     * \f[
     *     v(i_0, i_1, \ldots, i_{d - 1}) = \cos(i_0 h) \cos(i_1 h) \cdots
     *     \cos(i_{d - 1} h), \quad h = \frac{2\pi}{n}
     * \f]
     *
     * @param [out] cnqs_vector Will be set to \f$v\f$ defined as above
     */
    void ConstructInitialState(CnqsVector &cnqs_vector) const;

    /**
     * @brief Apply the operator to a CnqsVector
     *
     * Compute \f$w = H v\f$ using the finite difference stencil on the uniform
     * grid.
     *
     * @param [in] input_vector CnqsVector \f$v\f$
     * @param [out] output_vector CnqsVector \f$w\f$
     */
    void Apply(const CnqsVector &input_vector, CnqsVector &output_vector) const;

    /**
     * @brief Lower bound on the smallest eigenvalue of the Hamiltonian
     *
     * As long as \f$g J > 0\f$, it can be shown that \f$\lambda_\text{min} \geq
     * \mu = -J | E |\f$.
     *
     * @return Estimate \f$\mu\f$
     */
    double EigValLowerBound() const { return -J_ * edges_.size(); }

    /**
     * @brief Create a string representation of the CnqsBasicOperator object
     *
     * @return `std::string` with description
     */
    std::string Describe() const;

private:
    int num_rotor_;
    int num_grid_point_;
    std::vector<std::tuple<int, int>> edges_;
    double g_;
    double J_;
    std::vector<int> unfolding_factor_;
    std::vector<double> theta_;
};

#endif
