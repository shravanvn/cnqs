#ifndef CNQS_BASIC_OPERATOR_HPP
#define CNQS_BASIC_OPERATOR_HPP

#include <tuple>
#include <vector>

#include "CnqsOperator.hpp"

/**
 * @brief Hamiltonian class implemented with simple finite difference
 *
 * This class approximates the differential part of the Hamiltonian in
 * CnqsOperator using a five-point stencil:
 * \f[
 * f''(x) \approx \frac{-f(x + 2h) + 16 f(x + h) - 30 f(x) + 16 f(x - h) - f(x -
 * 2h)}{12 h^2}
 * \f]
 * The CnqsVector object should therefore be a discretization of the function
 * \f$\psi\f$ on a \f$d\f$-dimensional \f$n \times n \times \cdots \times n\f$
 * uniform grid where the \f$(i_0, i_1, \ldots, i_{d - 1})\f$-th grid point
 * corresponds to \f$(i_0 h, i_1 h, \ldots, i_{d - 1} h) \in [0, 2\pi]^d\f$
 * with \f$h = 2 \pi / n\f$.
 */
class CnqsBasicOperator : public CnqsOperator {
public:
    /**
     * @brief Construct a new CnqsBasicOperator object
     *
     * @param d Dimensionality
     * @param n Number of grid points per dimension
     * @param edges Connection between the rotors/nodes
     * @param g Parameter
     * @param J Parameter
     */
    CnqsBasicOperator(int d, int n,
                      const std::vector<std::tuple<int, int>> &edges, double g,
                      double J);

    /**
     * @brief Default destructor
     *
     */
    ~CnqsBasicOperator() = default;

    /**
     * @brief Set up the initial condition
     *
     * Sets the vector to the discretization of smallest energy eigenstate of
     * the Laplacian
     * \f[
     * \psi(\theta_0, \theta_1, \dots, \theta_{d - 1}) = \cos(\theta_0)
     * \cos(\theta_1) \cdots \cos(\theta_{d - 1})
     * \f]
     * on the uniform grid
     *
     * @param cnqs_vector Initial CnqsVector
     */
    void ConstructInitialState(CnqsVector &cnqs_vector) const;

    void Apply(const CnqsVector &input_vector, CnqsVector &output_vector) const;

private:
    std::vector<double> theta_;
};

#endif
