#ifndef CNQS_FOURIER_OPERATOR_HPP
#define CNQS_FOURIER_OPERATOR_HPP

#include <tuple>
#include <vector>

#include "CnqsOperator.hpp"

/**
 * @brief Hamiltonian class implemented with Fourier series
 *
 * This class approximates Hamiltonian in the frequency domain. The operator is
 * given by
 * \f[
 * \widehat{H} \widehat{\psi}(k) = \frac{g J}{2} \| k \|^2 \widehat{\psi}(k) -
 * \frac{J}{2} \sum_{(j, k) \in E} [\widehat{\psi}(k + e_j - e_k) +
 * \widehat{\psi}(k - e_j + e_k)]
 * \f]
 * The CnqsVector object should therefore be defined on a \f$d\f$-dimensional
 * integer lattice of size \f$n \times n \times \cdots \times n\f$. The \f$(i_0,
 * i_1, \ldots, i_{d - 1})\f$-th entry corresponds to the fourier coefficeint
 * \f$\widehat{\psi}(k_0, k_1, \ldots, k_{d - 1})\f$ with \f$k_j = i_j -
 * k_\text{max}\f$ and \f$n = 2 k_\text{max} + 1\f$.
 */
class CnqsFourierOperator : public CnqsOperator {
public:
    /**
     * @brief Construct a new CnqsFourierOperator object
     *
     * @param d Dimensionality
     * @param n Discretization parameter (\f$2 k_\text{max} + 1\f$) where
     * \f$k_\text{max}\f$ is the maximum value of the wavenumber
     * @param edges Connections between the nodes/edges
     * @param g Parameter
     * @param J Parameter
     */
    CnqsFourierOperator(int d, int n,
                        const std::vector<std::tuple<int, int>> &edges,
                        double g, double J);

    /**
     * @brief Destroy the CnqsFourierOperator object
     *
     */
    ~CnqsFourierOperator() = default;

    /**
     * @brief Set up the initial condition
     *
     * Set up the initial vector to the smallest eigenstate of the Laplacian
     * \f[
     * \widehat{\psi}(k_0, k_1, \ldots, k_{d - 1}) =
     * \begin{cases}
     * \pi^d & \text{if} \quad |k_0| = |k_1| = \cdots = |k_{d - 1}| = 1 \\
     * 0 & \text{otherwise}
     * \end{cases}
     * \f]
     *
     * @param cnqs_vector Initial CnqsVector
     */
    void ConstructInitialState(CnqsVector &cnqs_vector) const;

    /**
     * @brief Apply the CnqsFourierOperator to CnqsVector
     *
     * The ouput is computed in the frequency domain.
     *
     * @param input_vector Input vector
     * @param output_vector Output vector
     */
    void Apply(const CnqsVector &input_vector, CnqsVector &output_vector) const;

private:
    bool IndexQualifiesForInitialState(int i) const;

    double SquaredDistanceFromCenter(int i) const;

    int max_freq_;
};

#endif
