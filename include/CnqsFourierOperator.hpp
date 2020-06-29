#ifndef CNQS_FOURIER_OPERATOR_HPP
#define CNQS_FOURIER_OPERATOR_HPP

#include <string>
#include <tuple>
#include <vector>

#include "CnqsOperator.hpp"

/**
 * @brief CNQS Hamiltonian implemented in the Fourier domain
 *
 * This class implements a trucation of the CNQS Hamiltonian in the frequency
 * domain. It is given by
 *
 * \f[
 *     (\hat{H} \hat{\psi})(k) = \frac{g J}{2} \| k \|^2 \hat{\psi}(k)
 *     - \frac{J}{2} \sum_{(j_1, j_2) \in E} [\hat{\psi}(k + e_{j_1} - e_{j_2})
 *     + \hat{\psi}(k - e_{j_1} + e_{j_2})]
 * \f]
 *
 * where \f$e_j\f$ represents the \f$j\f$-th standard basis vector in \f$d\f$
 * dimensions. If \f$k = (k_0, k_1, \ldots, k_{d - 1})\f$ then assuming
 * \f$j_1 < j_2\f$ we have
 *
 * \f[
 *     k + e_{j_1} - e_{j_2} = (k_0, k_1, \ldots, k_{j_1 - 1}, k_{j_1} + 1,
 *     k_{j_1 + 1}, \ldots, k_{j_2 - 1}, k_{j_2} - 1, k_{j_2 + 1}, \ldots,
 *     k_{d - 1})
 * \f]
 *
 * The corresponding CnqsVector object should be a truncation of the Fouier
 * coefficient lattice \f$\hat{\psi}(k)\f$ on \f$k \in \mathbb{Z}^d\f$. Only a
 * basic truncation of
 *
 * \f[
 *     -k_\text{max} \leq k_j \leq k_\text{max}, \quad 0 \leq j \leq d - 1
 * \f]
 *
 * is supported; all the Fourier coefficients outside this hypercube is presumed
 * to be zeros. This leads to a \f$d\f$ dimensional tensor tensor \f$v(i_0, i_1,
 * \ldots, i_d)\f$ of size \f$(2 k_\text{max} + 1) \times (2 k_\text{max} + 1)
 * \times \cdots \times (2 k_\text{max} + 1)\f$ with
 *
 * \f[
 *     v(i_0, i_1, \ldots, i_{d - 1}) \approx \hat{\psi}(i_0 - k_\text{max},
 *     i_1 - k_\text{max}, \ldots, i_{d - 1} - k_\text{max})
 * \f]
 *
 * The CnqsVector object representing \f$v\f$ is assumed to used column-major
 * indexing scheme: the \f$d\f$-dimensional index \f$(i_0, i_1, \ldots,
 * i_{d - 1})\f$ corresponds to linear index
 *
 * \f[
 *     i = i_0 + n i_1 + \cdots + n^{d - 1} i_{d - 1}
 * \f]
 */
class CnqsFourierOperator : public CnqsOperator {
public:
    /**
     * @brief Construct a new CnqsFourierOperator object
     *
     * @param [in] num_rotor Number of quantum rotors \f$d\f$. At least two
     * rotors is needed to construct the operator
     *
     * @param [in] max_freq Frequency cut-off parameter \f$k_\text{max}\f$. This
     * maximum frequency should be at least one.
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
    CnqsFourierOperator(int num_rotor, int max_freq,
                        const std::vector<std::tuple<int, int>> &edges,
                        double g, double J);

    /**
     * @brief Default destructor
     *
     */
    ~CnqsFourierOperator() = default;

    /**
     * @brief Test if operator can be applied on a CnqsVector
     *
     * Checks if \f$\hat{H} v\f$ can be computed by ensuring the number of
     * elements in the \f$v\f$ is \f$(2 k_\text{max} + 1)^d\f$.
     *
     * @param [in] cnqs_vector CnqsVector vector \f$v\f$
     *
     * @throw std::length_error If \f$H v\f$ cannot be computed
     */
    void TestCompatibility(const CnqsVector &cnqs_vector) const;

    /**
     * @brief Set up the initial condition
     *
     * Sets the CnqsVector to the smallest energy eigenstate of the Laplacian
     *
     * \f[
     *     v(i_0, i_1, \ldots, i_{d - 1}) =
     *     \begin{cases}
     *         \pi^d & \text{if} \quad |i_0 - k_\text{max}| = |i_1 -
     *         k_\text{max}| = \cdots = |i_{d - 1} - k_\text{max}| = 1 \\
     *         0 & \text{otherwise}
     *     \end{cases}
     * \f]
     *
     * Note that \f$k_j = i_j - k_\text{max}\f$, so the above condition boils
     * down to setting only the Fourier coefficients corresponding to the \f$k =
     * (\pm 1, \pm 1, \ldots, \pm 1)\f$ frequencies to \f$\pi^d\f$.
     *
     * @param [out] cnqs_vector Will be set to \f$v\f$ defined as above
     */
    void ConstructInitialState(CnqsVector &cnqs_vector) const;

    /**
     * @brief Apply the operator to a CnqsVector
     *
     * Compute \f$w = \hat{H} v\f$ using on the truncated integer lattice.
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
     * @brief Create a string representation of the CnqsFourierOperator object
     *
     * @return `std::string` with description
     */
    std::string Describe() const;

private:
    bool IndexQualifiesForInitialState(int i) const;

    double SquaredDistanceFromCenter(int i) const;

    int num_rotor_;
    int max_freq_;
    std::vector<std::tuple<int, int>> edges_;
    double g_;
    double J_;
    int num_tot_freq_;
    std::vector<int> unfolding_factor_;
};

#endif
