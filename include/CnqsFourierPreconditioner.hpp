#ifndef CNQS_FOURIER_PRECONDITIONER_HPP
#define CNQS_FOURIER_PRECONDITIONER_HPP

#include <stdexcept>
#include <string>

#include "CnqsPreconditioner.hpp"
#include "CnqsVector.hpp"

/**
 * @brief Preconditioner for CnqsVector objects corresponding to the Laplacian
 * part of a CnqsFourierOperator
 *
 * A CnqsFourierPreconditioner \f$\hat{M}_\mu\f$ with shift \f$\mu\f$ acts on a
 * \f$d\f$-dimensional Fourier coefficients \f$\hat{\psi}\f$ defined on the
 * lattice \f$\mathbb{Z}^d\f$ as follows
 *
 * \f[
 *     \hat{M}_\mu \hat{\psi}(k) = \frac{g J}{2} \| k \|^2 \hat{\psi}(k) - \mu
 *     \hat{\psi}(k)
 * \f]
 *
 * Note that this is just the shifted Laplacian part of the CnqsFourierOperator.
 * Just as in that case, the corresponding CnqsVector object should be a
 * truncation of the Fourier coefficient lattice \f$\hat{\psi}(k)\f$ on \f$k \in
 * \mathbb{Z}^d\f$: only the basic truncation
 *
 * \f[
 *     -k_\text{max} \leq k \leq k_\text{max}
 * \f]
 *
 * is supported and all Fourier coefficients outside this hypercube are assumed
 * to be zero. This leads to a \f$d\f$ dimensional tensor \f$v(i_0, i_1, \ldots,
 * i_{d - 1})\f$ of size \f$(2 k_\text{max} + 1)
 * \times (2 k_\text{max} + 1) \times \cdots \times (2 k_\text{max} + 1)\f$ with
 *
 * \f[
 *     v(i_0, i_1, \ldots, i_{d - 1}) \approx \hat{\psi}(i_0 - k_\text{max},
 *     i_1 - k_\text{max}, \ldots, i_{d - 1} - k_\text{max})
 * \f]
 *
 * The CnqsVector object representing \f$v\f$ is assumed to have used a
 * column-major indexing scheme: the \f$d\f$-dimensional index \f$(i_0, i_1,
 * \ldots, i_{d - 1})\f$ corresponds to the linear index
 *
 * \f[
 *     i = i_0 + n i_1 + \cdots + n^{d - 1} i_{d - 1}
 * \f]
 */
class CnqsFourierPreconditioner : public CnqsPreconditioner {
public:
    /**
     * @brief Construct a new CnqsFourierPreconditioner object
     *
     * @param [in] num_rotor Number of quantum rotors \f$d\f$. At least two
     * rotors is needed to construct the operator
     *
     * @param [in] max_freq Frequency cut-off parameter \f$k_\text{max}\f$. This
     * maximum frequency should be at least one.
     *
     * @param [in] g Parameter \f$g\f$
     *
     * @param [in] J Parameter \f$J\f$. To ensure a finite lower bound of for
     * the spectram of the Hamiltonian, the product \f$g J\f$ is required to be
     * positive
     *
     * @param shift Shift \f$\mu\f$
     *
     * @throw std::domain_error If the input arguments are illegal as described
     * above
     */
    CnqsFourierPreconditioner(int num_rotor, int max_freq, double g, double J,
                              double shift);

    /**
     * @brief Default destructor
     *
     */
    ~CnqsFourierPreconditioner() = default;

    /**
     * @brief Test if preconditioner can be applied on a CnqsVector
     *
     * Checks if \f$\hat{M}_\mu^{-1} v\f$ can be computed by ensuring the number
     * of elements in the \f$v\f$ is \f$(2 k_\text{max} + 1)^d\f$.
     *
     * @param [in] cnqs_vector CnqsVector vector \f$v\f$
     */
    void TestCompatibility(const CnqsVector &cnqs_vector) const;

    /**
     * @brief Apply the preconditioner to a CnqsVector
     *
     * Compute \f$w = \hat{M}_\mu^{-1} v\f$ using on the truncated integer
     * lattice.
     *
     * @param input_vector CnqsVector \f$v\f$
     * @param output_vector CnqsVector \f$w\f$
     */
    void Solve(const CnqsVector &input_vector, CnqsVector &output_vector) const;

    /**
     * @brief Create a string representation of the CnqsFourierPreconditioner
     * object
     *
     * @return `std::string` with description
     */
    std::string Describe() const;

private:
    double SquaredDistanceFromCenter(int i) const;

    int num_rotor_;
    int max_freq_;
    double g_;
    double J_;
    double shift_;
    int n_;
    int num_element_;
};

#endif
