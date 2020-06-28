#ifndef CNQS_FOURIER_PRECONDITIONER_HPP
#define CNQS_FOURIER_PRECONDITIONER_HPP

#include <stdexcept>
#include <string>

#include "CnqsPreconditioner.hpp"
#include "CnqsVector.hpp"

/**
 * @brief Preconditioner for CnqsFourierOperator objects
 *
 * This class preconditions by inverting the Laplace part of the Hamiltonian:
 * \f[
 * \widehat{L} \widehat{\psi}(k) = \frac{g J}{2} \| k \|^2 \widehat{\psi}(k)
 * \f]
 */
class CnqsFourierPreconditioner : public CnqsPreconditioner {
public:
    /**
     * @brief Default constructor
     *
     */
    CnqsFourierPreconditioner(int d, int n, double g, double J, double shift);

    /**
     * @brief Default destructor
     *
     */
    ~CnqsFourierPreconditioner() = default;

    /**
     * @brief Test if preconditioner can be applied on a CnqsVector
     *
     * @param cnqs_vector CnqsVector object that will be operated on
     */
    void TestCompatibility(const CnqsVector &cnqs_vector) const {
        if (cnqs_vector.Size() != num_element_) {
            throw std::length_error(
                "==CnqsFourierPreconditioner== Vector length is not compatible "
                "with operator");
        }
    }

    /**
     * @brief Apply the preconditioner to a CnqsVector
     *
     * This function inverts the Laplacian part of the Hamiltonian.
     *
     * @param input_vector Input Vector
     * @param output_vector Output Vector
     */
    void Solve(const CnqsVector &input_vector, CnqsVector &output_vector) const;

private:
    double SquaredDistanceFromCenter(int i) const;

    int d_;
    int n_;
    double g_;
    double J_;
    double shift_;
    int max_freq_;
    int num_element_;
};

#endif
