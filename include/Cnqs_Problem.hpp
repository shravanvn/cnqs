#ifndef CNQS_PROBLEM_HPP
#define CNQS_PROBLEM_HPP

#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "Cnqs_Network.hpp"

namespace Cnqs {

/**
 * @brief Abstract interface to defining the eigenvalue minimization problem on
 * quantum networks
 *
 * Given a quantum rotor network \f$(\mathcal{V}, \mathcal{E})\f$ (implemented
 * via Network), one can associate a Hamiltonian on the Hilbert space
 * \f$\mathcal{H}([0, 2\pi]^d)\f$
 *
 * \f[
 *     H \psi(\theta) = -\frac{1}{2} \sum_{j = 0}^{d - 1}
 *     \frac{\partial^2 \psi}{\partial \theta_j^2} (\theta) -
 *     \sum_{(j, k) \in \mathcal{E}} g_{jk} \cos(\theta_j - \theta_k)
 *     \psi(\theta)
 * \f]
 *
 * The objective is to obtain the minimum energy eigenstate of this Hamiltonian.
 * This class provides an abstract interface for defining this problem.
 *
 * Its concrete subclasses implement various ways to discretize the Hamiltonian.
 */
template <class Scalar, class Index>
class Problem {
public:
    /**
     * @brief Default destructor
     *
     */
    virtual ~Problem() = default;

    /**
     * @brief Run inverse power iteration
     *
     * Computing the lowest-energy eigenpair of the Hamiltonian \f$H\f$ can be
     * achieved using inverse power iteration. Given a lower-bound \f$\mu\f$ of
     * the eigenvalues of this operator, one starts at an arbitrary state
     * \f$\psi_0\f$ and iterates
     *
     * \f[
     *     \psi_{k + 1} = \frac{(H - \mu I)^{-1} \psi_k}{\| (H - \mu I)^{-1}
     *     \psi_k \|}, \quad \lambda_{k + 1} = \langle \psi_{k + 1}, H
     *     \psi_{k + 1} \rangle
     * \f]
     *
     * This iteration is stopped when
     *
     * -    Iteration counter \f$k\f$ reaches a maximal value
     *      \f$k_\text{power}\f$
     * -    The difference between the eigenvalue estimates drop below a
     *      threshold: \f$| \lambda_{k + 1} - \lambda_k | < \tau_\text{power}\f$
     *
     * To solve the linear system in the power iteration, we employ the CG
     * iterative solver with maxinum number of iterations \f$k_\text{CG}\f$ and
     * tolerance \f$\tau_\text{CG}\f$.
     *
     * @param [in] maxPowerIter Maximum number of power iterations,
     * \f$k_\text{power}\f$.
     * @param [in] tolPowerIter Tolerance at which to stop power iteration,
     * \f$\tau_\text{power}\f$.
     * @param [in] maxCgIter Number of CG iterations to run per power iteration,
     * \f$k_\text{CG}\f$.
     * @param [in] tolCgIter Tolerance at which to stop CG iteration,
     * \f$\tau_\text{CG}\f$.
     * @param [in] fileName File where to store the estimated eigenstate. If set
     * to the empty string `""`, then the eigenstate is not saved.
     * @return Estimated smallest eigenvalue of the Hamiltonian.
     */
    virtual Scalar runInversePowerIteration(
        Index maxPowerIter, Scalar tolPowerIter, Index maxCgIter,
        Scalar tolCgIter, const std::string &fileName) const = 0;

    /**
     * @brief Describe the eigenvalue minimization problem
     *
     * @return Description of the problem.
     */
    virtual nlohmann::json description() const = 0;
};

}  // namespace Cnqs

#endif
