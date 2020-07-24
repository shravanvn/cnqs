#ifndef CNQS_FOURIERPROBLEM_HPP
#define CNQS_FOURIERPROBLEM_HPP

#include <memory>
#include <string>

#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Time.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>

#include "Cnqs_Network.hpp"
#include "Cnqs_Problem.hpp"

namespace Cnqs {

/**
 * @brief Fourier-series based implementation of Problem
 *
 * Since the state \f$\psi \in \mathcal{H}^1([0, 2\pi]^d)\f$ is assumed to be
 * \f$2\pi\f$-periodic along each of the dimensions, it can be represented in
 * terms of a Fourier series, with the Fourier coefficients
 * \f$\hat{\psi}(\omega)\f$ defined for \f$\omega \in \mathbb{Z}^d\f$. The
 * Hamiltonian for these coefficients is given by
 *
 * \f[
 *     \hat{H} \hat{\psi}(\omega) = \frac{1}{2} \| \omega \|^2
 *     \hat{\psi}(\omega) - \frac{1}{2} \sum_{(j, k) \in \mathcal{E}} g_{jk}
 *     [\hat{\psi}(\omega + e_j - e_k) + \hat{\psi}(\omega - e_j + e_k)]
 * \f]
 *
 * where \f$e_j\f$ is the \f$j\f$-th standard basis of \f$\mathbb{R}^d\f$. We
 * truncate this lattice by restricting \f$-\omega_\text{max} \leq \omega_j \leq
 * \omega_\text{max}\f$ (in other words, setting \f$\hat{\psi}(\omega) = 0\f$
 * for \f$\omega\f$ outside this limit). This leads to a \f$d\f$ dimensional
 * \f$(2 \omega_\text{max} + 1) \times \cdots \times (2 \omega_\text{max} +
 * 1)\f$ tensor with
 *
 * \f[
 *     v(i_0, \ldots, i_{d - 1}) = \hat{\psi}(i_0 - \omega_\text{max}, \ldots,
 *     i_{d - 1} - \omega_\text{max}), \quad 0 \leq i_k \leq 2
 *     \omega_\text{max}, \quad 0 \leq k \leq d - 1
 * \f]
 *
 * @note This class is built on top of Trilinos to support distributed
 * computing.
 */
class FourierProblem : public Problem {
public:
    /**
     * @brief Construct a FourierProblem given network and cutoff frequency
     *
     * @param [in] network Quantum rotor network \f$(\mathcal{V},
     * \mathcal{E})\f$, implemented in Network.
     * @param [in] maxFreq Cutoff frequency, \f$\omega_\text{max}\f$.
     * @param [in] comm Communicator.
     */
    FourierProblem(const std::shared_ptr<const Cnqs::Network> &network,
                   int maxFreq,
                   const Teuchos::RCP<const Teuchos::Comm<int>> &comm);

    /**
     * @brief Run inverse power iteration
     *
     * Computing the lowest-energy eigenpair of the Hamiltonian \f$\hat{H}\f$
     * can be achieved using inverse power iteration. Given a lower-bound
     * \f$\mu\f$ of the eigenvalues of this operator, one starts at an arbitrary
     * state \f$\hat{\psi}_0\f$ and iterates
     *
     * \f[
     *     \hat{\psi}_{k + 1} = \frac{(\hat{H} - \mu I)^{-1} \hat{\psi}_k}{\|
     *     (\hat{H} - \mu I)^{-1} \hat{\psi}_k \|}, \quad \lambda_{k + 1} =
     *     \langle \hat{\psi}_{k + 1}, \hat{H} \hat{\psi}_{k + 1} \rangle
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
     * tolerance \f$\tau_\text{CG}\f$. To accelerate the CG iterations, we
     * empoly a preconditioner defined as
     *
     * \f[
     *     \hat{M} \hat{\psi}(\omega) = \sqrt{\left[\frac{1}{2} \| \omega \|^2
     *     - \mu\right]} \;\; \hat{\psi}(\omega)
     * \f]
     *
     * and solve the linear system
     *
     * \f[
     *     \hat{M}^{-1} (\hat{H} - \mu I) \hat{M}^{-1} \hat{M}
     *     \hat{\phi}_{k + 1} = \hat{M}^{-1} \hat{\psi}_k
     * \f]
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
    double runInversePowerIteration(int numPowerIter, double tolPowerIter,
                                    int numCgIter, double tolCgIter,
                                    const std::string &fileName) const;

    std::string description() const;

private:
    Teuchos::RCP<const Tpetra::Map<int, int>>
    constructMap(const Teuchos::RCP<Teuchos::Time> &timer) const;

    Teuchos::RCP<Tpetra::MultiVector<double, int, int>>
    constructInitialState(const Teuchos::RCP<const Tpetra::Map<int, int>> &map,
                          const Teuchos::RCP<Teuchos::Time> &timer) const;

    Teuchos::RCP<const Tpetra::CrsMatrix<double, int, int>>
    constructHamiltonian(const Teuchos::RCP<const Tpetra::Map<int, int>> &map,
                         const Teuchos::RCP<Teuchos::Time> &timer) const;

    Teuchos::RCP<const Tpetra::CrsMatrix<double, int, int>>
    constructPreconditioner(
        const Teuchos::RCP<const Tpetra::Map<int, int>> &map,
        const Teuchos::RCP<Teuchos::Time> &timer) const;

    std::shared_ptr<const Cnqs::Network> network_;
    int maxFreq_;
    std::vector<int> unfoldingFactors_;
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
};

} // namespace Cnqs

#endif
