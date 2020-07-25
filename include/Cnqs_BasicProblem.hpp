#ifndef CNQS_BASICPROBLEM_HPP
#define CNQS_BASICPROBLEM_HPP

#include <memory>
#include <string>

#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Time.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <nlohmann/json.hpp>

#include "Cnqs_Network.hpp"
#include "Cnqs_Problem.hpp"

namespace Cnqs {

/**
 * @brief Basic finite-difference implementation of Problem
 *
 * The space \f$[0, 2\pi]^d\f$ is discretized uniformly with \f$n\f$ grid points
 * per dimension. Then any state \f$\psi \in \mathcal{H}^1([0, 2\pi]^d)\f$ that
 * is \f$2\pi\f$-periodic along each of the dimensions is discretized as a
 * \f$d\f$-dimensional \f$n \times \cdots \times n\f$ tensor \f$v\f$ with
 *
 * \f[
 *     v(i_0, \ldots, i_{d - 1}) \approx \psi(i_0 h, \ldots, i_{d - 1} h), \quad
 *     0 \leq i_k \leq n - 1, \quad 0 \leq k \leq d - 1, \quad
 *     h = \frac{2 \pi}{n}
 * \f]
 *
 * The derivative is approximated using a fourth-order five-point
 * center-difference stencil
 *
 * \f[
 *     f''(x) \approx \frac{-f(x - 2h) + 16 f(x - h) - 30 f(x) + 16 f(x + h)
 *     - f(x + 2h)}{h^2}
 * \f]
 *
 * Wrap-around is employed at the dimension edges to capture periodicy of the
 * state \f$\psi\f$ in the tensor \f$v\f$.
 *
 * @note This class is built on top of Trilinos to support distributed
 * computing.
 */
class BasicProblem : public Problem {
public:
    /**
     * @brief Construct a BasicProblem given network and discretization
     *
     * @param [in] network Quantum rotor network \f$(\mathcal{V},
     * \mathcal{E})\f$, implemented in Network.
     * @param [in] numGridPoint Number of grid points per dimension, \f$n\f$.
     * @param [in] comm Communicator.
     */
    BasicProblem(const std::shared_ptr<const Cnqs::Network> &network,
                 int numGridPoint,
                 const Teuchos::RCP<const Teuchos::Comm<int>> &comm);

    double runInversePowerIteration(int numPowerIter, double tolPowerIter,
                                    int numCgIter, double tolCgIter,
                                    const std::string &fileName) const;

    nlohmann::json description() const;

    /**
     * @brief Print BasicProblem object to output streams
     *
     * @param [in,out] os Output stream
     * @param [in] problem Quantum rotor network Hamiltonian eigenproblem
     * discretized on finite difference grid
     * @return Output stream
     */
    friend std::ostream &operator<<(std::ostream &os,
                                    const BasicProblem &problem) {
        os << problem.description().dump(4);
        return os;
    }

private:
    Teuchos::RCP<const Tpetra::Map<int, int>>
    constructMap(const Teuchos::RCP<Teuchos::Time> &timer) const;

    Teuchos::RCP<Tpetra::MultiVector<double, int, int>>
    constructInitialState(const Teuchos::RCP<const Tpetra::Map<int, int>> &map,
                          const Teuchos::RCP<Teuchos::Time> &timer) const;

    Teuchos::RCP<const Tpetra::CrsMatrix<double, int, int>>
    constructHamiltonian(const Teuchos::RCP<const Tpetra::Map<int, int>> &map,
                         const Teuchos::RCP<Teuchos::Time> &timer) const;

    std::shared_ptr<const Cnqs::Network> network_;
    int numGridPoint_;
    std::vector<int> unfoldingFactors_;
    std::vector<double> theta_;
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
};

} // namespace Cnqs

#endif
