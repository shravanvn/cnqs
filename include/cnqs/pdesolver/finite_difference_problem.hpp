#ifndef CNQS_PDESOLVER_FINITEDIFFERENCEPROBLEM_HPP
#define CNQS_PDESOLVER_FINITEDIFFERENCEPROBLEM_HPP

#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Time.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <memory>
#include <string>
#include <vector>

#include "cnqs/pdesolver/hamiltonian.hpp"
#include "cnqs/pdesolver/problem.hpp"

namespace cnqs {

namespace pdesolver {

/// @brief Basic finite-difference implementation of Problem
///
/// The space \f$[0, 2\pi]^d\f$ is discretized uniformly with \f$n\f$ grid
/// points per dimension. Then any state \f$\psi \in \mathcal{H}^1([0,
/// 2\pi]^d)\f$ that is \f$2\pi\f$-periodic along each of the dimensions is
/// discretized as a \f$d\f$-dimensional \f$n \times \cdots \times n\f$ tensor
/// \f$v\f$ with
///
/// \f[
///     v(i_0, \ldots, i_{d - 1}) \approx \psi(i_0 h, \ldots, i_{d - 1} h),
///     \quad 0 \leq i_k \leq n - 1, \quad 0 \leq k \leq d - 1, \quad h =
///     \frac{2 \pi}{n}
/// \f]
///
/// The derivative is approximated using a fourth-order five-point
/// center-difference stencil
///
/// \f[
///     f''(x) \approx \frac{-f(x - 2h) + 16 f(x - h) - 30 f(x) + 16 f(x + h)
///     - f(x + 2h)}{h^2}
/// \f]
///
/// Wrap-around is employed at the dimension edges to capture periodicy of the
/// state \f$\psi\f$ in the tensor \f$v\f$.
///
/// @note This class is built on top of Trilinos to support distributed
/// computing.
class FiniteDifferenceProblem : public Problem {
public:
    /// @brief Construct a FiniteDifferenceProblem given hamiltonian and
    /// discretization
    ///
    /// @param [in] hamiltonian Quantum rotor hamiltonian \f$(\mathcal{V},
    /// \mathcal{E})\f$, implemented in Hamiltonian.
    ///
    /// @param [in] num_grid_point Number of grid points per dimension, \f$n\f$.
    ///
    /// @param [in] comm Communicator.
    FiniteDifferenceProblem(
        const std::shared_ptr<const Hamiltonian> &hamiltonian,
        long num_grid_point,
        const Teuchos::RCP<const Teuchos::Comm<int>> &comm);

    double RunInversePowerIteration(long num_power_iter, double tol_power_iter,
                                    long num_cg_iter, double tol_cg_iter,
                                    const std::string &file_name) const;

private:
    Teuchos::RCP<const Tpetra::Map<int, long>> ConstructMap(
        const Teuchos::RCP<Teuchos::Time> &timer) const;

    Teuchos::RCP<Tpetra::MultiVector<double, int, long>> ConstructInitialState(
        const Teuchos::RCP<const Tpetra::Map<int, long>> &map,
        const Teuchos::RCP<Teuchos::Time> &timer) const;

    Teuchos::RCP<const Tpetra::CrsMatrix<double, int, long>>
    ConstructHamiltonian(const Teuchos::RCP<const Tpetra::Map<int, long>> &map,
                         const Teuchos::RCP<Teuchos::Time> &timer) const;

    std::shared_ptr<const Hamiltonian> hamiltonian_;
    long num_grid_point_;
    std::vector<long> unfolding_factors_;
    std::vector<double> theta_;
    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
};

}  // namespace pdesolver

}  // namespace cnqs

#endif
