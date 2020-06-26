#ifndef CNQS_HAMILTONIAN_HPP
#define CNQS_HAMILTONIAN_HPP

#include <string>
#include <tuple>
#include <vector>

#include "cnqs_preconditioner.hpp"
#include "cnqs_state.hpp"

class CnqsHamiltonian {
  public:
    CnqsHamiltonian(int d, int n, std::vector<std::tuple<int, int>> edges,
                    double g, double J);

    virtual ~CnqsHamiltonian() = default;

    double g() const { return g_; }

    double J() const { return J_; }

    virtual CnqsState initialize_state() const = 0;

    virtual CnqsState operator*(const CnqsState &state) const = 0;

    void inverse_power_iteration(const CnqsPreconditioner &preconditioner,
                                 int cg_max_iter, double cg_tol,
                                 int power_max_iter, double power_tol,
                                 const std::string &file_name) const;

  protected:
    int d_;
    int n_;
    std::vector<std::tuple<int, int>> edges_;
    double g_;
    double J_;
    int num_element_;
    int num_edge_;
    std::vector<double> theta_;
};

#endif
