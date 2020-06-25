#ifndef CNQS_HAMILTONIAN_DIRECT_HPP
#define CNQS_HAMILTONIAN_DIRECT_HPP

#include <tuple>
#include <vector>

#include "cnqs_state.hpp"

class CnqsHamiltonianDirect {
  public:
    CnqsHamiltonianDirect(int d, int n, std::vector<std::tuple<int, int>> edges,
                          double g, double J);

    ~CnqsHamiltonianDirect() = default;

    CnqsState initialize_state() const;

    CnqsState operator*(const CnqsState &state) const;

    void inverse_power_iteration(int cg_max_iter, double cg_tol,
                                 int power_max_iter, double power_tol,
                                 const std::string &file_name) const;

  private:
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
