#ifndef CNQS_HAMILTONIAN_HPP
#define CNQS_HAMILTONIAN_HPP

#include <tuple>
#include <vector>

#include "cnqs_state.hpp"

class CnqsHamiltonian {
  public:
    CnqsHamiltonian(
        unsigned long d, unsigned long n,
        std::vector<const std::tuple<unsigned long, unsigned long>> edges,
        double g, double J)
        : d_(d), n_(n), edges_(edges), g_(g), J_(J) {}

    ~CnqsHamiltonian() = default;

    CnqsState initialize_state() const;

    CnqsState operator*(const CnqsState &state) const;

    inline CnqsState trans_mult(const CnqsState &state) const {
        return *this * state;
    }

  private:
    unsigned long d_;
    unsigned long n_;
    std::vector<const std::tuple<unsigned long, unsigned long>> edges_;
    double g_;
    double J_;
};

#endif
