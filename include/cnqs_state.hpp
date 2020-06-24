#ifndef CNQS_STATE_HPP
#define CNQS_STATE_HPP

#include <armadillo>

class CnqsState {
  public:
    CnqsState() = default;

    ~CnqsState() = default;

    CnqsState(const CnqsState &) = default;

    CnqsState(CnqsState &&) = default;

    CnqsState &operator=(const CnqsState &) = default;

    CnqsState &operator=(CnqsState &&) = default;

    CnqsState(unsigned long n)
        : vec_(arma::Col<double>(n, arma::fill::zeros)) {}

    CnqsState(const arma::Col<double> &vec) : vec_(vec) {}

    CnqsState operator=(double c) {
        vec_.fill(c);
        return *this;
    }

    CnqsState operator+(const CnqsState &v) { return CnqsState(vec_ + v.vec_); }

    CnqsState operator-(const CnqsState &v) { return CnqsState(vec_ - v.vec_); }

    double &operator()(unsigned long i) { return vec_(i); }

    void save(const std::string &file_name, unsigned long id = 0);

    void load(const std::string &file_name, unsigned long id = 0);

    friend CnqsState operator*(double c, const CnqsState &v) {
        return CnqsState(c * v.vec_);
    }

    friend double dot(const CnqsState &v1, const CnqsState &v2) {
        return arma::dot(v1.vec_, v2.vec_);
    }

    friend double norm(const CnqsState &v) { return arma::norm(v.vec_); }

  private:
    arma::Col<double> vec_;
};

#endif
