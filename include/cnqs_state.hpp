#ifndef CNQS_STATE_HPP
#define CNQS_STATE_HPP

#include <string>
#include <vector>

class CnqsState {
  public:
    CnqsState() = default;

    ~CnqsState() = default;

    CnqsState(const CnqsState &) = default;

    CnqsState(CnqsState &&) = default;

    CnqsState &operator=(const CnqsState &) = default;

    CnqsState &operator=(CnqsState &&) = default;

    CnqsState(int n) : size_(n), data_(std::vector<double>(n, 0.0)) {}

    CnqsState(const std::vector<double> &data)
        : size_(data.size()), data_(data) {}

    CnqsState operator=(double c);

    const double &operator()(int i) const { return data_[i]; }

    double &operator()(int i) { return data_[i]; }

    CnqsState operator+=(const CnqsState &v);

    CnqsState operator+(const CnqsState &v) const;

    CnqsState operator-=(const CnqsState &v);

    CnqsState operator-(const CnqsState &v) const;

    void save(const std::string &file_name) const;

    friend CnqsState operator*(double c, const CnqsState &v);

    friend double dot(const CnqsState &v1, const CnqsState &v2);

    friend double norm(const CnqsState &v);

  private:
    int size_;
    std::vector<double> data_;
};

#endif
