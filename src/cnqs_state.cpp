#include "cnqs_state.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

CnqsState CnqsState::operator=(double c) {
    for (int i = 0; i < size_; ++i) {
        data_[i] = c;
    }

    return *this;
}

CnqsState CnqsState::operator+=(const CnqsState &v) {
    if (size_ != v.size_) {
        throw std::length_error("Size mismatch in CnqsState += operator");
    }

    for (int i = 0; i < size_; ++i) {
        data_[i] += v(i);
    }

    return *this;
}

CnqsState CnqsState::operator+(const CnqsState &v) const {
    if (size_ != v.size_) {
        throw std::length_error("Size mismatch in CnqsState + operator");
    }

    CnqsState w(size_);

    for (int i = 0; i < size_; ++i) {
        w(i) = (*this)(i) + v(i);
    }

    return w;
}

CnqsState CnqsState::operator-=(const CnqsState &v) {
    if (size_ != v.size_) {
        throw std::length_error("Size mismatch in CnqsState -= operator");
    }

    for (int i = 0; i < size_; ++i) {
        data_[i] -= v(i);
    }

    return *this;
}

CnqsState CnqsState::operator-(const CnqsState &v) const {
    if (size_ != v.size_) {
        throw std::length_error("Size mismatch in CnqsState - operator");
    }

    CnqsState w(size_);

    for (int i = 0; i < size_; ++i) {
        w(i) = (*this)(i)-v(i);
    }

    return w;
}

void CnqsState::save(const std::string &file_name) const {
    std::ofstream file(file_name);

    file << std::scientific << std::setprecision(16);

    for (int i = 0; i < size_; ++i) {
        file << data_[i] << std::endl;
    }

    file.close();
}

CnqsState operator*(double c, const CnqsState &v) {
    int size = v.size_;

    CnqsState w(size);

    for (int i = 0; i < size; ++i) {
        w(i) = c * v(i);
    }

    return w;
}

double dot(const CnqsState &v1, const CnqsState &v2) {
    int size = v1.size_;

    if (size != v2.size_) {
        throw std::length_error("Size mismatch in CnqsState dot function");
    }

    double dot_product = 0.0;

    for (int i = 0; i < size; ++i) {
        dot_product += v1(i) * v2(i);
    }

    return dot_product;
}

double norm(const CnqsState &v) { return std::sqrt(dot(v, v)); }
