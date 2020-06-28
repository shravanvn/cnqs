#include "CnqsVector.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

CnqsVector CnqsVector::operator=(double c) {
    for (auto &entry : entries_) {
        entry = c;
    }

    return *this;
}

CnqsVector CnqsVector::operator+=(const CnqsVector &v) {
    int size = entries_.size();

    if (v.Size() != size) {
        throw std::length_error("==CnqsVector== Size mismatch in operator +=");
    }

    for (int i = 0; i < size; ++i) {
        entries_[i] += v(i);
    }

    return *this;
}

CnqsVector CnqsVector::operator-=(const CnqsVector &v) {
    int size = entries_.size();

    if (v.Size() != size) {
        throw std::length_error("==CnqsVector== Size mismatch in operator -=");
    }

    for (int i = 0; i < size; ++i) {
        entries_[i] -= v(i);
    }

    return *this;
}

CnqsVector CnqsVector::operator*=(double c) {
    int size = entries_.size();

    for (int i = 0; i < size; ++i) {
        entries_[i] *= c;
    }

    return *this;
}

CnqsVector CnqsVector::operator/=(double c) {
    if (std::abs(c) < 1.0e-16) {
        throw std::logic_error(
            "==CnqsVector== Cannot divide by a near-zero number");
    }

    int size = entries_.size();

    for (int i = 0; i < size; ++i) {
        entries_[i] /= c;
    }

    return *this;
}

CnqsVector CnqsVector::operator+(const CnqsVector &v) const {
    CnqsVector w = *this;
    return w += v;
}

CnqsVector CnqsVector::operator-(const CnqsVector &v) const {
    CnqsVector w = *this;
    return w -= v;
}

CnqsVector CnqsVector::operator*(double c) const {
    CnqsVector w = *this;
    return w *= c;
}

CnqsVector CnqsVector::operator/(double c) const {
    CnqsVector w = *this;
    return w /= c;
}

double CnqsVector::Dot(const CnqsVector &v) const {
    int size = entries_.size();

    if (v.Size() != size) {
        throw std::length_error(
            "==CnqsVector== Size mismatch in computing dot product");
    }

    double dot_product = 0.0;

    for (int i = 0; i < size; ++i) {
        dot_product += entries_[i] * v(i);
    }

    return dot_product;
}

void CnqsVector::Save(const std::string &file_name) const {
    std::ofstream file(file_name);

    file << std::scientific << std::setprecision(16);

    for (const auto &entry : entries_) {
        file << entry << std::endl;
    }

    file.close();
}

std::string CnqsVector::Describe() const {
    std::string description = "{\n";
    description += "    \"name\": \"CnqsVector\",\n";
    description += "    \"size\": " + std::to_string(entries_.size()) + "\n";
    description += "}";

    return description;
}

std::ostream &operator<<(std::ostream &os, const CnqsVector &v) {
    os << v.Describe() << std::flush;
    return os;
}
