#include "cnqs_state.hpp"

#include <cmath>
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

void CnqsState::save(const hid_t &file_id, int snapshot_id) const {
    const std::string dataset_name = "/state_" + std::to_string(snapshot_id);

    const hsize_t dims[1] = {static_cast<hsize_t>(size_)};

    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    if (dataspace_id < 0) {
        throw std::runtime_error("--HDF5-- Could not create dataspace");
    }

    hid_t dataset_id =
        H5Dcreate(file_id, dataset_name.data(), H5T_NATIVE_DOUBLE, dataspace_id,
                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        throw std::runtime_error("--HDF5-- Could not create dataset");
    }

    herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, data_.data());
    if (status < 0) {
        throw std::runtime_error("--HDF5-- Could not write to dataset");
    }

    status = H5Dclose(dataset_id);
    if (status < 0) {
        throw std::runtime_error("--HDF5-- Could not close dataset");
    }

    status = H5Sclose(dataspace_id);
    if (status < 0) {
        throw std::runtime_error("--HDF5-- Could not close dataspace");
    }
}

void CnqsState::load(const hid_t &file_id, int snapshot_id) {
    std::string dataset_name = "/state_" + std::to_string(snapshot_id);

    hid_t dataset_id = H5Dopen(file_id, dataset_name.data(), H5P_DEFAULT);
    if (dataset_id < 0) {
        throw std::runtime_error("--HDF5-- Could not open dataset");
    }

    hid_t dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        throw std::runtime_error(
            "--HDF5-- Could not retrieve dataspace from dataset");
    }

    hsize_t dims[1];
    int status = H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    if (status < 0) {
        throw std::runtime_error(
            "--HDF5-- Could not extract dataset dimension");
    }

    size_ = static_cast<int>(dims[0]);
    data_.reserve(size_);

    herr_t status2 = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, data_.data());
    if (status2 < 0) {
        throw std::runtime_error("--HDF5-- Could not read data from dataset");
    }

    status2 = H5Sclose(dataspace_id);
    if (status2 < 0) {
        throw std::runtime_error("--HDF5-- Could not clean dataspace");
    }

    status2 = H5Dclose(dataset_id);
    if (status2 < 0) {
        throw std::runtime_error("--HDF5-- Could not close dataset");
    }
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
