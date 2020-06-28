#ifndef CNQS_VECTOR_HPP
#define CNQS_VECTOR_HPP

#include <cmath>
#include <string>
#include <vector>

/**
 * @brief Provides functionality of vector objects from linear algebra
 *
 * This class is essentially a wrapper around the C++ standard vector with
 * support for vector addition/subtraction, scalar multipication/division, dot
 * product and Euclidean norm computation.
 */
class CnqsVector {
public:
    /**
     * @brief Default constructor
     */
    CnqsVector() = default;

    /**
     * @brief Default copy constructor
     */
    CnqsVector(const CnqsVector &) = default;

    /**
     * @brief Default move constructor
     */
    CnqsVector(CnqsVector &&) = default;

    /**
     * @brief Default copy assignment
     */
    CnqsVector &operator=(const CnqsVector &) = default;

    /**
     * @brief Default move assignment
     */
    CnqsVector &operator=(CnqsVector &&) = default;

    /**
     * @brief Default destructor
     */
    ~CnqsVector() = default;

    /**
     * @brief Construct a new CnqsVector object given the length
     *
     * All entries of the new object are zeros.
     *
     * @param n Length of the vector
     */
    CnqsVector(int n) : entries_(std::vector<double>(n, 0.0)) {}

    /**
     * @brief Construct a new CnqsVector object given the entries
     *
     * @param entries A C++ standard vector of doubles
     */
    CnqsVector(const std::vector<double> &entries) : entries_(entries) {}

    /**
     * @brief Size of the CnqsVector
     *
     * @return Number of elements in the vector
     */
    int Size() const { return entries_.size(); }

    /**
     * @brief Access element of CnqsVector at index
     *
     * @param i Vector Index
     * @return Constant reference to element
     */
    const double &operator()(int i) const { return entries_[i]; }

    /**
     * @brief Access element of CnqsVector at index
     *
     * @param i Vector index
     * @return Non-constant reference to element
     */
    double &operator()(int i) { return entries_[i]; }

    /**
     * @brief Assign all entries of a CnqsVector to a scalar
     *
     * Running `v = c` sets \f$v(i) \gets c\f$ for all indices \f$i\f$.
     *
     * @param c Scalar value to set all vector entries equal to
     * @return Self with updated entries
     */
    CnqsVector operator=(double c);

    /**
     * @brief In-place addition operator
     *
     * Running `v += w` sets \f$v(i) \gets v(i) + w(i)\f$ for all indices
     * \f$i\f$.
     *
     * @param v Other CnqsVector
     * @return Self with updated entries
     */
    CnqsVector operator+=(const CnqsVector &v);

    /**
     * @brief In-place subtraction operator
     *
     * Running `v -= w` sets \f$v(i) \gets v(i) - w(i)\f$ for all indices
     * \f$i\f$.
     *
     * @param v Other CnqsVector
     * @return Self with updated entries
     */
    CnqsVector operator-=(const CnqsVector &v);

    /**
     * @brief In-place scalar multiplication operator
     *
     * Running `v *= c` sets \f$v(i) \gets c v(i)\f$ for all indices \f$i\f$.
     *
     * @param c Scalar
     * @return Self with updated entries
     */
    CnqsVector operator*=(double c);

    /**
     * @brief In-place scalar division operator
     *
     * Running `v /= c` sets \f$v(i) \gets v(i) / c\f$ for all indices \f$i\f$.
     *
     * @param c Scalar
     * @returns Self with updated values
     */
    CnqsVector operator/=(double c);

    /**
     * @brief Addition operator
     *
     * Running `x = v + w` sets \f$x(i) \gets v(i) + w(i)\f$ for all indices
     * \f$i\f$.
     *
     * @param v Other CnqsVector
     * @return New CnqsVector with the sum.
     */
    CnqsVector operator+(const CnqsVector &v) const;

    /**
     * @brief Substraction operator
     *
     * Running `x = v - w` sets \f$x(i) \gets v(i) - w(i)\f$ for all indices
     * \f$i\f$.
     *
     * @param v Other CnqsVector
     * @returns New CnqsVector with the difference
     */
    CnqsVector operator-(const CnqsVector &v) const;

    /**
     * @brief Scalar right multiplication operator
     *
     * Running `x = v * c` sets \f$x(i) \gets c v(i)\f$ for all indices \f$i\f$.
     *
     * @param c Scalar
     * @returns New CnqsVector with scaled values
     */
    CnqsVector operator*(double c) const;

    /**
     * @brief Scalar right division operator
     *
     * Running `x = v / c` sets \f$x(i) \gets v(i) / c\f$ for all indices
     * \f$i\f$.
     *
     * @param c Scalar
     * @returns New CnqsVector with scaled values
     */
    CnqsVector operator/(double c) const;

    /**
     * @brief Scalar left multiplication
     *
     * Running `x = c * v` sets \f$x(i) \gets c v(i)\f$ for all indices \f$i\f$.
     *
     * @param c Scalar
     * @param v CnqsVector
     * @return New CnqsVector with scaled values
     */
    friend CnqsVector operator*(double c, const CnqsVector &v) { return v * c; }

    /**
     * @brief Save CnqsVector data to ASCII formatted file
     *
     * @param file_name Name of the text file
     */
    void Save(const std::string &file_name) const;

    /**
     * @brief Dot/Scalar product between two CnqsVector objects
     *
     * Given two vectors \f$v\f$ and \f$w\f$ computes \f$\langle v, w \rangle =
     * \sum_{i = 0}^{n - 1} v(i) w(i)\f$ where \f$n\f$ is the size of the
     * vector.
     *
     * @param v Other CnqsVector
     * @return Dot product
     */
    double Dot(const CnqsVector &v) const;

    /**
     * @brief Euclidean norm of CnqsVector
     *
     * Given vector \f$v\f$ computes \f$\| v \|_2 = \sqrt{\langle v, v
     * \rangle}\f$.
     *
     * @return Two-norm
     */
    double Norm() const { return std::sqrt(this->Dot(*this)); }

    /**
     * @brief Normalize CnqsVector in Euclidean norm
     *
     */
    void Normalize();

    /**
     * @brief Create a string representation of the CnqsVector object
     *
     * @return C++ standard string with description
     */
    std::string Describe() const;

private:
    std::vector<double> entries_;
};

/**
 * @brief Print CnqsVector objects to output streams (e.g. `std::cout`)
 *
 */
std::ostream &operator<<(std::ostream &os, const CnqsVector &v);

#endif
