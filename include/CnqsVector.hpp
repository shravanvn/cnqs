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
    CnqsVector(const CnqsVector &v) = default;

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
     * @brief Construct a CnqsVector of given length
     *
     * @param [in] n Length of the vector
     *
     * @returns A CnqsVector of all zeros with specified length
     */
    CnqsVector(int n) : entries_(std::vector<double>(n, 0.0)) {}

    /**
     * @brief Construct a CnqsVector from data
     *
     * @param [in] entries A C++ stanard vector of doubles
     *
     * @returns A CnqsVector with specified entries
     */
    CnqsVector(const std::vector<double> &entries) : entries_(entries) {}

    /**
     * @brief Size of CnqsVector
     *
     * @returns Length of the CnqsVector
     */
    int Size() const { return entries_.size(); }

    /**
     * @brief Access element of CnqsVector at index
     *
     * @param [in] i Vector index
     *
     * @returns Constant reference to the CnqsVector element
     */
    const double &operator()(int i) const { return entries_[i]; }

    /**
     * @brief Access element of CnqsVector at index
     *
     * @param [in] i Vector index
     *
     * @returns Non-constant reference to the CnqsVector element
     */
    double &operator()(int i) { return entries_[i]; }

    /**
     * @brief Assign all entries of a CnqsVector to a scalar
     *
     * @param [in] c Scalar value to set the vector entries equal to
     *
     * @returns A CnqsVector all of whose entries are set to c
     */
    CnqsVector operator=(double c);

    /**
     * @brief In-place addition operator
     *
     * @param [in] v Other CnqsVector
     *
     * @returns Self with updated values
     */
    CnqsVector operator+=(const CnqsVector &v);

    /**
     * @brief In-place subtraction operator
     *
     * @param [in] v Other CnqsVector
     *
     * @returns Self with updated values
     */
    CnqsVector operator-=(const CnqsVector &v);

    /**
     * @brief In-place scalar multiplication operator
     *
     * @param [in] c Scalar
     *
     * @returns Self with updated values
     */
    CnqsVector operator*=(double c);

    /**
     * @brief In-place scalar division operator
     *
     * @param [in] c Scalar
     *
     * @returns Self with updated values
     */
    CnqsVector operator/=(double c);

    /**
     * @brief Addition operator
     *
     * @param [in] v Other CnqsVector
     *
     * @returns New CnqsVector with sum of the values
     */
    CnqsVector operator+(const CnqsVector &v) const;

    /**
     * @brief Substraction operator
     *
     * @param [in] v Other CnqsVector
     *
     * @returns New CnqsVector with signed difference of the values
     */
    CnqsVector operator-(const CnqsVector &v) const;

    /**
     * @brief Scalar right multiplication operator
     *
     * @param [in] c Scalar
     *
     * @returns New CnqsVector with scaled values
     */
    CnqsVector operator*(double c) const;

    /**
     * @brief Scalar right division operator
     *
     * @param [in] c Scalar
     *
     * @returns New CnqsVector with scaled values
     */
    CnqsVector operator/(double c) const;

    /**
     * @brief Scalar left multiplication
     *
     * @param [in] c Scalar
     * @param [in] v CnqsVector
     *
     * @returns New CnqsVector with scaled values
     */
    friend CnqsVector operator*(double c, const CnqsVector &v) { return v * c; }

    /**
     * @brief Save CnqsVector data to ASCII formatted file
     *
     * @param [in] file_name Name of the text file
     */
    void Save(const std::string &file_name) const;

    /**
     * @brief Dot/Scalar product between two CnqsVector objects
     *
     * @param [in] v Other CnqsVector
     *
     * @returns The dot product of two CnqsVector objects
     */
    double Dot(const CnqsVector &v) const;

    /**
     * @brief Euclidean norm of CnqsVector
     *
     * @param [in] v CnqsVector
     *
     * @returns The 2-norm of the CnqsVector
     */
    double Norm() const { return std::sqrt(this->Dot(*this)); }

  private:
    std::vector<double> entries_;
};

#endif
