#ifndef VECTOR_H
#define VECTOR_H

#include <utils/constants.hpp>

#include <cereal/cereal.hpp>

#include <cmath>
#include <iostream>

namespace scarabee {

//============================================================================
// Vector Class
//----------------------------------------------------------------------------
class Vector {
 public:
  Vector(double i_x, double i_y) : x_(i_x), y_(i_y) {};
  ~Vector() = default;

  double x() const { return x_; }
  double y() const { return y_; }

  double dot(const Vector& v) const { return x_ * v.x() + y_ * v.y(); }

  double norm() const { return std::sqrt(x_ * x_ + y_ * y_); }

  Vector operator-() const { return Vector(-this->x_, -this->y_); }

  bool operator==(const Vector& v) const {
    return (std::abs(x_ - v.x()) < VEC_FP_TOL) &&
           (std::abs(y_ - v.y()) < VEC_FP_TOL);
  }

  bool operator!=(const Vector& v) const { return !this->operator==(v); }

 protected:
  double x_, y_;

  friend class cereal::access;
  Vector() {}
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(x_), CEREAL_NVP(y_));
  }
};

//============================================================================
// Overloaded Operator Declarations
//----------------------------------------------------------------------------
inline Vector operator+(const Vector& v1, const Vector& v2) {
  return Vector(v1.x() + v2.x(), v1.y() + v2.y());
}

inline Vector operator-(const Vector& v1, const Vector& v2) {
  return Vector(v1.x() - v2.x(), v1.y() - v2.y());
}

inline Vector operator*(const Vector& v, double d) {
  return Vector(v.x() * d, v.y() * d);
}

inline Vector operator*(double d, const Vector& v) { return v * d; }

inline Vector operator/(const Vector& v, double d) {
  return Vector(v.x() / d, v.y() / d);
}

inline double operator*(const Vector& v1, const Vector& v2) {
  return v1.dot(v2);
}

inline std::ostream& operator<<(std::ostream& output, const Vector& v) {
  output << "<" << v.x() << "," << v.y() << ">";
  return output;
}

}  // namespace scarabee

#endif
