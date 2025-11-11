#ifndef LEGENDRE_H
#define LEGENDRE_H

#include <array>
#include <span>

namespace scarabee {

// Available N are 2, 4, 6, 8, 10, 12, 16, 32, and 64

template <std::size_t N>
class Legendre {
 public:
  std::span<const double> sin() const { return {sin_.begin(), sin_.end()}; }

  std::span<const double> invs_sin() const {
    return {invs_sin_.begin(), invs_sin_.end()};
  }

  std::span<const double> wgt() const { return {wgt_.begin(), wgt_.end()}; }

  std::span<const double> wsin() const { return {wsin_.begin(), wsin_.end()}; }

  std::span<const double> polar_angle() const {
    return {polar_angle_.begin(), polar_angle_.end()};
  }

 private:
  static const std::array<double, N / 2> invs_sin_;
  static const std::array<double, N / 2> wsin_;
  static const std::array<double, N / 2> sin_;
  static const std::array<double, N / 2> wgt_;
  static const std::array<double, N / 2> polar_angle_;
};

}  // namespace scarabee

#endif
