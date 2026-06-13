#ifndef SCARABEE_TAB1_H
#define SCARABEE_TAB1_H

#include <xtensor/containers/xtensor.hpp>
#include <xtensor-python/pytensor.hpp>

#include <vector>

namespace scarabee {

class Tab1 {
 public:
  // Lin-Lin constructors
  Tab1(const std::vector<double>& x, const std::vector<double>& y);
  Tab1(const xt::xtensor<double, 1>& x, const xt::xtensor<double, 1>& y);

  // Arbitrary interpolation constructors
  Tab1(const std::vector<double>& x, const std::vector<double>& y,
       const std::vector<std::size_t>& breakpoints,
       const std::vector<int>& interpolations);
  Tab1(const xt::xtensor<double, 1>& x, const xt::xtensor<double, 1>& y,
       const std::vector<std::size_t>& breakpoints,
       const std::vector<int>& interpolations);

  const xt::xtensor<double, 1>& x() const { return x_; }
  const xt::xtensor<double, 1>& y() const { return y_; }
  const std::vector<std::size_t>& breakpoints() const { return breakpoints_; }
  const std::vector<int>& interpolations() const { return interpolations_; }

  double operator()(const double x) const;
  xt::xtensor<double, 1> operator()(const xt::pytensor<double, 1>& x) const;

  double integrate(const double a, const double b) const;

 private:
  xt::xtensor<double, 1> x_, y_;
  std::vector<std::size_t> breakpoints_;
  std::vector<int> interpolations_;

  double interpolate(const int p, const double x, const double x0,
                     const double y0, const double x1, const double y1) const;
  double integrate(const int p, const double x0, const double y0,
                   const double x1, const double y1) const;

  int get_interpolation(const std::size_t indx) const;

  void check_sign_interpolation_compatability() const;
};

}  // namespace scarabee

#endif
