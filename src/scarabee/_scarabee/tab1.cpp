#include <utils/tab1.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>

#include <algorithm>
#include <cmath>
#include <span>

namespace scarabee {

Tab1::Tab1(const std::vector<double>& x, const std::vector<double>& y) {
  if (x.size() != y.size()) {
    const auto mssg = "The x and y arrays must have the same size.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (x.size() < 2) {
    const auto mssg = "Must provide at least 2 tabulated points.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (std::is_sorted(x.begin(), x.end()) == false) {
    const auto mssg = "The x values must be sorted.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  x_ = xt::zeros<double>({x.size()});
  y_ = xt::zeros<double>({y.size()});
  for (std::size_t i = 0; i < x.size(); i++) {
    x_(i) = x[i];
    y_(i) = y[i];
  }

  breakpoints_ = {x_.size()};
  interpolations_ = {2};

  this->check_sign_interpolation_compatability();
}

Tab1::Tab1(const xt::xtensor<double, 1>& x, const xt::xtensor<double, 1>& y)
    : x_(x), y_(y) {
  if (x.size() != y.size()) {
    const auto mssg = "The x and y arrays must have the same size.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (x.size() < 2) {
    const auto mssg = "Must provide at least 2 tabulated points.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (std::is_sorted(x.begin(), x.end()) == false) {
    const auto mssg = "The x values must be sorted.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  breakpoints_ = {x_.size()};
  interpolations_ = {2};

  this->check_sign_interpolation_compatability();
}

// Arbitrary interpolation constructors
Tab1::Tab1(const std::vector<double>& x, const std::vector<double>& y,
           const std::vector<std::size_t>& breakpoints,
           const std::vector<int>& interpolations)
    : breakpoints_(breakpoints), interpolations_(interpolations) {
  if (x.size() != y.size()) {
    const auto mssg = "The x and y arrays must have the same size.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (x.size() < 2) {
    const auto mssg = "Must provide at least 2 tabulated points.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (std::is_sorted(x.begin(), x.end()) == false) {
    const auto mssg = "The x values must be sorted.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (breakpoints_.size() != interpolations_.size()) {
    const auto mssg =
        "The breakpoints and interpolations must have the same size.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (breakpoints_.size() == 0) {
    const auto mssg = "Must provide at least 1 breakpoint and interpolation.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (std::is_sorted(breakpoints_.begin(), breakpoints_.end()) == false) {
    const auto mssg = "Breakpoints must be sorted.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (breakpoints_.front() == 0) {
    const auto mssg = "The first breakpoint cannot be zero.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (breakpoints_.back() != x.size()) {
    const auto mssg =
        "The last breakpoint must be equal to the number of tabulated points.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Check all interpolations
  for (const auto p : interpolations_) {
    if (p < 1 || p > 5) {
      const auto mssg = "All interpolations must be in the range [1,5].";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }
  }

  x_ = xt::zeros<double>({x.size()});
  y_ = xt::zeros<double>({y.size()});
  for (std::size_t i = 0; i < x.size(); i++) {
    x_(i) = x[i];
    y_(i) = y[i];
  }

  this->check_sign_interpolation_compatability();
}

Tab1::Tab1(const xt::xtensor<double, 1>& x, const xt::xtensor<double, 1>& y,
           const std::vector<std::size_t>& breakpoints,
           const std::vector<int>& interpolations)
    : x_(x), y_(y), breakpoints_(breakpoints), interpolations_(interpolations) {
  if (x.size() != y.size()) {
    const auto mssg = "The x and y arrays must have the same size.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (x.size() < 2) {
    const auto mssg = "Must provide at least 2 tabulated points.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (std::is_sorted(x.begin(), x.end()) == false) {
    const auto mssg = "The x values must be sorted.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (breakpoints_.size() != interpolations_.size()) {
    const auto mssg =
        "The breakpoints and interpolations must have the same size.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (breakpoints_.size() == 0) {
    const auto mssg = "Must provide at least 1 breakpoint and interpolation.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (std::is_sorted(breakpoints_.begin(), breakpoints_.end()) == false) {
    const auto mssg = "Breakpoints must be sorted.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (breakpoints_.front() == 0) {
    const auto mssg = "The first breakpoint cannot be zero.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (breakpoints_.back() != x.size()) {
    const auto mssg =
        "The last breakpoint must be equal to the number of tabulated points.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Check all interpolations
  for (const auto p : interpolations_) {
    if (p < 1 || p > 5) {
      const auto mssg = "All interpolations must be in the range [1,5].";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }
  }

  this->check_sign_interpolation_compatability();
}

double Tab1::operator()(const double x) const {
  // Get index for interpolation
  const auto itr = std::upper_bound(x_.begin(), x_.end(), x);
  if (itr == x_.begin())
    return y_.front();
  else if (itr == x_.end())
    return y_.back();

  const std::size_t indx = static_cast<std::size_t>(itr - x_.begin()) - 1;

  // Loop over interpolation regions
  const int p = get_interpolation(indx);

  // Perform interpolation
  return this->interpolate(p, x, x_[indx], y_[indx], x_[indx + 1],
                           y_[indx + 1]);
}

xt::xtensor<double, 1> Tab1::operator()(
    const xt::pytensor<double, 1>& x) const {
  xt::xtensor<double, 1> out = xt::zeros<double>({x.size()});

  for (std::size_t i = 0; i < x.size(); i++) {
    out.flat(i) = (*this)(x.flat(i));
  }

  return out;
}

double Tab1::integrate(const double ia, const double ib) const {
  // If the integration range doesn't go from low to high, flip the order
  double a = ia;
  double b = ib;
  bool flipped = false;
  if (a > b) {
    flipped = true;
    const double tmp = a;
    a = b;
    b = tmp;
  }

  // Clip the bounds if necessary
  if (a < x_.front()) a = x_.front();
  if (x_.back() < b) b = x_.back();

  // Check for this special case
  if (a == b) return 0.;

  // Now we can start to perform the real integral
  double integral = 0.;
  double x_lower_bound = a;
  double x_upper_bound = b;

  // Get the first index for interpolation
  const auto itr = std::upper_bound(x_.begin(), x_.end(), x_lower_bound);
  std::size_t idx = static_cast<std::size_t>(itr - x_.begin()) - 1;

  while (idx < x_.size() - 1) {
    // Get the interpolation rule
    const int interp = get_interpolation(idx);

    // Get tabulated values
    double xi = x_[idx];       // low edge of the corresponding bin
    double xi1 = x_[idx + 1];  // high edge of the corresponding bin
    double yi = y_[idx];
    double yi1 = y_[idx + 1];

    // If we are at one of the end points, perform the necessary interpolation
    if (xi < x_lower_bound) {
      yi = interpolate(interp, x_lower_bound, xi, yi, xi1, yi1);
      xi = x_lower_bound;
    }

    if (x_upper_bound < xi1) {
      yi1 = interpolate(interp, x_upper_bound, xi, yi, xi1, yi1);
      xi1 = x_upper_bound;
    }

    // Add check to ensure the loop will stop
    if (x_upper_bound == xi1) idx = x_.size();

    // Contribute to the integral
    integral += integrate(interp, xi, yi, xi1, yi1);

    // Prepare for next iteration
    idx += 1;
    x_lower_bound = xi1;
  }

  // If we had to flip integration bounds, multiply by - 1
  if (flipped) integral = -integral;

  return integral;
}

double Tab1::interpolate(const int p, const double x, const double x0,
                         const double y0, const double x1,
                         const double y1) const {
  switch (p) {
    case 1:
      // Histogram
      return y0;
      break;
    case 2:
      // Lin-Lin
      return y0 + (x - x0) / (x1 - x0) * (y1 - y0);
      break;
    case 3:
      // Lin-Log
      return y0 + std::log(x / x0) / std::log(x1 / x0) * (y1 - y0);
      break;
    case 4:
      // Log-Lin
      return y0 * std::exp((x - x0) / (x1 - x0) * std::log(y1 / y0));
      break;
    case 5:
      // Log-Log
      return y0 *
             std::exp(std::log(x / x0) / std::log(x1 / x0) * std::log(y1 / y0));
      break;
  }

  // Never gets here
  return 0.;
}

double Tab1::integrate(const int p, const double x0, const double y0,
                       const double x1, const double y1) const {
  switch (p) {
    case 1:
      // Histogram
      return y0 * (x1 - x0);
      break;
    case 2:
      // Lin-Lin
      {
        const double m = (y1 - y0) / (x1 - x0);
        return (y0 - m * x0) * (x1 - x0) + 0.5 * m * (x1 * x1 - x0 * x0);
      }
      break;
    case 3:
      // Lin-Log
      {
        const double logx = std::log(x1 / x0);
        const double m = (y1 - y0) / logx;
        return y0 + m * (x1 * (logx - 1.) + x0);
      }
      break;
    case 4:
      // Log-Lin
      {
        const double m = std::log(y1 / y0) / (x1 - x0);
        return y0 / m * (std::exp(m * (x1 - x0)) - 1.);
      }
      break;
    case 5:
      // Log-Log
      {
        const double m = std::log(y1 / y0) / std::log(x1 / x0);
        return y0 / ((m + 1) * std::pow(x0, m)) *
               (std::pow(x1, (m + 1)) - std::pow(x0, (m + 1)));
      }
      break;
  }

  // Never gets here
  return 0.;
}

int Tab1::get_interpolation(const std::size_t indx) const {
  for (std::size_t k = 0; k < breakpoints_.size(); k++) {
    if (indx < breakpoints_[k] - 1) {
      return interpolations_[k];
    }
  }

  // Should never get here
  return interpolations_.back();
}

void Tab1::check_sign_interpolation_compatability() const {
  // Get each interpolation range
  auto x_strt_it = x_.begin();
  auto y_strt_it = y_.begin();
  for (std::size_t ri = 0; ri < breakpoints_.size(); ri++) {
    std::span<const double> xr{x_strt_it, x_.begin() + breakpoints_[ri]};
    std::span<const double> yr{y_strt_it, y_.begin() + breakpoints_[ri]};
    const int interp = interpolations_[ri];

    for (int i = 0; i < static_cast<int>(xr.size()) - 1; i++) {
      const std::size_t indx = static_cast<std::size_t>(i);
      if (interp == 3 &&
          (std::signbit(xr[indx]) != std::signbit(xr[indx + 1]))) {
        // Lin-Log : Make sure all x have same sign
        const auto mssg =
            "Encountered x values of different sign in Lin-Log interpolation.";
        spdlog::error(mssg);
        throw ScarabeeException(mssg);
      } else if (interp == 4 &&
                 (std::signbit(yr[indx]) != std::signbit(yr[indx + 1]))) {
        // Log-Lin : Make sure all y have same sign
        const auto mssg =
            "Encountered y values of different sign in Log-Lin interpolation.";
        spdlog::error(mssg);
        throw ScarabeeException(mssg);
      } else if (interp == 5 &&
                 ((std::signbit(xr[indx]) != std::signbit(xr[indx + 1])) ||
                  (std::signbit(yr[indx]) != std::signbit(yr[indx + 1])))) {
        // Log-Log : Make sure all y have same sign
        const auto mssg =
            "Encountered x or y values of different sign in Log-Log "
            "interpolation.";
        spdlog::error(mssg);
        throw ScarabeeException(mssg);
      }
    }

    // For next iteration
    x_strt_it = x_.begin() + breakpoints_[ri];
    y_strt_it = y_.begin() + breakpoints_[ri];
  }
}

}  // namespace scarabee
