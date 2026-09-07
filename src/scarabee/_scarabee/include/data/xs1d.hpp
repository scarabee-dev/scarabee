#ifndef SCARABEE_XS1D_H
#define SCARABEE_XS1D_H

#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>
#include <utils/serialization.hpp>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include <cereal/cereal.hpp>

struct XS1DPickler;

namespace scarabee {

class XS1D {
 public:
  XS1D() : xs_() {}

  XS1D(const xt::xtensor<double, 1>& xs) : xs_(xs) {}

  double operator()(const std::size_t g) const {
    if (g >= xs_.size()) {
      return 0.;
    }

    return xs_(g);
  }

  double xs_fast(const std::size_t g) const { return xs_(g); }

  void set_value(const std::size_t g, const double v) {
    if (g >= xs_.size()) {
      const auto mssg = "Group index is out of range.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    xs_(g) = v;
  }

  std::size_t ngroups() const { return xs_.size(); }

  void resize(const std::size_t NG) {
    xt::xtensor<double, 1> tmp = xs_;
    xs_ = xt::zeros<double>({NG});

    for (std::size_t g = 0; g < std::min(NG, tmp.size()); g++) {
      xs_(g) = tmp(g);
    }
  }

  XS1D zeros_like() const {
    XS1D out(*this);
    out.xs_.fill(0.);
    return out;
  }

  XS1D& operator+=(const XS1D xs2) {
    if (this->ngroups() < xs2.ngroups()) {
      this->resize(xs2.ngroups());
    }

    for (std::size_t g = 0; g < this->ngroups(); g++) xs_(g) += xs2(g);

    return *this;
  }

  XS1D& operator-=(const XS1D xs2) {
    if (this->ngroups() < xs2.ngroups()) {
      this->resize(xs2.ngroups());
    }

    for (std::size_t g = 0; g < this->ngroups(); g++) xs_(g) -= xs2(g);

    return *this;
  }

  XS1D& operator*=(const double v) {
    xs_ *= v;
    return *this;
  }

  XS1D& operator*=(const XS1D& xs2) {
    for (std::size_t g = 0; g < this->ngroups(); g++) xs_(g) *= xs2(g);

    return *this;
  }

  XS1D& operator/=(const double v) {
    xs_ /= v;
    return *this;
  }

  XS1D operator+(const XS1D& xs2) const {
    XS1D out(*this);
    out += xs2;
    return out;
  }

  XS1D operator-(const XS1D& xs2) const {
    XS1D out(*this);
    out -= xs2;
    return out;
  }

  XS1D operator*(const double v) const {
    XS1D out(*this);
    out *= v;
    return out;
  }

  XS1D operator*(const XS1D& xs2) const {
    XS1D out(*this);
    out *= xs2;
    return out;
  }

  XS1D operator/(const double v) const {
    XS1D out(*this);
    out /= v;
    return out;
  }

 private:
  xt::xtensor<double, 1> xs_;

  friend class cereal::access;
  friend struct ::XS1DPickler;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(xs_));
  }
};

inline XS1D operator*(const double v, const XS1D& xs) {
  XS1D out(xs);
  out *= v;
  return out;
}

}  // namespace scarabee

#endif
