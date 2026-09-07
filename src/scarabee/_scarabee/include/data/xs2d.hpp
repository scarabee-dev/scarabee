#ifndef SCARABEE_XS2D_H
#define SCARABEE_XS2D_H

#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>
#include <utils/serialization.hpp>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include <cereal/cereal.hpp>

#include <cstdint>
#include <sstream>

struct XS2DPickler;

namespace scarabee {

class XS2D {
 public:
  XS2D() : xs_(), packing_() {
    xs_ = xt::zeros<double>(
        {static_cast<std::size_t>(0), static_cast<std::size_t>(0)});
  }

  XS2D(const xt::xtensor<double, 2>& xs,
       const xt::xtensor<std::uint32_t, 2>& packing)
      : xs_(xs), packing_(packing) {
    // Make sure packing is valid
    if (packing_.shape()[1] != 3) {
      const auto mssg =
          "Second dimension of packing array must have a length of 3.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    for (std::size_t g = 0; g < packing_.shape()[0]; g++) {
      if (packing_(g, 1) > packing_(g, 2)) {
        const auto mssg =
            "The lowest outgoing group is larger than the highest outgoing "
            "group.";
        spdlog::error(mssg);
        throw ScarabeeException(mssg);
      }
    }

    // Make sure the shapes align
    if (xs_.shape()[0] == 0) {
      const auto mssg = "Must provide at least the P0 scattering moment data.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    const std::size_t NG = packing_.shape()[0] - 1;
    const std::size_t NDAT =
        packing_(NG, 0) + packing_(NG, 2) + 1 - packing_(NG, 1);

    if (xs.shape()[1] != NDAT) {
      const auto mssg =
          "Length of the cross section data array does not agree with length "
          "indicated by packing scheme.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }
  }

  XS2D(const xt::xtensor<double, 2>& Es) : xs_(), packing_() {
    if (Es.shape()[0] != Es.shape()[1]) {
      const auto mssg = "Scattering matrix is not square.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    const std::size_t NG = Es.shape()[0];

    packing_.resize({NG, 3});
    packing_.fill(0);

    for (std::size_t g = 0; g < NG; g++) {
      // Get the min and max outgoing group for each incident group
      std::size_t g_min = 0;
      for (std::size_t gg = 0; gg < NG; gg++) {
        if (Es(g, gg) != 0.) {
          g_min = gg;
          break;
        }
      }

      std::size_t g_max = NG - 1;
      for (int gg = static_cast<int>(NG) - 1; gg >= 0; gg--) {
        if (Es(g, static_cast<std::size_t>(gg)) != 0.) {
          g_max = static_cast<std::size_t>(gg);
          break;
        }
      }

      if (g == 0) {
        packing_(g, 0) = 0;
      } else {
        packing_(g, 0) =
            packing_(g - 1, 0) + packing_(g - 1, 2) + 1 - packing_(g - 1, 1);
      }
      packing_(g, 1) = static_cast<std::uint32_t>(g_min);
      packing_(g, 2) = static_cast<std::uint32_t>(g_max);
    }

    const std::size_t NDAT =
        packing_(NG - 1, 0) + packing_(NG - 1, 2) + 1 - packing_(NG - 1, 1);
    xs_ = xt::zeros<double>({static_cast<std::size_t>(1), NDAT});

    for (std::size_t g = 0; g < NG; g++) {
      const auto strt = packing_(g, 0);
      const auto g_min = packing_(g, 1);
      const auto g_max = packing_(g, 2);

      for (std::size_t gg = g_min; gg <= g_max; gg++) {
        xs_(0, strt + gg - g_min) += Es(g, gg);
      }
    }
  }

  XS2D(const xt::xtensor<double, 3>& Es) : xs_(), packing_() {
    if (Es.shape()[1] != Es.shape()[2]) {
      const auto mssg = "Scattering matrix is not square.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }
    const std::size_t NG = Es.shape()[1];

    packing_.resize({NG, 3});
    packing_.fill(0);

    for (std::size_t g = 0; g < NG; g++) {
      // Get the min and max outgoing group for each incident group
      std::size_t g_min = 0;
      for (std::size_t gg = 0; gg < NG; gg++) {
        if (Es(0, g, gg) != 0.) {
          g_min = gg;
          break;
        }
      }

      std::size_t g_max = NG - 1;
      for (int gg = static_cast<int>(NG) - 1; gg >= 0; gg--) {
        if (Es(0, g, static_cast<std::size_t>(gg)) != 0.) {
          g_max = static_cast<std::size_t>(gg);
          break;
        }
      }

      if (g == 0) {
        packing_(g, 0) = 0;
      } else {
        packing_(g, 0) =
            packing_(g - 1, 0) + packing_(g - 1, 2) + 1 - packing_(g - 1, 1);
      }
      packing_(g, 1) = static_cast<std::uint32_t>(g_min);
      packing_(g, 2) = static_cast<std::uint32_t>(g_max);
    }

    const std::size_t NDAT =
        packing_(NG - 1, 0) + packing_(NG - 1, 2) + 1 - packing_(NG - 1, 1);
    xs_ = xt::zeros<double>({Es.shape()[0], NDAT});

    for (std::size_t l = 0; l < Es.shape()[0]; l++) {
      for (std::size_t g = 0; g < NG; g++) {
        const auto strt = packing_(g, 0);
        const auto g_min = packing_(g, 1);
        const auto g_max = packing_(g, 2);

        for (std::size_t gg = g_min; gg <= g_max; gg++) {
          xs_(l, strt + gg - g_min) += Es(l, g, gg);
        }
      }
    }
  }

  std::size_t ngroups() const { return packing_.shape()[0]; }

  bool anisotropic() const { return xs_.shape()[0] > 1; }

  std::size_t max_legendre_order() const { return xs_.shape()[0] - 1; }

  std::string to_string() const {
    std::stringstream out;
    out << std::scientific;
    out << '[';
    for (std::size_t l = 0; l < max_legendre_order() + 1; l++) {
      out << '[';
      for (std::size_t g = 0; g < ngroups(); g++) {
        out << '[';
        for (std::size_t gg = 0; gg < ngroups(); gg++) {
          out << (*this)(l, g, gg);
          if (gg < ngroups() - 1) out << ", ";
        }
        out << ']';
        if (g < ngroups() - 1) out << ",\n  ";
      }
      out << ']';
      if (l != max_legendre_order()) out << ",\n ";
    }
    out << ']';

    return out.str();
  }

  double operator()(const std::size_t l, const std::size_t g) const {
    if (g >= ngroups()) {
      const auto mssg = "Incident group index out of range.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    if (l > this->max_legendre_order()) {
      return 0.;
    }

    const std::size_t strt = packing_(g, 0);
    const std::size_t g_min = packing_(g, 1);
    const std::size_t g_max = packing_(g, 2);
    const std::size_t max = strt + 1 + g_max - g_min;

    double Es = 0.;
    for (std::size_t g = strt; g < max; g++) {
      Es += xs_(l, g);
    }

    return Es;
  }

  double operator()(const std::size_t l, const std::size_t gin,
                    const std::size_t gout) const {
    if (gin >= ngroups()) {
      const auto mssg = "Incident group index out of range.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    if (gout >= ngroups()) {
      const auto mssg = "Outgoing group index out of range.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    if (l > this->max_legendre_order()) {
      return 0.;
    }

    const std::size_t strt = packing_(gin, 0);
    const std::size_t g_min = packing_(gin, 1);
    const std::size_t g_max = packing_(gin, 2);

    if ((gout < g_min) || (gout > g_max)) return 0.;

    return xs_(l, strt + gout - g_min);
  }

  void set_value(const std::size_t l, const std::size_t gin,
                 const std::size_t gout, double v) {
    if (l > this->max_legendre_order()) {
      const auto mssg = "Legendre order index out of range.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    if (gin >= ngroups()) {
      const auto mssg = "Incident group index out of range.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    if (gout >= ngroups()) {
      const auto mssg = "Outgoing group index out of range.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    const std::size_t strt = packing_(gin, 0);
    const std::size_t g_min = packing_(gin, 1);
    const std::size_t g_max = packing_(gin, 2);

    if ((gout < g_min) || (gout > g_max)) {
      const auto mssg =
          "The desired incident group -> outgoing group pair is not in the "
          "sparse matrix.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    xs_(l, strt + gout - g_min) = v;
  }

  void repack_to_be_compatible(const xt::xtensor<std::uint32_t, 2>& packing2) {
    if (ngroups() != packing2.shape()[0]) {
      const auto mssg = "Data packing schemes have different number of groups.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    if (packing2.shape()[1] != 3) {
      const auto mssg = "Provided packing scheme does not have a valid shape.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    // First, check if the two packing strategies are the same, or already
    // compatible. If so, we don't need to do anything.
    bool not_compat = false;
    for (std::size_t g = 0; g < ngroups(); g++) {
      if (packing_(g, 1) > packing2(g, 1)) {
        not_compat = true;
        break;
      } else if (packing_(g, 2) < packing2(g, 2)) {
        not_compat = true;
        break;
      }
    }
    if (not_compat == false) return;

    // Unfortunately, they are not compatible. We need to re-pack the data.
    auto new_data = packing_;

    for (std::size_t g = 0; g < ngroups(); g++) {
      // Set start index based on previous group data
      if (g == 0) {
        new_data(g, 0) = 0;
      } else {
        new_data(g, 0) =
            new_data(g - 1, 0) + new_data(g - 1, 2) + 1 - new_data(g - 1, 1);
      }

      // Find the new min and max outgoing groups for incident group g
      new_data(g, 1) = std::min(packing_(g, 1), packing2(g, 1));
      new_data(g, 2) = std::max(packing_(g, 2), packing2(g, 2));
    }

    // Get the required length of the data array
    const std::size_t NDAT = new_data(ngroups() - 1, 0) +
                             new_data(ngroups() - 1, 2) + 1 -
                             new_data(ngroups() - 1, 1);

    // Allocate new array for data
    xt::xtensor<double, 2> new_xs = xt::zeros<double>({xs_.shape()[0], NDAT});

    // Fill new array
    for (std::size_t l = 0; l <= max_legendre_order(); l++) {
      for (std::size_t g = 0; g < ngroups(); g++) {
        const auto strt = new_data(g, 0);
        const auto g_min = new_data(g, 1);
        const auto g_max = new_data(g, 2);

        for (std::size_t gg = g_min; gg <= g_max; gg++) {
          new_xs(l, strt + gg - g_min) = (*this)(l, g, gg);
        }
      }
    }

    xs_ = new_xs;
    packing_ = new_data;
  }

  const xt::xtensor<std::uint32_t, 2>& packing() const { return packing_; }

  XS2D zeros_like() const {
    XS2D out(*this);
    out.xs_.fill(0.);
    return out;
  }

  XS2D& operator+=(const XS2D& xs2) {
    if (ngroups() != xs2.ngroups()) {
      const auto mssg =
          "Cross section matrices have different number of groups.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    // Make the packing format compatible
    this->repack_to_be_compatible(xs2.packing_);

    // Must deal with unequal legendre orders
    if (max_legendre_order() < xs2.max_legendre_order()) {
      const std::size_t NL = std::max(this->xs_.shape()[0], xs2.xs_.shape()[0]);
      xt::xtensor<double, 2> new_xs = xt::zeros<double>({NL, xs_.shape()[1]});

      xt::view(new_xs, xt::range(0, xs_.shape()[0]), xt::all()) = xs_;
      xs_ = new_xs;
    }

    // Shapes are now completely equivalent.
    for (std::size_t l = 0; l <= max_legendre_order(); l++) {
      for (std::size_t g = 0; g < ngroups(); g++) {
        const auto strt = packing_(g, 0);
        const auto g_min = packing_(g, 1);
        const auto g_max = packing_(g, 2);

        for (std::size_t gg = g_min; gg <= g_max; gg++) {
          xs_(l, strt + gg - g_min) += xs2(l, g, gg);
        }
      }
    }

    return *this;
  }

  XS2D& operator-=(const XS2D& xs2) {
    if (ngroups() != xs2.ngroups()) {
      const auto mssg =
          "Cross section matrices have different number of groups.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    // Make the packing format compatible
    this->repack_to_be_compatible(xs2.packing_);

    // Must deal with unequal legendre orders
    if (max_legendre_order() < xs2.max_legendre_order()) {
      const std::size_t NL = std::max(xs_.shape()[0], xs2.xs_.shape()[0]);
      xt::xtensor<double, 2> new_xs = xt::zeros<double>({NL, xs_.shape()[1]});

      xt::view(new_xs, xt::range(0, xs_.shape()[0]), xt::all()) = xs_;
      xs_ = new_xs;
    }

    // Shapes are now completely equivalent.
    for (std::size_t l = 0; l <= max_legendre_order(); l++) {
      for (std::size_t g = 0; g < ngroups(); g++) {
        const auto strt = packing_(g, 0);
        const auto g_min = packing_(g, 1);
        const auto g_max = packing_(g, 2);

        for (std::size_t gg = g_min; gg <= g_max; gg++) {
          xs_(l, strt + gg - g_min) -= xs2(l, g, gg);
        }
      }
    }

    return *this;
  }

  XS2D operator+(const XS2D& xs2) {
    XS2D out(*this);
    out += xs2;
    return out;
  }

  XS2D operator-(const XS2D& xs2) {
    XS2D out(*this);
    out += xs2;
    return out;
  }

  XS2D& operator*=(const double v) {
    xs_ *= v;
    return *this;
  }

  XS2D& operator/=(const double v) {
    xs_ /= v;
    return *this;
  }

  XS2D operator*(const double v) const {
    XS2D out(*this);
    out *= v;
    return out;
  }

  XS2D operator/(const double v) const {
    XS2D out(*this);
    out /= v;
    return out;
  }

 private:
  xt::xtensor<double, 2> xs_;
  xt::xtensor<std::uint32_t, 2>
      packing_;  // First index incident group, second (data start, g_low, g_hi)

  friend class cereal::access;
  friend struct ::XS2DPickler;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(xs_), CEREAL_NVP(packing_));
  }
};

inline XS2D operator*(const double v, const XS2D& xs) {
  XS2D out(xs);
  out *= v;
  return out;
}

}  // namespace scarabee

#endif
