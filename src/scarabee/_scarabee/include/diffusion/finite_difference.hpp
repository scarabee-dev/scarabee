#ifndef SCARABEE_FINITE_DIFFERENCE_H
#define SCARABEE_FINITE_DIFFERENCE_H

#include <diffusion/nodal_method.hpp>
#include <diffusion/node.hpp>
#include <data/diffusion_cross_section.hpp>

#include <cereal/cereal.hpp>

#include <array>
#include <cstddef>
#include <cmath>
#include <span>

namespace scarabee {

class FiniteDifference {
 public:
  FiniteDifference() = default;
  FiniteDifference(std::size_t /*NG*/) {}

  static constexpr bool update_currents{false};
  static constexpr bool reconstruct_flux{false};

  double compute_keff_nonlinear_diffusion_coefficient(
      std::span<Node> lnode, const Side side, std::span<Node> rnode,
      std::span<const double> D, const DiffusionCrossSection& lxs,
      const DiffusionCrossSection& rxs, const std::array<double, 3> ld,
      const std::array<double, 3> rd, std::span<double> Dnl,
      const double /*invs_keff*/) {
    // The formula used in the loop below can be derived from setting the last
    // equation in [1] to be equal to the CMFD current. This allows us to
    // derive an expression for the nonlinear diffusion coefficient that
    // preserves the flux discontinuities. If no ADFs are used, Dnl = 0.

    const double ldx = get_node_width(ld, side);
    const double rdx = get_node_width(rd, side);

    double max_diff = 0.;
    for (std::size_t g = 0; g < Dnl.size(); g++) {
      const double phi_l = lnode[g].phi0();
      const double phi_r = rnode[g].phi0();
      const double f_l = get_adf(lnode[g], side);
      const double f_r = get_op_adf(rnode[g], side);
      const double r = f_l / f_r;
      const double Dl = lxs.D(g);
      const double Dr = rxs.D(g);
      const double num =
          ((2. * Dl * Dr) / (rdx * Dr + r * ldx * Dl)) * (phi_r - r * phi_l) -
          D[g] * (phi_r - phi_l);
      const double denom = phi_l + phi_r;

      const double old_Dnlg = Dnl[g];
      Dnl[g] = num / denom;
      const double diff = std::abs((Dnl[g] - old_Dnlg) / Dnl[g]);
      if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
  }

  double compute_keff_nonlinear_diffusion_coefficient(
      std::span<Node> /*lnode*/, const Side /*side*/,
      std::span<const double> /*D*/, double /*B*/,
      const DiffusionCrossSection& /*lxs*/, const std::array<double, 3> /*ld*/,
      std::span<double> Dnl, const double /*invs_keff*/) {
    for (auto& Dg : Dnl) Dg = 0.;
    return 0.;
  }

 private:
  double get_adf(const Node& n, Side s) const {
    switch (s) {
      case Side::XN:
        return n.adf_xn();
      case Side::XP:
        return n.adf_xp();
      case Side::YN:
        return n.adf_yn();
      case Side::YP:
        return n.adf_yp();
      case Side::ZN:
        return n.adf_zn();
      case Side::ZP:
        return n.adf_zp();
      default:
        return 1;  // Should never get here
    }
  }

  double get_op_adf(const Node& n, Side s) const {
    switch (s) {
      case Side::XN:
        return n.adf_xp();
      case Side::XP:
        return n.adf_xn();
      case Side::YN:
        return n.adf_yp();
      case Side::YP:
        return n.adf_yn();
      case Side::ZN:
        return n.adf_zp();
      case Side::ZP:
        return n.adf_zn();
      default:
        return 1.;  // Should never get here
    }
  }

  double get_node_width(const std::array<double, 3>& dx, Side side) const {
    switch (side) {
      case Side::XP:
      case Side::XN:
        return dx[0];
      case Side::YP:
      case Side::YN:
        return dx[1];
      case Side::ZP:
      case Side::ZN:
        return dx[2];
      default:
        return dx[0];  // Should never get here !
    }
  }

 private:
  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& /*arc*/) {}
};

}  // namespace scarabee

// [1] R. Sanchez, G. Dante, and I. Zmijarevic, “DIFFUSION PIECEWISE
//     HOMOGENIZATION VIA FLUX DISCONTINUITY RATIOS,” Nucl. Eng. Technol.,
//     vol. 45, no. 6, pp. 707–720, 2013, doi: 10.5516/net.02.2013.518.

#endif
