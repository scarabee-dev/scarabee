#ifndef SCARABEE_SANM_H
#define SCARABEE_SANM_H

#include <diffusion/nodal_method.hpp>
#include <diffusion/node.hpp>
#include <data/diffusion_cross_section.hpp>
#include <utils/serialization.hpp>

#include <Eigen/Dense>

#include <cereal/cereal.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <span>

namespace scarabee {

class SANM {
 private:
  using Matrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vector = Eigen::VectorXd;
  using Solver = Eigen::PartialPivLU<Matrix>;

 public:
  SANM(std::size_t NG)
      : M1n_(4 * NG, 4 * NG),
        M2n_(8 * NG, 8 * NG),
        Q1n_(4 * NG),
        Q2n_(8 * NG),
        A1n_(4 * NG),
        A2n_(8 * NG),
        eta_l_(NG),
        eta_r_(NG),
        NG_{NG} {}

  static constexpr bool update_currents{true};
  static constexpr bool reconstruct_flux{true};

  double compute_keff_nonlinear_diffusion_coefficient(
      std::span<Node> lnode, const Side side, std::span<Node> rnode,
      std::span<const double> D, const DiffusionCrossSection& lxs,
      const DiffusionCrossSection& rxs, const std::array<double, 3> ld,
      const std::array<double, 3> rd, std::span<double> Dnl,
      const double invs_keff) {
    const double ldx = get_node_width(ld, side);
    const double rdx = get_node_width(rd, side);
    const double invs_ldx = 1. / ldx;
    const double invs_rdx = 1. / rdx;
    M2n_.setZero();
    Q2n_.setZero();
    eta_l_.setZero();
    eta_r_.setZero();

    // First, get the eta values for each node in each group
    for (std::size_t g = 0; g < NG_; g++) {
      const Vector::Index ig = static_cast<Vector::Index>(g);
      eta_l_(ig) = ldx * std::sqrt(lxs.Er(g) / lxs.D(g));
      eta_r_(ig) = rdx * std::sqrt(rxs.Er(g) / rxs.D(g));
    }

    // We have 8 equations for each energy group
    std::size_t e = 0;
    for (std::size_t g = 0; g < NG_; g++) {
      const Vector::Index ig = static_cast<Vector::Index>(g);
      // Flux balance equation for left node
      fill_neutron_balance(M2n_, Q2n_, lnode, side, lxs, ld, eta_l_, invs_keff,
                           invs_ldx, g, e++, 0);
      //  Flux balance equation for right node
      fill_neutron_balance(M2n_, Q2n_, rnode, side, rxs, rd, eta_r_, invs_keff,
                           invs_rdx, g, e++, 1);

      // First moment for left node
      fill_first_moment(M2n_, Q2n_, lnode, side, lxs, ld, eta_l_, invs_keff, g,
                        e++, 0);
      // First moment for right node
      fill_first_moment(M2n_, Q2n_, rnode, side, rxs, rd, eta_r_, invs_keff, g,
                        e++, 1);

      // Second moment for left node
      fill_second_moment(M2n_, Q2n_, lnode, side, lxs, ld, eta_l_, invs_keff, g,
                         e++, 0);
      // Second moment for right node
      fill_second_moment(M2n_, Q2n_, rnode, side, rxs, rd, eta_r_, invs_keff, g,
                         e++, 1);

      // Flux (dis)continuity condition
      fill_flux_discontinuity(M2n_, Q2n_, lnode[g], rnode[g], side, eta_l_(ig),
                              eta_r_(ig), g, e++);

      // Current continuity condition
      fill_current_continuity(M2n_, Q2n_, lxs, rxs, eta_l_(ig), eta_r_(ig),
                              invs_ldx, invs_rdx, g, e++);
    }

    // Solve system of equations
    Solver solver;
    solver.compute(M2n_);
    A2n_ = solver.solve(Q2n_);
    const auto& A = A2n_;

    // Compute current and nonlinear diffusion coefficient for each group
    double max_diff = 0.;
    for (std::size_t g = 0; g < NG_; g++) {
      const Vector::Index ig = static_cast<Vector::Index>(g);
      const double nl = eta_l_(ig);
      const double nr = eta_r_(ig);

      const double J = -lxs.D(g) * invs_ldx *
                       (A(ind(g, 0, 1)) + 3. * A(ind(g, 0, 2)) +
                        nl * std::cosh(0.5 * nl) * A(ind(g, 0, 3)) +
                        nl * std::sinh(0.5 * nl) * A(ind(g, 0, 4)));
      const double phi_i = lnode[g].phi0();
      const double phi_i1 = rnode[g].phi0();
      const double old_Dnlg = Dnl[g];

      Dnl[g] = -(J + D[g] * (phi_i1 - phi_i)) / (phi_i1 + phi_i);
      const double diff = std::abs(Dnl[g] - old_Dnlg);
      if (diff > max_diff) max_diff = diff;

      // Compute surface flux for each side
      const double lnode_phi_p = lnode[g].phi0() + 0.5 * A(ind(g, 0, 1)) +
                                 0.5 * A(ind(g, 0, 2)) +
                                 std::sinh(0.5 * nl) * A(ind(g, 0, 3)) +
                                 (alpha_1(nl) / nl) * A(ind(g, 0, 4));
      const double rnode_phi_n = rnode[g].phi0() - 0.5 * A(ind(g, 1, 1)) +
                                 0.5 * A(ind(g, 1, 2)) -
                                 std::sinh(0.5 * nr) * A(ind(g, 1, 3)) +
                                 (alpha_1(nr) / nr) * A(ind(g, 1, 4));
      switch (side) {
        case Side::XN:
        case Side::XP:
          lnode[g].phi_xp() = lnode_phi_p;
          rnode[g].phi_xn() = rnode_phi_n;
          break;
        case Side::YN:
        case Side::YP:
          lnode[g].phi_yp() = lnode_phi_p;
          rnode[g].phi_yn() = rnode_phi_n;
          break;
        case Side::ZN:
        case Side::ZP:
          lnode[g].phi_zp() = lnode_phi_p;
          rnode[g].phi_zn() = rnode_phi_n;
          break;
      }
    }
    return max_diff;
  }

  double compute_keff_nonlinear_diffusion_coefficient(
      std::span<Node> node, const Side side, std::span<const double> D,
      double B, const DiffusionCrossSection& xs, const std::array<double, 3> d,
      std::span<double> Dnl, const double invs_keff) {
    const double dx = get_node_width(d, side);
    const double invs_dx = 1. / dx;
    M1n_.setZero();
    Q1n_.setZero();
    eta_l_.setZero();

    // First, get the eta values for each node in each group
    for (std::size_t g = 0; g < NG_; g++) {
      const Vector::Index ig = static_cast<Vector::Index>(g);
      eta_l_(ig) = dx * std::sqrt(xs.Er(g) / xs.D(g));
    }

    // We have 4 equations for each energy group
    std::size_t e = 0;
    for (std::size_t g = 0; g < NG_; g++) {
      const Vector::Index ig = static_cast<Vector::Index>(g);

      // Flux balance equation
      fill_neutron_balance(M1n_, Q1n_, node, side, xs, d, eta_l_, invs_keff,
                           invs_dx, g, e++, 0);

      // First moment for left node
      fill_first_moment(M1n_, Q1n_, node, side, xs, d, eta_l_, invs_keff, g,
                        e++, 0);

      // Second moment for left node
      fill_second_moment(M1n_, Q1n_, node, side, xs, d, eta_l_, invs_keff, g,
                         e++, 0);

      // Albedo boundary condition
      fill_albedo_bc(M1n_, Q1n_, node[g], xs, eta_l_(ig), invs_dx, side, B, g,
                     e++);
    }

    // Solve system of equations
    Solver solver;
    solver.compute(M1n_);
    A1n_ = solver.solve(Q1n_);
    const auto& A = A1n_;

    // Compute current and nonlinear diffusion coefficient for each group
    double max_diff = 0.;
    for (std::size_t g = 0; g < NG_; g++) {
      const Vector::Index ig = static_cast<Vector::Index>(g);
      const double n = eta_l_(ig);
      const double old_Dnlg = Dnl[g];

      const double R = xs.D(g) * invs_dx;
      if (side == Side::XP || side == Side::YP || side == Side::ZP) {
        // See Eq. 2.45 in [1]
        const double J = -R * (A(ind(g, 0, 1)) + 3. * A(ind(g, 0, 2)) +
                               n * std::cosh(0.5 * n) * A(ind(g, 0, 3)) +
                               n * std::sinh(0.5 * n) * A(ind(g, 0, 4)));
        // See Eq. 2.141 in [1]
        Dnl[g] = -(J - D[g] * node[g].phi0()) / node[g].phi0();
      } else {
        // See Eq. 2.44 in [1]
        const double J = -R * (A(ind(g, 0, 1)) - 3. * A(ind(g, 0, 2)) +
                               n * std::cosh(0.5 * n) * A(ind(g, 0, 3)) -
                               n * std::sinh(0.5 * n) * A(ind(g, 0, 4)));
        // See Eq. 2.142 in [1]
        Dnl[g] = -(J + D[g] * node[g].phi0()) / node[g].phi0();
      }

      const double diff = std::abs(Dnl[g] - old_Dnlg);
      if (diff > max_diff) max_diff = diff;

      // Compute surface flux for side being treated
      switch (side) {
        case Side::XN:
          node[g].phi_xn() = node[g].phi0() - 0.5 * A(ind(g, 0, 1)) +
                             0.5 * A(ind(g, 0, 2)) -
                             std::sinh(0.5 * n) * A(ind(g, 0, 3)) +
                             (alpha_1(n) / n) * A(ind(g, 0, 4));
          break;
        case Side::XP:
          node[g].phi_xp() = node[g].phi0() + 0.5 * A(ind(g, 0, 1)) +
                             0.5 * A(ind(g, 0, 2)) +
                             std::sinh(0.5 * n) * A(ind(g, 0, 3)) +
                             (alpha_1(n) / n) * A(ind(g, 0, 4));
          break;
        case Side::YN:
          node[g].phi_yn() = node[g].phi0() - 0.5 * A(ind(g, 0, 1)) +
                             0.5 * A(ind(g, 0, 2)) -
                             std::sinh(0.5 * n) * A(ind(g, 0, 3)) +
                             (alpha_1(n) / n) * A(ind(g, 0, 4));
          break;
        case Side::YP:
          node[g].phi_yp() = node[g].phi0() + 0.5 * A(ind(g, 0, 1)) +
                             0.5 * A(ind(g, 0, 2)) +
                             std::sinh(0.5 * n) * A(ind(g, 0, 3)) +
                             (alpha_1(n) / n) * A(ind(g, 0, 4));
          break;
        case Side::ZN:
          node[g].phi_zn() = node[g].phi0() - 0.5 * A(ind(g, 0, 1)) +
                             0.5 * A(ind(g, 0, 2)) -
                             std::sinh(0.5 * n) * A(ind(g, 0, 3)) +
                             (alpha_1(n) / n) * A(ind(g, 0, 4));
          break;
        case Side::ZP:
          node[g].phi_zp() = node[g].phi0() + 0.5 * A(ind(g, 0, 1)) +
                             0.5 * A(ind(g, 0, 2)) +
                             std::sinh(0.5 * n) * A(ind(g, 0, 3)) +
                             (alpha_1(n) / n) * A(ind(g, 0, 4));
          break;
      }
    }
    return max_diff;
  }

 private:
  Matrix M1n_, M2n_;
  Vector Q1n_, Q2n_, A1n_, A2n_;
  Vector eta_l_, eta_r_;
  std::size_t NG_;

  Eigen::Index ind(std::size_t g, std::size_t n, std::size_t a) const {
    return static_cast<Eigen::Index>(n * (NG_ * 4) + g * 4 + (a - 1));
  }

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

  double get_L(const Node& node, const Side side,
               const std::array<double, 3>& d) {
    switch (side) {
      case Side::XP:
      case Side::XN: {
        const double Ly = node.J_yp() - node.J_yn();
        const double Lz = node.J_zp() - node.J_zn();
        return (Ly / d[1]) + (Lz / d[2]);
      }
      case Side::YP:
      case Side::YN: {
        const double Lx = node.J_xp() - node.J_xn();
        const double Lz = node.J_zp() - node.J_zn();
        return (Lx / d[0]) + (Lz / d[2]);
      }
      case Side::ZP:
      case Side::ZN: {
        const double Lx = node.J_xp() - node.J_xn();
        const double Ly = node.J_yp() - node.J_yn();
        return (Lx / d[0]) + (Ly / d[1]);
      }
      default:
        return 0.;
    }
  }

  double get_rho1(const Node& node, const Side side,
                  const std::array<double, 3>& d) {
    switch (side) {
      case Side::XP:
      case Side::XN: {
        return node.Lx_rho_y1() / d[1] + node.Lx_rho_z1() / d[2];
      }
      case Side::YP:
      case Side::YN: {
        return node.Ly_rho_x1() / d[0] + node.Ly_rho_z1() / d[2];
      }
      case Side::ZP:
      case Side::ZN: {
        return node.Lz_rho_x1() / d[0] + node.Lz_rho_y1() / d[1];
      }
      default:
        return 0.;
    }
  }

  double get_rho2(const Node& node, const Side side,
                  const std::array<double, 3>& d) {
    switch (side) {
      case Side::XP:
      case Side::XN: {
        return node.Lx_rho_y2() / d[1] + node.Lx_rho_z2() / d[2];
      }
      case Side::YP:
      case Side::YN: {
        return node.Ly_rho_x2() / d[0] + node.Ly_rho_z2() / d[2];
      }
      case Side::ZP:
      case Side::ZN: {
        return node.Lz_rho_x2() / d[0] + node.Lz_rho_y2() / d[1];
      }
      default:
        return 0.;
    }
  }

  double alpha_0(const double eta) const {
    return 2. * eta * std::sinh(0.5 * eta);
  }

  double alpha_1(const double eta) const {
    return eta * std::cosh(0.5 * eta) - 2. * std::sinh(0.5 * eta);
  }

  double alpha_2(const double eta) const {
    return ((12. / eta) + eta) * std::sinh(0.5 * eta) -
           6. * std::cosh(0.5 * eta);
  }

  void fill_neutron_balance(Matrix& M, Vector& Q, std::span<const Node> node,
                            Side side, const DiffusionCrossSection& xs,
                            const std::array<double, 3>& d, const Vector& eta,
                            const double invs_keff, const double invs_dx,
                            const std::size_t g, const std::size_t e,
                            const std::size_t n) {
    // Flux balance equation. See Eq. 2.33 in [1]
    M.coeffRef(e, ind(g, n, 2)) -= 6. * xs.D(g) * invs_dx * invs_dx;
    M.coeffRef(e, ind(g, n, 4)) -=
        alpha_0(eta(g)) * xs.D(g) * invs_dx * invs_dx;
    const double phi_g = node[g].phi0();
    Q(e) = -xs.Er(g) * phi_g - get_L(node[g], side, d);

    const double chi_g = xs.chi(g);
    for (std::size_t gg = 0; gg < NG_; gg++) {
      const double phi_gg = node[gg].phi0();
      if (gg != g) Q(e) += xs.Es(gg, g) * phi_gg;
      Q(e) += chi_g * invs_keff * xs.vEf(gg) * phi_gg;
    }
  }

  void fill_first_moment(Matrix& M, Vector& Q, std::span<const Node> node,
                         Side side, const DiffusionCrossSection& xs,
                         const std::array<double, 3>& d, const Vector& eta,
                         const double invs_keff, const std::size_t g,
                         const std::size_t e, const std::size_t n) {
    // First residiual moment equation. See Eq. 2.34 in [1]
    M.coeffRef(e, ind(g, n, 1)) += xs.Er(g) / 12.;
    Q(e) = -get_rho1(node[g], side, d) / 12.;

    const double chi_g = xs.chi(g);
    for (std::size_t gg = 0; gg < NG_; gg++) {
      const double h = eta(gg);
      const double a1 = alpha_1(h);
      if (gg != g) {
        M.coeffRef(e, ind(gg, n, 1)) -= xs.Es(gg, g) / 12.;
        M.coeffRef(e, ind(gg, n, 3)) -= (a1 / (h * h)) * xs.Es(gg, g);
      }
      M.coeffRef(e, ind(gg, n, 1)) -= chi_g * invs_keff * xs.vEf(gg) / 12.;
      M.coeffRef(e, ind(gg, n, 3)) -=
          chi_g * invs_keff * (a1 / (h * h)) * xs.vEf(gg);
    }
  }

  void fill_second_moment(Matrix& M, Vector& Q, std::span<const Node> node,
                          Side side, const DiffusionCrossSection& xs,
                          const std::array<double, 3>& d, const Vector& eta,
                          const double invs_keff, const std::size_t g,
                          const std::size_t e, const std::size_t n) {
    // Second residiual moment equation. See Eq. 2.35 in [1]
    M.coeffRef(e, ind(g, n, 2)) += xs.Er(g) / 20.;
    Q(e) = -get_rho2(node[g], side, d) / 20.;

    const double chi_g = xs.chi(g);
    for (std::size_t gg = 0; gg < NG_; gg++) {
      const double h = eta(gg);
      const double a2 = alpha_2(h);
      if (gg != g) {
        M.coeffRef(e, ind(gg, n, 2)) -= xs.Es(gg, g) / 20.;
        M.coeffRef(e, ind(gg, n, 4)) -= (a2 / (h * h)) * xs.Es(gg, g);
      }
      M.coeffRef(e, ind(gg, n, 2)) -= chi_g * invs_keff * xs.vEf(gg) / 20.;
      M.coeffRef(e, ind(gg, n, 4)) -=
          chi_g * invs_keff * (a2 / (h * h)) * xs.vEf(gg);
    }
  }

  void fill_flux_discontinuity(Matrix& M, Vector& Q, const Node& lnode,
                               const Node& rnode, Side side, const double eta_l,
                               const double eta_r, const std::size_t g,
                               const std::size_t e) {
    // See Eq. 2.48 in [1]
    const double r = get_op_adf(rnode, side) / get_adf(lnode, side);

    M.coeffRef(e, ind(g, 0, 1)) += 0.5;
    M.coeffRef(e, ind(g, 0, 2)) += 0.5;
    M.coeffRef(e, ind(g, 0, 3)) += std::sinh(0.5 * eta_l);
    M.coeffRef(e, ind(g, 0, 4)) += alpha_1(eta_l) / eta_l;

    M.coeffRef(e, ind(g, 1, 1)) += 0.5 * r;
    M.coeffRef(e, ind(g, 1, 2)) -= 0.5 * r;
    M.coeffRef(e, ind(g, 1, 3)) += std::sinh(0.5 * eta_r) * r;
    M.coeffRef(e, ind(g, 1, 4)) -= alpha_1(eta_r) / eta_r * r;

    Q(e) = r * rnode.phi0() - lnode.phi0();
  }

  void fill_current_continuity(Matrix& M, Vector& Q,
                               const DiffusionCrossSection& lxs,
                               const DiffusionCrossSection& rxs,
                               const double eta_l, const double eta_r,
                               const double invs_ldx, const double invs_rdx,
                               const std::size_t g, const std::size_t e) {
    // See Eq. 2.49 in [1]
    const double Rl = lxs.D(g) * invs_ldx;
    const double Rr = rxs.D(g) * invs_rdx;

    M.coeffRef(e, ind(g, 0, 1)) += Rl;
    M.coeffRef(e, ind(g, 0, 2)) += 3. * Rl;
    M.coeffRef(e, ind(g, 0, 3)) += eta_l * std::cosh(0.5 * eta_l) * Rl;
    M.coeffRef(e, ind(g, 0, 4)) += eta_l * std::sinh(0.5 * eta_l) * Rl;

    M.coeffRef(e, ind(g, 1, 1)) -= Rr;
    M.coeffRef(e, ind(g, 1, 2)) += 3. * Rr;
    M.coeffRef(e, ind(g, 1, 3)) -= eta_r * std::cosh(0.5 * eta_r) * Rr;
    M.coeffRef(e, ind(g, 1, 4)) += eta_r * std::sinh(0.5 * eta_r) * Rr;

    Q(e) = 0.;
  }

  void fill_albedo_bc(Matrix& M, Vector& Q, const Node& node,
                      const DiffusionCrossSection& xs, const double eta,
                      const double invs_dx, Side side, const double B,
                      const std::size_t g, const std::size_t e) {
    const double D = xs.D(g);
    const double R = D * invs_dx;
    const double G = 0.5 * ((1. - B) / (1. + B));
    if (side == Side::XP || side == Side::YP || side == Side::ZP) {
      // See Eq. 2.58 in [1]
      M.coeffRef(e, ind(g, 0, 1)) = 0.5 * G + R;
      M.coeffRef(e, ind(g, 0, 2)) = 0.5 * G + 3. * R;
      M.coeffRef(e, ind(g, 0, 3)) =
          G * std::sinh(0.5 * eta) + R * eta * std::cosh(0.5 * eta);
      M.coeffRef(e, ind(g, 0, 4)) =
          G * alpha_1(eta) / eta + R * eta * std::sinh(0.5 * eta);
      Q(e) = -G * node.phi0();
    } else {
      // See Eq. 2.57 in [1]
      M.coeffRef(e, ind(g, 0, 1)) = -(0.5 * G + R);
      M.coeffRef(e, ind(g, 0, 2)) = 0.5 * G + 3. * R;
      M.coeffRef(e, ind(g, 0, 3)) =
          -G * std::sinh(0.5 * eta) - R * eta * std::cosh(0.5 * eta);
      M.coeffRef(e, ind(g, 0, 4)) =
          G * alpha_1(eta) / eta + R * eta * std::sinh(0.5 * eta);
      Q(e) = -G * node.phi0();
    }
  }

  friend class cereal::access;
  SANM() = default;

  template <class Archive>
  void save(Archive& arc) const {
    arc(NG_);
  }

  template <class Archive>
  void load(Archive& arc) {
    arc(NG_);
    M1n_.resize(4 * NG_, 4 * NG_);
    M2n_.resize(8 * NG_, 8 * NG_);
    Q1n_.resize(4 * NG_);
    Q2n_.resize(8 * NG_);
    A1n_.resize(4 * NG_);
    A2n_.resize(8 * NG_);
    eta_l_.resize(NG_);
    eta_r_.resize(NG_);
  }
};

}  // namespace scarabee

// [1] A. Hébert, "The BRISINGR Theory and User Guide", IGE-380

#endif
