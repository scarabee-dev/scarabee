#ifndef SCARABEE_NEM4_H
#define SCARABEE_NEM4_H

#include <diffusion/nodal_method.hpp>
#include <data/diffusion_cross_section.hpp>

#include <Eigen/Sparse>

#include <array>
#include <cstddef>
#include <span>

namespace scarabee {

class NEM4 {
 public:
  NEM4(std::size_t NG) : M_{}, Q_{}, NG_{NG} {
    M_.resize(8 * NG, 8 * NG);
    Q_.resize(8 * NG);
  }

  static constexpr bool update_currents{true};

  void compute_keff_nonlinear_diffusion_coefficient(
      std::span<const Node> lnode, const Side side, std::span<const Node> rnode,
      std::span<const double> D, const DiffusionCrossSection& lxs,
      const DiffusionCrossSection& rxs, const std::array<double, 3> ld,
      const std::array<double, 3> rd, std::span<double> Dnl,
      const double invs_keff) {
    const double invs_ldx = 1. / get_node_width(ld, side);
    const double invs_rdx = 1. / get_node_width(rd, side);

    M_.resize(8 * NG_, 8 * NG_);
    M_.setZero();
    Q_.resize(8 * NG_);
    Q_.setZero();

    // We have 8 equations for each energy group
    int e = 0;
    for (std::size_t g = 0; g < NG_; g++) {
      // Flux balance equation for left node
      fill_neutron_balance(lnode, side, lxs, ld, invs_keff, invs_ldx, g, e++,
                           0);
      // fill_neutron_leakage(lnode, side, lxs, invs_ldx, g, e++, 0);
      //  Flux balance equation for right node
      fill_neutron_balance(rnode, side, rxs, rd, invs_keff, invs_rdx, g, e++,
                           1);
      // fill_neutron_leakage(rnode, side, rxs, invs_rdx, g, e++, 1);

      // First moment for left node
      fill_first_moment(lnode, side, lxs, ld, invs_keff, invs_ldx, g, e++, 0);
      // First moment for right node
      fill_first_moment(rnode, side, rxs, rd, invs_keff, invs_rdx, g, e++, 1);

      // Second moment for left node
      fill_second_moment(lnode, side, lxs, ld, invs_keff, invs_ldx, g, e++, 0);
      // Second moment for right node
      fill_second_moment(rnode, side, rxs, rd, invs_keff, invs_rdx, g, e++, 1);

      // Flux (dis)continuity condition
      fill_flux_discontinuity(lnode[g], rnode[g], side, g, e++);

      // Current continuity condition
      fill_current_continuity(lxs, rxs, invs_ldx, invs_rdx, g, e++);
    }

    // Solve system of equations
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver;
    solver.setTolerance(1.E-60);
    solver.compute(M_);
    auto A = solver.solve(Q_);

    // Compute current and nonlinear diffusion coefficient for each group
    for (std::size_t g = 0; g < NG_; g++) {
      const double J = -lxs.D(g) * invs_ldx *
                       (A(ind(g, 0, 1)) + 3. * A(ind(g, 0, 2)) +
                        0.5 * A(ind(g, 0, 3)) + 0.2 * A(ind(g, 0, 4)));
      const double phi_i = lnode[g].phi0();
      const double phi_i1 = rnode[g].phi0();

      Dnl[g] = -(J + D[g] * (phi_i1 - phi_i)) / (phi_i1 + phi_i);
    }
  }

  void compute_keff_nonlinear_diffusion_coefficient(
      std::span<const Node> node, const Side side, std::span<const double> D,
      double B, const DiffusionCrossSection& xs, const std::array<double, 3> d,
      std::span<double> Dnl, const double invs_keff) {
    const double invs_dx = 1. / get_node_width(d, side);
    M_.resize(4 * NG_, 4 * NG_);
    M_.setZero();
    Q_.resize(4 * NG_);
    Q_.setZero();

    // We have 8 equations for each energy group
    int e = 0;
    for (std::size_t g = 0; g < NG_; g++) {
      // Flux balance equation
      fill_neutron_balance(node, side, xs, d, invs_keff, invs_dx, g, e++, 0);
      // fill_neutron_leakage(node, side, xs, invs_dx, g, e++, 0);

      // First moment for left node
      fill_first_moment(node, side, xs, d, invs_keff, invs_dx, g, e++, 0);

      // Second moment for left node
      fill_second_moment(node, side, xs, d, invs_keff, invs_dx, g, e++, 0);

      // Albedo boundary condition
      fill_albedo_bc(node[g], xs, invs_dx, side, B, g, e++);
    }

    // Solve system of equations
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver;
    solver.setTolerance(1.E-60);
    solver.compute(M_);
    auto A = solver.solve(Q_);

    // Compute current and nonlinear diffusion coefficient for each group
    for (std::size_t g = 0; g < NG_; g++) {
      const double R = xs.D(g) * invs_dx;
      if (side == Side::XP || side == Side::YP || side == Side::ZP) {
        // See Eq. 22a in [2]
        const double J = -R * (A(ind(g, 0, 1)) + 3. * A(ind(g, 0, 2)) +
                               0.5 * A(ind(g, 0, 3)) + 0.2 * A(ind(g, 0, 4)));
        // See Eq. 2.141 in [1]
        Dnl[g] = -(J - D[g] * node[g].phi0()) / node[g].phi0();
      } else {
        // See Eq. 22b in [2]
        const double J = -R * (A(ind(g, 0, 1)) - 3. * A(ind(g, 0, 2)) +
                               0.5 * A(ind(g, 0, 3)) - 0.2 * A(ind(g, 0, 4)));
        // See Eq. 2.142 in [2]
        Dnl[g] = -(J + D[g] * node[g].phi0()) / node[g].phi0();
      }
    }
  }

 private:
  Eigen::SparseMatrix<double, Eigen::RowMajor> M_;
  Eigen::VectorXd Q_;
  std::size_t NG_;

  int ind(std::size_t g, int n, int a) const {
    return n * (static_cast<int>(NG_) * 4) + static_cast<int>(g) * 4 + (a - 1);
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

  void fill_neutron_balance(std::span<const Node>& node, Side side,
                            const DiffusionCrossSection& xs,
                            const std::array<double, 3>& d,
                            const double invs_keff, const double invs_dx,
                            std::size_t g, const int e, const int n) {
    // Flux balance equation. See Eq. 2.30 in [1]
    M_.coeffRef(e, ind(g, n, 2)) -= 6. * xs.D(g) * invs_dx * invs_dx;
    M_.coeffRef(e, ind(g, n, 4)) -= (2. * xs.D(g) / 5.) * invs_dx * invs_dx;
    const double phi_g = node[g].phi0();
    Q_(e) = -xs.Er(g) * phi_g - get_L(node[g], side, d);

    const double chi_g = xs.chi(g);
    for (std::size_t gg = 0; gg < NG_; gg++) {
      const double phi_gg = node[gg].phi0();
      if (gg != g) Q_(e) += xs.Es(gg, g) * phi_gg;
      Q_(e) += chi_g * invs_keff * xs.vEf(gg) * phi_gg;
    }
  }

  void fill_first_moment(std::span<const Node>& node, Side side,
                         const DiffusionCrossSection& xs,
                         const std::array<double, 3>& d, const double invs_keff,
                         const double invs_dx, std::size_t g, const int e,
                         const int n) {
    // First residiual moment equation. See Eq. 2.31 in [1]
    M_.coeffRef(e, ind(g, n, 3)) -= 0.5 * xs.D(g) * invs_dx * invs_dx;
    M_.coeffRef(e, ind(g, n, 1)) += xs.Er(g) / 12.;
    M_.coeffRef(e, ind(g, n, 3)) -= xs.Er(g) / 120.;
    Q_(e) = -get_rho1(node[g], side, d) / 12.;

    const double chi_g = xs.chi(g);
    for (std::size_t gg = 0; gg < NG_; gg++) {
      if (gg != g) {
        M_.coeffRef(e, ind(gg, n, 1)) -= xs.Es(gg, g) / 12.;
        M_.coeffRef(e, ind(gg, n, 3)) += xs.Es(gg, g) / 120.;
      }
      M_.coeffRef(e, ind(gg, n, 1)) -= chi_g * invs_keff * xs.vEf(gg) / 12.;
      M_.coeffRef(e, ind(gg, n, 3)) += chi_g * invs_keff * xs.vEf(gg) / 120.;
    }
  }

  void fill_second_moment(std::span<const Node>& node, Side side,
                          const DiffusionCrossSection& xs,
                          const std::array<double, 3>& d,
                          const double invs_keff, const double invs_dx,
                          std::size_t g, const int e, const int n) {
    // Second residiual moment equation. See Eq. 2.32 in [1]
    M_.coeffRef(e, ind(g, n, 4)) -= 0.2 * xs.D(g) * invs_dx * invs_dx;
    M_.coeffRef(e, ind(g, n, 2)) += xs.Er(g) / 20.;
    M_.coeffRef(e, ind(g, n, 4)) -= xs.Er(g) / 700.;
    Q_(e) = -get_rho2(node[g], side, d) / 20.;

    const double chi_g = xs.chi(g);
    for (std::size_t gg = 0; gg < NG_; gg++) {
      if (gg != g) {
        M_.coeffRef(e, ind(gg, n, 2)) -= xs.Es(gg, g) / 20.;
        M_.coeffRef(e, ind(gg, n, 4)) += xs.Es(gg, g) / 700.;
      }
      M_.coeffRef(e, ind(gg, n, 2)) -= chi_g * invs_keff * xs.vEf(gg) / 20.;
      M_.coeffRef(e, ind(gg, n, 4)) += chi_g * invs_keff * xs.vEf(gg) / 700.;
    }
  }

  void fill_flux_discontinuity(const Node& lnode, const Node& rnode, Side side,
                               std::size_t g, int e) {
    // See Eq. 2.46 in [1]
    const double r = get_op_adf(rnode, side) / get_adf(lnode, side);

    M_.coeffRef(e, ind(g, 0, 1)) += 0.5;
    M_.coeffRef(e, ind(g, 0, 2)) += 0.5;

    M_.coeffRef(e, ind(g, 1, 1)) += 0.5 * r;
    M_.coeffRef(e, ind(g, 1, 2)) -= 0.5 * r;

    Q_(e) = r * rnode.phi0() - lnode.phi0();
  }

  void fill_current_continuity(const DiffusionCrossSection& lxs,
                               const DiffusionCrossSection& rxs,
                               const double invs_ldx, const double invs_rdx,
                               std::size_t g, int e) {
    // See Eq. 2.47 in [1]
    const double Rl = lxs.D(g) * invs_ldx;
    const double Rr = rxs.D(g) * invs_rdx;

    M_.coeffRef(e, ind(g, 0, 1)) += Rl;
    M_.coeffRef(e, ind(g, 0, 2)) += 3. * Rl;
    M_.coeffRef(e, ind(g, 0, 3)) += 0.5 * Rl;
    M_.coeffRef(e, ind(g, 0, 4)) += 0.2 * Rl;

    M_.coeffRef(e, ind(g, 1, 1)) -= Rr;
    M_.coeffRef(e, ind(g, 1, 2)) += 3. * Rr;
    M_.coeffRef(e, ind(g, 1, 3)) -= 0.5 * Rr;
    M_.coeffRef(e, ind(g, 1, 4)) += 0.2 * Rr;

    Q_(e) = 0.;
  }

  void fill_albedo_bc(const Node& node, const DiffusionCrossSection& xs,
                      const double invs_dx, Side side, const double B,
                      std::size_t g, int e) {
    const double D = xs.D(g);
    const double R = D * invs_dx;
    const double G = 0.5 * ((1. - B) / (1. + B));
    if (side == Side::XP || side == Side::YP || side == Side::ZP) {
      /*
      // See Eq. 22a in [2]
      M_.coeffRef(e, ind(g, 0, 1)) = -R;
      M_.coeffRef(e, ind(g, 0, 2)) = -3. * R;
      M_.coeffRef(e, ind(g, 0, 3)) = -0.5 * R;
      M_.coeffRef(e, ind(g, 0, 4)) = -0.2 * R;
      Q_(e) = 1. - B;
      */

      // See Eq. 2.55 in [1]
      M_.coeffRef(e, ind(g, 0, 1)) = 0.5 * G + R;
      M_.coeffRef(e, ind(g, 0, 2)) = 0.5 * G + 3. * R;
      M_.coeffRef(e, ind(g, 0, 3)) = 0.5 * R;
      M_.coeffRef(e, ind(g, 0, 4)) = 0.2 * R;
      Q_(e) = -G * node.phi0();
    } else {
      /*
      // See Eq. 22b in [2]
      M_.coeffRef(e, ind(g, 0, 1)) = R;
      M_.coeffRef(e, ind(g, 0, 2)) = -3. * R;
      M_.coeffRef(e, ind(g, 0, 3)) = 0.5 * R;
      M_.coeffRef(e, ind(g, 0, 4)) = -0.2 * R;
      Q_(e) = 1. - B;
      */

      // See Eq. 2.54 in [1]
      M_.coeffRef(e, ind(g, 0, 1)) = -(0.5 * G + R);
      M_.coeffRef(e, ind(g, 0, 2)) = 0.5 * G + 3. * R;
      M_.coeffRef(e, ind(g, 0, 3)) = -0.5 * R;
      M_.coeffRef(e, ind(g, 0, 4)) = 0.2 * R;
      Q_(e) = -G * node.phi0();
    }
  }
};

}  // namespace scarabee

// [1] A. Hébert, "The BRISINGR Theory and User Guide", IGE-380
//
// [2] R. D. Lawrence, “Progress in nodal methods for the solution of the
//     neutron diffusion and transport equations,” Prog Nucl Energ, vol. 17,
//     no. 3, pp. 271–301, 1986, doi: 10.1016/0149-1970(86)90034-x.

#endif
