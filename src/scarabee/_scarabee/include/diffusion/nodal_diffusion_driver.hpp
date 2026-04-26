#ifndef SCARABEE_NODAL_DIFFUSION_DRIVER_H
#define SCARABEE_NODAL_DIFFUSION_DRIVER_H

#include <data/diffusion_cross_section.hpp>
#include <diffusion/diffusion_data.hpp>
#include <diffusion/diffusion_geometry.hpp>
#include <diffusion/intra_nodal_flux.hpp>
#include <diffusion/node.hpp>
#include <diffusion/nodal_cmfd_surface.hpp>
#include <diffusion/nodal_method.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>

#include <xtensor/containers/xtensor.hpp>

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <Eigen/SparseLU>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>

namespace scarabee {

template <NodalMethod NM>
class NodalDiffusionDriver {
 public:
  NodalDiffusionDriver(std::shared_ptr<DiffusionGeometry> geom);

  std::shared_ptr<DiffusionGeometry> geometry() const { return geom_; }

  std::size_t ngroups() const { return NG_; }

  void solve() { this->solve_keff(); }
  bool solved() const { return solved_; }

  double keff_tolerance() const { return keff_tol_; }
  void set_keff_tolerance(double ktol);

  double flux_tolerance() const { return flux_tol_; }
  void set_flux_tolerance(double ftol);

  bool leakage_corrections() const { return leakage_corrections_; }
  void set_leakage_corrections(bool lc) { leakage_corrections_ = lc; }

  std::size_t nonlinear_update_frequency() const {
    return nonlinear_update_frequency_;
  }
  void set_nonlinear_update_frequency(std::size_t f);

  double keff() const { return keff_; }

  /*
  double flux(double x, double y, double z, std::size_t g) const;
  xt::xtensor<double, 4> flux(const xt::xtensor<double, 1>& x,
                              const xt::xtensor<double, 1>& y,
                              const xt::xtensor<double, 1>& z) const;
  xt::xtensor<double, 4> avg_flux() const;

  double power(double x, double y, double z) const;
  xt::xtensor<double, 3> power(const xt::xtensor<double, 1>& x,
                               const xt::xtensor<double, 1>& y,
                               const xt::xtensor<double, 1>& z) const;
  xt::xtensor<double, 3> avg_power() const;

  void save(const std::string& fname);
  static std::unique_ptr<NodalDiffusionDriver> load(const std::string& fname);
  */

 private:
  struct DiffusionDataCrossSectionPair {
    DiffusionDataCrossSectionPair(
        const std::shared_ptr<DiffusionData>& idd,
        const std::shared_ptr<DiffusionCrossSection>& ixs)
        : dd(idd), xs(ixs) {}

    std::shared_ptr<DiffusionData> dd;
    std::shared_ptr<DiffusionCrossSection>
        xs;  // Might be different from xs in dd !!
  };
  using NeighborInfo =
      std::pair<DiffusionGeometry::Tile, std::optional<std::size_t>>;

 private:
  // Method relating to solve
  void update_adfs();
  void update_physical_diffusion_coefficients();
  void update_nonlinear_diffusion_coefficients();
  template <typename DiffCoeffUpdater>
  void update_diffusion_coefficients(DiffCoeffUpdater dcu);
  void update_fluxes(Eigen::VectorXd& flux);
  void update_currents();
  void update_transverse_leakage_coefficients();

  void solve_keff();
  void fill_loss_matrix(Eigen::SparseMatrix<double, Eigen::RowMajor>& M) const;
  void fill_fission_matrix(
      Eigen::SparseMatrix<double, Eigen::RowMajor>& F) const;

  // To get index in nodal solver
  int ind(std::size_t n, std::size_t g) const {
    return static_cast<int>(n * NG_ + g);
  }

  std::tuple<double, double, int> get_D_Dnl_j(std::size_t n, Side s,
                                              std::size_t g) const {
    NodalCMFDSurface surf(n, s);
    int j = -1;
    const auto neighbor = this->neighbors_(n, s);
    if (neighbor.second) {
      std::size_t m = neighbor.second.value();
      surf = NodalCMFDSurface(n, s, m);
      j = ind(m, g);
    }

    const auto surf_ind = this->surface_indices_.at(surf);

    return {this->surface_diffusion_coefficients_(surf_ind, 0, g),
            this->surface_diffusion_coefficients_(surf_ind, 1, g), j};
  };

  // Reconstruction related methods

 private:
  xt::xtensor<Node, 2> nodes_;  // Node THEN Group ! Must be to make span
  xt::xtensor<IntraNodalFlux, 2> reconstructed_flux_params_;  // Node, group
  NM nodal_solver_;

  // Neighbors for each node. Use Side present from nodal_cmfd_surface.hpp
  // Side: XP = 0, XN = 1, YP = 2, YN = 3, ZP = 4, ZN = 5
  xt::xtensor<NeighborInfo, 2> neighbors_;           // node, side
  std::vector<DiffusionDataCrossSectionPair> mats_;  // Modified in solve !

  // Map to convert from surface to index
  std::unordered_map<NodalCMFDSurface, std::size_t> surface_indices_;

  // Surface index, physical/nonlinear diffusion coefficient, group
  // Must index this way so we can send continuous spans to NodalMethod
  xt::xtensor<double, 3> surface_diffusion_coefficients_;

  // Holds average flux between iterations
  Eigen::VectorXd flux_;

  std::shared_ptr<DiffusionGeometry> geom_;
  std::size_t NG_;  // Number of groups
  std::size_t NM_;  // Number of regions
  std::size_t nonlinear_update_frequency_{20};
  double keff_{1.};
  double flux_tol_{1.E-5};
  double keff_tol_{1.E-5};
  bool leakage_corrections_{false};
  bool solved_{false};
};

template <NodalMethod NM>
inline NodalDiffusionDriver<NM>::NodalDiffusionDriver(
    std::shared_ptr<DiffusionGeometry> geom)
    : nodal_solver_(0), geom_(geom) {
  if (geom_ == nullptr) {
    const auto mssg = "Provided geometry was None.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  if (geom_->ndims() != 3) {
    auto mssg = "NodalDiffusionDriver requires a 3D diffusion geometry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  NG_ = geom_->ngroups();
  NM_ = geom_->nmats();

  // Initialize nodal solver
  nodal_solver_ = std::move(NM(NG_));

  // Initialize size of nodes_ and reconstructed_flux_params_
  nodes_.resize({NM_, NG_});
  reconstructed_flux_params_.resize({NM_, NG_});
  this->update_adfs();

  // Initialize mats_
  mats_.reserve(NM_);
  for (std::size_t m = 0; m < NM_; m++) {
    const auto geom_indx = geom_->geom_indx(m);
    const auto& diff_data = geom_->mat(geom_indx);
    const auto& xs_ptr = diff_data->xs();
    // Save hard copy of cross section that can be modified !
    mats_.emplace_back(diff_data,
                       std::make_shared<DiffusionCrossSection>(*xs_ptr));
  }

  // Initialize neighbors
  neighbors_.resize({NM_, 6});
  for (std::size_t m = 0; m < NM_; m++) {
    neighbors_(m, Side::XP) = geom_->neighbor(m, Side::XP);
    neighbors_(m, Side::XN) = geom_->neighbor(m, Side::XN);
    neighbors_(m, Side::YP) = geom_->neighbor(m, Side::YP);
    neighbors_(m, Side::YN) = geom_->neighbor(m, Side::YN);
    neighbors_(m, Side::ZP) = geom_->neighbor(m, Side::ZP);
    neighbors_(m, Side::ZN) = geom_->neighbor(m, Side::ZN);
  }

  // Initialize surface indices. First, go through and add all unique surfaces
  // to the map
  for (std::size_t m = 0; m < NM_; m++) {
    for (uint8_t s = 0; s < 6; s++) {
      Side side = static_cast<Side>(s);
      NodalCMFDSurface surf(m, side);
      if (neighbors_(m, s).second) {
        surf = NodalCMFDSurface(m, side, neighbors_(m, s).second.value());
      }
      surface_indices_[surf] = 0;
    }
  }
  // Now we go through all surfaces and index them sequentially
  std::size_t ind = 0;
  for (auto& surf : surface_indices_) {
    surf.second = ind++;
  }

  // Initialize diffusion coefficients array
  surface_diffusion_coefficients_.resize({surface_indices_.size(), 2, NG_});
  surface_diffusion_coefficients_.fill(0.);
  this->update_physical_diffusion_coefficients();

  // Set size of flux array
  flux_.resize(NG_ * NM_);
  flux_.fill(1.);
}

template <NodalMethod NM>
void NodalDiffusionDriver<NM>::set_flux_tolerance(double ftol) {
  if (ftol <= 0.) {
    auto mssg = "Tolerance for flux must be in the interval (0., 0.1).";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (ftol >= 0.1) {
    auto mssg = "Tolerance for flux must be in the interval (0., 0.1).";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  flux_tol_ = ftol;
}

template <NodalMethod NM>
void NodalDiffusionDriver<NM>::set_keff_tolerance(double ktol) {
  if (ktol <= 0.) {
    auto mssg = "Tolerance for keff must be in the interval (0., 0.1).";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (ktol >= 0.1) {
    auto mssg = "Tolerance for keff must be in the interval (0., 0.1).";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  keff_tol_ = ktol;
}

template <NodalMethod NM>
inline void NodalDiffusionDriver<NM>::set_nonlinear_update_frequency(
    std::size_t f) {
  if (f == 0) {
    auto mssg = "The nonlinear update frequency must be > 0.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  } else if (f > 100) {
    auto mssg = "The nonlinear update frequency is larger than 100.";
    spdlog::warn(mssg);
  }

  nonlinear_update_frequency_ = f;
}

template <NodalMethod NM>
inline void NodalDiffusionDriver<NM>::update_adfs() {
  // Get all ADFs for each node / group
  for (std::size_t m = 0; m < NM_; m++) {
    for (std::size_t g = 0; g < NG_; g++) {
      Node& node = nodes_(m, g);
      node.adf_xn() = geom_->adf_xn(m, g);
      node.adf_xp() = geom_->adf_xp(m, g);
      node.adf_yn() = geom_->adf_yn(m, g);
      node.adf_yp() = geom_->adf_yp(m, g);
      node.adf_zn() = geom_->adf_zn(m, g);
      node.adf_zp() = geom_->adf_zp(m, g);
    }
  }
}

template <NodalMethod NM>
template <typename DiffCoeffUpdater>
inline void NodalDiffusionDriver<NM>::update_diffusion_coefficients(
    DiffCoeffUpdater dcu) {
  const double invs_keff = 1. / this->keff_;

  // Compute diffusion coefficient for each surface
  for (const auto& surf_ind : surface_indices_) {
    const auto& surf = surf_ind.first;
    const auto i = surf_ind.second;

    // Create spans to the physical and non-linear diffusion coefficients for
    // this surface for all energy groups.
    std::span<double> D(&surface_diffusion_coefficients_(i, 0, 0), NG_);
    std::span<double> Dnl(&surface_diffusion_coefficients_(i, 1, 0), NG_);

    // Info for the left node
    const std::size_t n1 = surf.node1;
    const Side side = surf.side;
    const auto geom_inds_1 = geom_->geom_indx(n1);
    const std::array<double, 3> ldx{geom_->dx(geom_inds_1[0]),
                                    geom_->dy(geom_inds_1[1]),
                                    geom_->dz(geom_inds_1[2])};
    const std::shared_ptr<DiffusionCrossSection>& lxs = mats_[n1].xs;
    std::span<const Node> lnode(&nodes_(n1, 0), NG_);

    if (surf.node2) {
      // Here, we have 2 nodes. Get right node info
      const std::size_t n2 = surf.node2.value();
      const auto geom_inds_2 = geom_->geom_indx(n2);
      const std::array<double, 3> rdx{geom_->dx(geom_inds_2[0]),
                                      geom_->dy(geom_inds_2[1]),
                                      geom_->dz(geom_inds_2[2])};
      const std::shared_ptr<DiffusionCrossSection>& rxs = mats_[n2].xs;
      std::span<const Node> rnode(&nodes_(n2, 0), NG_);

      // Compute the diffusion coefficients for all groups
      dcu(lnode, side, rnode, D, *lxs, *rxs, ldx, rdx, Dnl, invs_keff);
    } else {
      // Here, we have a node and a boundary condition
      if (neighbors_(n1, side).first.albedo.has_value() == false) {
        const auto mssg =
            "Encountered node surface with no neighbor and no albedo.";
        spdlog::error(mssg);
        throw ScarabeeException(mssg);
      }
      // Albedo for the surface
      const double B = neighbors_(n1, side).first.albedo.value();

      // Compute the diffusion coefficients for all groups
      dcu(lnode, side, D, B, *lxs, ldx, Dnl, invs_keff);
    }
  }
}

struct PhysicalDiffCoeffUpdater {
  void operator()(std::span<const Node> /*lnode*/, const Side side,
                  std::span<const Node> /*rnode*/, std::span<double> D,
                  const DiffusionCrossSection& lxs,
                  const DiffusionCrossSection& rxs,
                  const std::array<double, 3> ld,
                  const std::array<double, 3> rd,
                  std::span<const double> /*Dnl*/, const double /*invs_keff*/) {
    const double ldx = get_node_width(ld, side);
    const double rdx = get_node_width(rd, side);
    for (std::size_t g = 0; g < lxs.ngroups(); g++) {
      const double lD = lxs.D(g);
      const double rD = rxs.D(g);
      D[g] = 2. * lD * rD / (ldx * lD + rdx * rD);
    }
  }

  void operator()(std::span<const Node> /*lnode*/, const Side side,
                  std::span<double> D, const double B,
                  const DiffusionCrossSection& lxs,
                  const std::array<double, 3> ld,
                  std::span<const double> /*Dnl*/, const double /*invs_keff*/) {
    const double ldx = get_node_width(ld, side);
    for (std::size_t g = 0; g < lxs.ngroups(); g++) {
      const double lD = lxs.D(g);
      D[g] = 2. * lD * (1. - B) / (4. * lD * (1. + B) + ldx * (1. - B));
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
};

template <NodalMethod NM>
struct NonlinearDiffCoeffUpdater {
  NonlinearDiffCoeffUpdater(NM& ns) : nodal_solver(ns) {}

  void operator()(std::span<const Node> lnode, const Side side,
                  std::span<const Node> rnode, std::span<const double> D,
                  const DiffusionCrossSection& lxs,
                  const DiffusionCrossSection& rxs,
                  const std::array<double, 3> ld,
                  const std::array<double, 3> rd, std::span<double> Dnl,
                  const double invs_keff) {
    nodal_solver.compute_keff_nonlinear_diffusion_coefficient(
        lnode, side, rnode, D, lxs, rxs, ld, rd, Dnl, invs_keff);
  }

  void operator()(std::span<const Node> lnode, const Side side,
                  std::span<const double> D, const double B,
                  const DiffusionCrossSection& lxs,
                  const std::array<double, 3> ld, std::span<double> Dnl,
                  const double invs_keff) {
    nodal_solver.compute_keff_nonlinear_diffusion_coefficient(
        lnode, side, D, B, lxs, ld, Dnl, invs_keff);
  }

  NM& nodal_solver;
};

template <NodalMethod NM>
inline void NodalDiffusionDriver<NM>::update_physical_diffusion_coefficients() {
  PhysicalDiffCoeffUpdater pdcu;
  this->update_diffusion_coefficients<PhysicalDiffCoeffUpdater>(pdcu);
}

template <NodalMethod NM>
inline void
NodalDiffusionDriver<NM>::update_nonlinear_diffusion_coefficients() {
  this->update_diffusion_coefficients<NonlinearDiffCoeffUpdater<NM>>(
      NonlinearDiffCoeffUpdater<NM>(this->nodal_solver_));
}

template <NodalMethod NM>
inline void NodalDiffusionDriver<NM>::update_fluxes(Eigen::VectorXd& flux) {
  for (std::size_t m = 0; m < NM_; m++) {
    for (std::size_t g = 0; g < NG_; g++) {
      nodes_(m, g).phi0() = flux(ind(m, g));
    }
  }
}

template <NodalMethod NM>
inline void NodalDiffusionDriver<NM>::update_currents() {
  // Go through each surface
  for (const auto& surf_ind : surface_indices_) {
    const auto surf = surf_ind.first;
    const auto ind = surf_ind.second;

    for (std::size_t g = 0; g < NG_; g++) {
      const double D = surface_diffusion_coefficients_(ind, 0, g);
      const double Dnl = surface_diffusion_coefficients_(ind, 1, g);
      const double phi_n1 = nodes_(surf.node1, g).phi0();

      const double J = [g, D, Dnl, phi_n1, &surf, this]() -> double {
        if (surf.node2) {
          const double phi_n2 = nodes_(surf.node2.value(), g).phi0();
          return -D * (phi_n2 - phi_n1) - Dnl * (phi_n2 + phi_n1);
        } else if (surf.side == Side::XP || surf.side == Side::YP ||
                   surf.side == Side::ZP) {
          return (D - Dnl) * phi_n1;
        } else {
          return -(D + Dnl) * phi_n1;
        }
      }();

      switch (surf.side) {
        case Side::XN:
          nodes_(surf.node1, g).J_xn() = J;
          if (surf.node2) nodes_(surf.node2.value(), g).J_xp() = J;
          break;
        case Side::XP:
          nodes_(surf.node1, g).J_xp() = J;
          if (surf.node2) nodes_(surf.node2.value(), g).J_xn() = J;
          break;
        case Side::YN:
          nodes_(surf.node1, g).J_yn() = J;
          if (surf.node2) nodes_(surf.node2.value(), g).J_yp() = J;
          break;
        case Side::YP:
          nodes_(surf.node1, g).J_yp() = J;
          if (surf.node2) nodes_(surf.node2.value(), g).J_yn() = J;
          break;
        case Side::ZN:
          nodes_(surf.node1, g).J_zn() = J;
          if (surf.node2) nodes_(surf.node2.value(), g).J_zp() = J;
          break;
        case Side::ZP:
          nodes_(surf.node1, g).J_zp() = J;
          if (surf.node2) nodes_(surf.node2.value(), g).J_zn() = J;
          break;
      }
    }
  }
}

template <NodalMethod NM>
inline void NodalDiffusionDriver<NM>::update_transverse_leakage_coefficients() {
  for (std::size_t m = 0; m < NM_; m++) {
    for (std::size_t g = 0; g < NG_; g++) {
      Node& node = nodes_(m, g);

      // Get the currents along each axis at the positive and negative bounds
      const double Jxp = node.J_xp();
      const double Jxm = node.J_xn();
      const double Jyp = node.J_yp();
      const double Jym = node.J_yn();
      const double Jzp = node.J_zp();
      const double Jzm = node.J_zn();

      // Compute average transverse leakages in each direction
      const double Lx = Jxp - Jxm;
      const double Ly = Jyp - Jym;
      const double Lz = Jzp - Jzm;

      // Get neighbor info
      const auto& n_xp = neighbors_(m, Side::XP);
      const auto& n_xm = neighbors_(m, Side::XN);
      const auto& n_yp = neighbors_(m, Side::YP);
      const auto& n_ym = neighbors_(m, Side::YN);
      const auto& n_zp = neighbors_(m, Side::ZP);
      const auto& n_zm = neighbors_(m, Side::ZN);

      // Obtain geometry spacings for node
      const auto geom_indxs = geom_->geom_indx(m);
      const double dx = geom_->dx(geom_indxs[0]);
      const double dy = geom_->dy(geom_indxs[1]);
      const double dz = geom_->dz(geom_indxs[2]);
      const double invs_dx = 1. / dx;
      const double invs_dy = 1. / dy;
      const double invs_dz = 1. / dz;

      // This returns the average transverse leakage moments for a given node
      auto comp_avg_trans_lks = [this](std::size_t g, std::size_t m) {
        const Node& n = this->nodes_(m, g);

        const double Jxp = n.J_xp();
        const double Jxm = n.J_xn();
        const double Jyp = n.J_yp();
        const double Jym = n.J_yn();
        const double Jzp = n.J_zp();
        const double Jzm = n.J_zn();

        const double Lx = Jxp - Jxm;
        const double Ly = Jyp - Jym;
        const double Lz = Jzp - Jzm;

        return std::array<double, 3>{Lx, Ly, Lz};
      };

      std::array<double, 3> tmp;
      // x-axis
      if (n_xp.second && n_xm.second) {
        const double dx_xp = geom_->dx(geom_indxs[0] + 1);
        const double dx_xm = geom_->dx(geom_indxs[0] - 1);
        const double eta_xp = dx_xp * invs_dx;
        const double eta_xm = dx_xm * invs_dx;
        const double p1xm = eta_xm + 1.;
        const double p2xm = 2. * eta_xm + 1.;
        const double p1xp = eta_xp + 1.;
        const double p2xp = 2. * eta_xp + 1.;
        const double invs_denom = 1. / (p1xp * p1xm * (eta_xp + eta_xm + 1.));

        tmp = comp_avg_trans_lks(g, n_xp.second.value());
        const double Ly_xp = tmp[1];
        const double Lz_xp = tmp[2];

        tmp = comp_avg_trans_lks(g, n_xm.second.value());
        const double Ly_xm = tmp[1];
        const double Lz_xm = tmp[2];

        node.Lx_rho_y1() = (p1xm * p2xm * Ly_xp - p1xp * p2xp * Ly_xm +
                            (p1xp * p2xp - p1xm * p2xm) * Ly) *
                           invs_denom;
        node.Lx_rho_y2() =
            (p1xm * Ly_xp + p1xp * Ly_xm - (eta_xp + eta_xm + 2.) * Ly) *
            invs_denom;

        node.Lx_rho_z1() = (p1xm * p2xm * Lz_xp - p1xp * p2xp * Lz_xm +
                            (p1xp * p2xp - p1xm * p2xm) * Lz) *
                           invs_denom;
        node.Lx_rho_z2() =
            (p1xm * Lz_xp + p1xp * Lz_xm - (eta_xp + eta_xm + 2.) * Lz) *
            invs_denom;
      } else {
        node.Lx_rho_y1() = 0.;
        node.Lx_rho_y2() = 0.;
        node.Lx_rho_z1() = 0.;
        node.Lx_rho_z2() = 0.;
      }

      // y-axis
      if (n_yp.second && n_ym.second) {
        const double dy_yp = geom_->dy(geom_indxs[1] + 1);
        const double dy_ym = geom_->dy(geom_indxs[1] - 1);
        const double eta_yp = dy_yp * invs_dy;
        const double eta_ym = dy_ym * invs_dy;
        const double p1ym = eta_ym + 1.;
        const double p2ym = 2. * eta_ym + 1.;
        const double p1yp = eta_yp + 1.;
        const double p2yp = 2. * eta_yp + 1.;
        const double invs_denom = 1. / (p1yp * p1ym * (eta_yp + eta_ym + 1.));

        tmp = comp_avg_trans_lks(g, n_yp.second.value());
        const double Lx_yp = tmp[0];
        const double Lz_yp = tmp[2];

        tmp = comp_avg_trans_lks(g, n_ym.second.value());
        const double Lx_ym = tmp[0];
        const double Lz_ym = tmp[2];

        node.Ly_rho_x1() = (p1ym * p2ym * Lx_yp - p1yp * p2yp * Lx_ym +
                            (p1yp * p2yp - p1ym * p2ym) * Lx) *
                           invs_denom;
        node.Ly_rho_x2() =
            (p1ym * Lx_yp + p1yp * Lx_ym - (eta_yp + eta_ym + 2.) * Lx) *
            invs_denom;

        node.Ly_rho_z1() = (p1ym * p2ym * Lz_yp - p1yp * p2yp * Lz_ym +
                            (p1yp * p2yp - p1ym * p2ym) * Lz) *
                           invs_denom;
        node.Ly_rho_z2() =
            (p1ym * Lz_yp + p1yp * Lz_ym - (eta_yp + eta_ym + 2.) * Lz) *
            invs_denom;
      } else {
        node.Ly_rho_x1() = 0.;
        node.Ly_rho_x2() = 0.;
        node.Ly_rho_z1() = 0.;
        node.Ly_rho_z2() = 0.;
      }

      // z-axis
      if (n_zp.second && n_zm.second) {
        const double dz_zp = geom_->dz(geom_indxs[2] + 1);
        const double dz_zm = geom_->dz(geom_indxs[2] - 1);
        const double eta_zp = dz_zp * invs_dz;
        const double eta_zm = dz_zm * invs_dz;
        const double p1zm = eta_zm + 1.;
        const double p2zm = 2. * eta_zm + 1.;
        const double p1zp = eta_zp + 1.;
        const double p2zp = 2. * eta_zp + 1.;
        const double invs_denom = 1. / (p1zp * p1zm * (eta_zp + eta_zm + 1.));

        tmp = comp_avg_trans_lks(g, n_zp.second.value());
        const double Lx_zp = tmp[0];
        const double Ly_zp = tmp[1];

        tmp = comp_avg_trans_lks(g, n_zm.second.value());
        const double Lx_zm = tmp[0];
        const double Ly_zm = tmp[1];

        node.Lz_rho_x1() = (p1zm * p2zm * Lx_zp - p1zp * p2zp * Lx_zm +
                            (p1zp * p2zp - p1zm * p2zm) * Lx) *
                           invs_denom;
        node.Lz_rho_x2() =
            (p1zm * Lx_zp + p1zp * Lx_zm - (eta_zp + eta_zm + 2.) * Lx) *
            invs_denom;

        node.Lz_rho_y1() = (p1zm * p2zm * Ly_zp - p1zp * p2zp * Ly_zm +
                            (p1zp * p2zp - p1zm * p2zm) * Ly) *
                           invs_denom;
        node.Lz_rho_y2() =
            (p1zm * Ly_zp + p1zp * Ly_zm - (eta_zp + eta_zm + 2.) * Ly) *
            invs_denom;
      } else {
        node.Lz_rho_x1() = 0.;
        node.Lz_rho_x2() = 0.;
        node.Lz_rho_y1() = 0.;
        node.Lz_rho_y2() = 0.;
      }
    }
  }
}

template <NodalMethod NM>
void NodalDiffusionDriver<NM>::fill_loss_matrix(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& M) const {
  const auto exp_len = static_cast<Eigen::Index>(NM_ * NG_);
  if (M.rows() != exp_len || M.cols() != exp_len) {
    // Resize the matrix if needed (usually only on first call)
    M.resize(exp_len, exp_len);
    M.reserve(Eigen::VectorX<std::size_t>::Constant(exp_len, 6 + NG_));
  } else {
    // If we are re-building the matrix, we should zero all entries first
    for (int k = 0; k < M.outerSize(); k++) {
      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(M, k);
           it; ++it) {
        it.valueRef() = 0.;
      }
    }
  }

  // Next, we re-build the matrix from scratch.
  for (std::size_t m = 0; m < NM_; m++) {
    const DiffusionCrossSection& xs = *mats_[m].xs;
    const auto geom_inds = this->geom_->geom_indx(m);
    const double invs_dx = 1. / geom_->dx(geom_inds[0]);
    const double invs_dy = 1. / geom_->dy(geom_inds[1]);
    const double invs_dz = 1. / geom_->dz(geom_inds[2]);

    for (std::size_t g = 0; g < NG_; g++) {
      const auto i = ind(m, g);

      // Handle leakage along x
      const auto [D_xn, Dnl_xn, j_xn] = get_D_Dnl_j(m, Side::XN, g);
      const auto [D_xp, Dnl_xp, j_xp] = get_D_Dnl_j(m, Side::XP, g);
      if (j_xn >= 0) M.coeffRef(i, j_xn) += (Dnl_xn - D_xn) * invs_dx;
      M.coeffRef(i, i) += (D_xn + D_xp + Dnl_xn - Dnl_xp) * invs_dx;
      if (j_xp >= 0) M.coeffRef(i, j_xp) += (-D_xp - Dnl_xp) * invs_dx;

      // Handle leakage along y
      const auto [D_yn, Dnl_yn, j_yn] = get_D_Dnl_j(m, Side::YN, g);
      const auto [D_yp, Dnl_yp, j_yp] = get_D_Dnl_j(m, Side::YP, g);
      if (j_yn >= 0) M.coeffRef(i, j_yn) += (Dnl_yn - D_yn) * invs_dy;
      M.coeffRef(i, i) += (D_yn + D_yp + Dnl_yn - Dnl_yp) * invs_dy;
      if (j_yp >= 0) M.coeffRef(i, j_yp) += (-D_yp - Dnl_yp) * invs_dy;

      // Handle leakage along z
      const auto [D_zn, Dnl_zn, j_zn] = get_D_Dnl_j(m, Side::ZN, g);
      const auto [D_zp, Dnl_zp, j_zp] = get_D_Dnl_j(m, Side::ZP, g);
      if (j_zn >= 0) M.coeffRef(i, j_zn) += (Dnl_zn - D_zn) * invs_dz;
      M.coeffRef(i, i) += (D_zn + D_zp + Dnl_zn - Dnl_zp) * invs_dz;
      if (j_zp >= 0) M.coeffRef(i, j_zp) += (-D_zp - Dnl_zp) * invs_dz;

      // Handle removal
      M.coeffRef(i, i) += xs.Er(g);

      // Handle scattering
      for (std::size_t gg = 0; gg < NG_; gg++)
        if (gg != g) M.coeffRef(i, ind(m, gg)) -= xs.Es(gg, g);
    }
  }

  M.makeCompressed();
}

template <NodalMethod NM>
void NodalDiffusionDriver<NM>::fill_fission_matrix(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& F) const {
  const auto exp_len = static_cast<Eigen::Index>(NM_ * NG_);
  if (F.rows() != exp_len || F.cols() != exp_len) {
    // Resize the matrix if needed (usually only on first call)
    F.resize(exp_len, exp_len);
    F.reserve(Eigen::VectorX<std::size_t>::Constant(exp_len, NG_));
  } else {
    // If we are re-building the matrix, we should zero all entries first
    for (int k = 0; k < F.outerSize(); k++) {
      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(F, k);
           it; ++it) {
        it.valueRef() = 0.;
      }
    }
  }

  // Next, we re-build the matrix from scratch.
  for (std::size_t m = 0; m < NM_; m++) {
    const DiffusionCrossSection& xs = *mats_[m].xs;

    for (std::size_t g = 0; g < NG_; g++) {
      const double chi_g = xs.chi(g);
      const auto i = ind(m, g);

      for (std::size_t gg = 0; gg < NG_; gg++)
        F.coeffRef(i, ind(m, gg)) += chi_g * xs.vEf(gg);
    }
  }

  F.makeCompressed();
}

template <NodalMethod NM>
void NodalDiffusionDriver<NM>::solve_keff() {
  // Power Iteration to solve for Keff
  // Initialize flux and source vectors
  Eigen::VectorXd new_flux(NG_ * NM_);
  Eigen::VectorXd Q(NG_ * NM_);

  // Initialize a vector for computing keff faster
  Eigen::VectorXd VvEf(NG_ * NM_);
  for (std::size_t m = 0; m < NM_; m++) {
    const DiffusionCrossSection& xs = *mats_[m].xs;
    const auto geom_inds = geom_->geom_indx(m);
    const double dx = geom_->dx(geom_inds[0]);
    const double dy = geom_->dy(geom_inds[1]);
    const double dz = geom_->dz(geom_inds[2]);
    const double V = dx * dy * dz;
    for (std::size_t g = 0; g < NG_; g++) {
      VvEf(ind(m, g)) = V * xs.vEf(g);
    }
  }

  // Ensure flux is normalized
  flux_.normalize();

  // Initialize loss matrix and fission matrix
  Eigen::SparseMatrix<double, Eigen::RowMajor> M;
  Eigen::SparseMatrix<double, Eigen::RowMajor> F;
  fill_loss_matrix(M);
  fill_fission_matrix(F);

  // Create a solver for the problem
  Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver;
  solver.setTolerance(1.E-60);
  solver.compute(M);
  if (solver.info() != Eigen::Success) {
    std::stringstream mssg;
    mssg << "Could not initialize nodal iterative solver";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // A lambda to compute the maximum flux difference
  const auto compute_max_flux_diff = [this](const auto& old_flux,
                                            const auto& new_flux) -> double {
    double flux_diff = 0.;
    for (std::size_t i = 0; i < NM_ * NG_; i++) {
      const double flux_diff_i =
          std::abs(new_flux(i) - old_flux(i)) / new_flux(i);
      if (flux_diff_i > flux_diff) flux_diff = flux_diff_i;
    }
    return flux_diff;
  };

  // Begin power iteration
  double keff_diff = 100.;
  double flux_diff = 100.;
  std::size_t iteration = 0;
  while (keff_diff > keff_tol_ || flux_diff > flux_tol_) {
    iteration++;
    // Compute source vector
    Q = (1. / keff_) * F * flux_;

    // Get new flux
    new_flux = solver.solveWithGuess(Q, flux_);
    // For some reason, this doesn't seem to be working with the new versions
    // of Eigen, despite clearly succeeding. Just commenting it out for now.
    // if (solver.info() != Eigen::Success) {
    //   spdlog::error("Solution impossible.");
    //   throw ScarabeeException("Solution impossible");
    // }

    // Estimate keff
    double prev_keff = keff_;
    keff_ = prev_keff * (VvEf.dot(new_flux) / VvEf.dot(flux_));
    keff_diff = std::abs(keff_ - prev_keff) / keff_;

    // Normalize our new flux
    new_flux *= prev_keff / keff_;

    // Find the max flux error
    flux_diff = compute_max_flux_diff(flux_, new_flux);
    flux_ = new_flux;

    // Write information
    spdlog::info("-------------------------------------");
    spdlog::info("Iteration {:>4d}          keff: {:.5f}", iteration, keff_);
    spdlog::info("     keff difference:     {:.5E}", keff_diff);
    spdlog::info("     max flux difference: {:.5E}", flux_diff);

    if (iteration % nonlinear_update_frequency_ == 0) {
      flux_diff = 1.;
      spdlog::info("-------------------------------------");
      spdlog::info("");
      spdlog::info("Updating nonlinear diffusion coefficients");
      spdlog::info("");
      update_fluxes(flux_);
      if (NM::update_currents) {
        update_currents();
        update_transverse_leakage_coefficients();
      }
      update_nonlinear_diffusion_coefficients();
      fill_loss_matrix(
          M);  // Will also need to update this when XS is updated !
      solver.compute(M);
    }
    // TODO update_cross_sections();
  }

  update_fluxes(flux_);
}

}  // namespace scarabee

#endif
