#ifndef SCARABEE_NODAL_DIFFUSION_DRIVER_H
#define SCARABEE_NODAL_DIFFUSION_DRIVER_H

#include <data/diffusion_cross_section.hpp>
#include <diffusion/diffusion_data.hpp>
#include <diffusion/diffusion_geometry.hpp>
#include <diffusion/intra_nodal_flux.hpp>
#include <diffusion/node.hpp>
#include <diffusion/nodal_cmfd_surface.hpp>
#include <diffusion/nodal_method.hpp>
#include <utils/serialization.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>
#include <utils/timer.hpp>

#include <xtensor/containers/xtensor.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/archives/portable_binary.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>

namespace scarabee {

template <NodalMethod NM>
class NodalDiffusionDriver {
  using Vector = Eigen::VectorXd;
  using Matrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  using Solver = Eigen::BiCGSTAB<Matrix, Eigen::IdentityPreconditioner>;

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

  bool leakage_corrections() const
    requires(NM::update_currents)
  {
    return leakage_corrections_;
  }
  void set_leakage_corrections(bool lc)
    requires(NM::update_currents)
  {
    leakage_corrections_ = lc;
  }

  std::size_t nonlinear_update_frequency() const {
    return nonlinear_update_frequency_;
  }
  void set_nonlinear_update_frequency(std::size_t f);

  std::size_t source_extrapolation_frequency() const {
    return source_extrapolation_frequency_;
  }
  void set_source_extrapolation_frequency(std::size_t f);

  std::size_t max_inner_iterations() const { return max_bicgstab_iterations_; }
  void set_max_inner_iterations(std::size_t n);

  double keff() const { return keff_; }

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

 private:
  struct DiffusionDataCrossSectionPair {
    DiffusionDataCrossSectionPair(
        const std::shared_ptr<DiffusionData>& idd,
        const std::shared_ptr<DiffusionCrossSection>& ixs)
        : dd(idd), xs(ixs) {}
    DiffusionDataCrossSectionPair() : dd(nullptr), xs(nullptr) {}

    std::shared_ptr<DiffusionData> dd;
    std::shared_ptr<DiffusionCrossSection>
        xs;  // Might be different from xs in dd !!

    template <class Archive>
    void serialize(Archive& arc) {
      arc(CEREAL_NVP(dd), CEREAL_NVP(xs));
    }
  };

  using NeighborInfo =
      std::pair<DiffusionGeometry::Tile, std::optional<std::size_t>>;

  enum class Corner { PP, PM, MP, MM };

 private:
  // Method relating to solve
  void update_adfs();
  void update_physical_diffusion_coefficients();
  double update_nonlinear_diffusion_coefficients();

  template <typename DiffCoeffUpdater>
  double update_diffusion_coefficients(DiffCoeffUpdater dcu);

  void update_fluxes(const Vector& flux);
  void update_currents();
  void update_transverse_leakage_coefficients();

  void solve_keff();
  void fill_loss_matrix(Matrix& M) const;
  void fill_fission_matrix(Matrix& F) const;

  double calc_node_avg_DB2(const std::size_t g, const std::size_t m,
                           const double dx, const double dy,
                           const double dz) const;
  void update_node_xs()
    requires(NM::update_currents);

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
  void perform_flux_reconstruction()
    requires(NM::reconstruct_flux);
  IntraNodalFlux fit_node_recon_params(std::size_t g, std::size_t m) const
    requires(NM::reconstruct_flux);
  void fit_node_recon_params_corners(std::size_t g, std::size_t m)
    requires(NM::reconstruct_flux);
  double eval_heter_xy_corner_flux(std::size_t g, std::size_t m, Corner c) const
    requires(NM::reconstruct_flux);
  double avg_xy_corner_flux(std::size_t g, std::size_t m, Corner c) const
    requires(NM::reconstruct_flux);

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
  Vector flux_;

  std::shared_ptr<DiffusionGeometry> geom_;
  std::size_t NG_;                          // Number of groups
  std::size_t NM_;                          // Number of regions
  std::size_t nonlinear_update_frequency_;  // Set in constructor based on NM_
  std::size_t source_extrapolation_frequency_{5};
  std::size_t max_bicgstab_iterations_{2};
  double keff_{1.};
  double flux_tol_{1.E-5};
  double keff_tol_{1.E-5};
  double Dnl_tol_{1.E-3};
  bool leakage_corrections_{false};
  bool solved_{false};

  friend class cereal::access;
  NodalDiffusionDriver() : nodal_solver_(2) {}
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(nodes_), CEREAL_NVP(reconstructed_flux_params_),
        CEREAL_NVP(nodal_solver_), CEREAL_NVP(neighbors_), CEREAL_NVP(mats_),
        CEREAL_NVP(surface_indices_),
        CEREAL_NVP(surface_diffusion_coefficients_), CEREAL_NVP(flux_),
        CEREAL_NVP(geom_), CEREAL_NVP(NG_), CEREAL_NVP(NM_),
        CEREAL_NVP(nonlinear_update_frequency_),
        CEREAL_NVP(source_extrapolation_frequency_),
        CEREAL_NVP(max_bicgstab_iterations_), CEREAL_NVP(keff_),
        CEREAL_NVP(flux_tol_), CEREAL_NVP(keff_tol_), CEREAL_NVP(Dnl_tol_),
        CEREAL_NVP(leakage_corrections_), CEREAL_NVP(solved_));
  }
};

template <NodalMethod NM>
inline NodalDiffusionDriver<NM>::NodalDiffusionDriver(
    std::shared_ptr<DiffusionGeometry> geom)
    : nodal_solver_(2), geom_(geom) {
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

  // User lower diffusion coefficient tolerance for finite difference method
  if constexpr (NM::update_currents == false) {
    Dnl_tol_ = 1.E-1;
  }

  // Initialize the update frequency
  nonlinear_update_frequency_ =
      static_cast<std::size_t>(std::ceil(static_cast<double>(NM_) / 2.5));

  // Initialize nodal solver with correct number of groups if needed
  if (NG_ != 2) nodal_solver_ = NM(NG_);

  // Initialize size of nodes_
  nodes_.resize({NM_, NG_});

  // Only allocate memory for reconstructed flux if needed
  if constexpr (NM::reconstruct_flux) {
    reconstructed_flux_params_.resize({NM_, NG_});
  }

  // Set all ADFs on the nodes
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
  if (ftol <= 0. || ftol >= 0.1) {
    auto mssg = "Tolerance for flux must be in the interval (0., 0.1).";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  flux_tol_ = ftol;
}

template <NodalMethod NM>
void NodalDiffusionDriver<NM>::set_keff_tolerance(double ktol) {
  if (ktol <= 0. || ktol >= 0.1) {
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
inline void NodalDiffusionDriver<NM>::set_source_extrapolation_frequency(
    std::size_t f) {
  if (f == 0) {
    auto mssg = "The source update frequency must be > 0.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  } else if (f > 100) {
    auto mssg = "The source update frequency is larger than 100.";
    spdlog::warn(mssg);
  }

  source_extrapolation_frequency_ = f;
}

template <NodalMethod NM>
inline void NodalDiffusionDriver<NM>::set_max_inner_iterations(std::size_t n) {
  if (n == 0) {
    auto mssg = "The max number of inner iterations must be > 0.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  } else if (n > 20) {
    auto mssg = "The max number of inner iterations is larger than 20.";
    spdlog::warn(mssg);
  }

  max_bicgstab_iterations_ = n;
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
inline double NodalDiffusionDriver<NM>::update_diffusion_coefficients(
    DiffCoeffUpdater dcu) {
  const double invs_keff = 1. / this->keff_;

  // Compute diffusion coefficient for each surface
  double max_diff = 0.;
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
    std::span<Node> lnode(&nodes_(n1, 0), NG_);

    if (surf.node2) {
      // Here, we have 2 nodes. Get right node info
      const std::size_t n2 = surf.node2.value();
      const auto geom_inds_2 = geom_->geom_indx(n2);
      const std::array<double, 3> rdx{geom_->dx(geom_inds_2[0]),
                                      geom_->dy(geom_inds_2[1]),
                                      geom_->dz(geom_inds_2[2])};
      const std::shared_ptr<DiffusionCrossSection>& rxs = mats_[n2].xs;
      std::span<Node> rnode(&nodes_(n2, 0), NG_);

      // Compute the diffusion coefficients for all groups
      const double diff =
          dcu(lnode, side, rnode, D, *lxs, *rxs, ldx, rdx, Dnl, invs_keff);
      if (diff > max_diff) max_diff = diff;
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
      const double diff = dcu(lnode, side, D, B, *lxs, ldx, Dnl, invs_keff);
      if (diff > max_diff) max_diff = diff;
    }
  }
  return max_diff;
}

struct PhysicalDiffCoeffUpdater {
  double operator()(std::span<Node> /*lnode*/, const Side side,
                    std::span<Node> /*rnode*/, std::span<double> D,
                    const DiffusionCrossSection& lxs,
                    const DiffusionCrossSection& rxs,
                    const std::array<double, 3> ld,
                    const std::array<double, 3> rd,
                    std::span<const double> /*Dnl*/,
                    const double /*invs_keff*/) {
    const double ldx = get_node_width(ld, side);
    const double rdx = get_node_width(rd, side);
    for (std::size_t g = 0; g < lxs.ngroups(); g++) {
      const double lD = lxs.D(g);
      const double rD = rxs.D(g);
      D[g] = 2. * lD * rD / (ldx * lD + rdx * rD);
    }
    return 0.;  // Fake difference
  }

  double operator()(std::span<Node> /*lnode*/, const Side side,
                    std::span<double> D, const double B,
                    const DiffusionCrossSection& lxs,
                    const std::array<double, 3> ld,
                    std::span<const double> /*Dnl*/,
                    const double /*invs_keff*/) {
    const double ldx = get_node_width(ld, side);
    for (std::size_t g = 0; g < lxs.ngroups(); g++) {
      const double lD = lxs.D(g);
      D[g] = 2. * lD * (1. - B) / (4. * lD * (1. + B) + ldx * (1. - B));
    }
    return 0.;  // Fake difference
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
  NonlinearDiffCoeffUpdater(NM& ns) : nodal_solver(&ns) {}

  double operator()(std::span<Node> lnode, const Side side,
                    std::span<Node> rnode, std::span<const double> D,
                    const DiffusionCrossSection& lxs,
                    const DiffusionCrossSection& rxs,
                    const std::array<double, 3> ld,
                    const std::array<double, 3> rd, std::span<double> Dnl,
                    const double invs_keff) {
    return nodal_solver->compute_keff_nonlinear_diffusion_coefficient(
        lnode, side, rnode, D, lxs, rxs, ld, rd, Dnl, invs_keff);
  }

  double operator()(std::span<Node> lnode, const Side side,
                    std::span<const double> D, const double B,
                    const DiffusionCrossSection& lxs,
                    const std::array<double, 3> ld, std::span<double> Dnl,
                    const double invs_keff) {
    return nodal_solver->compute_keff_nonlinear_diffusion_coefficient(
        lnode, side, D, B, lxs, ld, Dnl, invs_keff);
  }

  NM* nodal_solver;
};

template <NodalMethod NM>
inline void NodalDiffusionDriver<NM>::update_physical_diffusion_coefficients() {
  PhysicalDiffCoeffUpdater pdcu;
  std::ignore =
      this->update_diffusion_coefficients<PhysicalDiffCoeffUpdater>(pdcu);
}

template <NodalMethod NM>
inline double
NodalDiffusionDriver<NM>::update_nonlinear_diffusion_coefficients() {
  return this->update_diffusion_coefficients<NonlinearDiffCoeffUpdater<NM>>(
      NonlinearDiffCoeffUpdater<NM>(this->nodal_solver_));
}

template <NodalMethod NM>
inline void NodalDiffusionDriver<NM>::update_fluxes(const Vector& flux) {
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
inline double NodalDiffusionDriver<NM>::calc_node_avg_DB2(
    const std::size_t g, const std::size_t m, const double dx, const double dy,
    const double dz) const {
  const Node& node = this->nodes_(m, g);
  double DB2 = 0.;
  DB2 += dy * dz * (node.J_xp() - node.J_xn());
  DB2 += dx * dz * (node.J_yp() - node.J_yn());
  DB2 += dx * dy * (node.J_zp() - node.J_zn());
  DB2 /= node.phi0() * dx * dy * dz;
  return DB2;
}

template <NodalMethod NM>
inline void NodalDiffusionDriver<NM>::update_node_xs()
  requires(NM::update_currents)
{
  // Go through all nodes and groups, updating the cross sections
  for (std::size_t m = 0; m < NM_; m++) {
    // const auto& dd = *diff_datas_[m];
    const DiffusionData& dd = *mats_[m].dd;
    if (dd.leakage_corrections().has_value() == false || dd.reflector())
      continue;
    const auto& lc = dd.leakage_corrections().value();
    const auto& sa_xs = *dd.xs();  // Single-Assembly (un-buckled) xs

    const auto geom_indx = geom_->geom_indx(m);
    const double del_x = geom_->dx(geom_indx[0]);
    const double del_y = geom_->dy(geom_indx[1]);
    const double del_z = geom_->dz(geom_indx[2]);
    DiffusionCrossSection& xs = *mats_[m].xs;

    for (std::size_t g_in = 0; g_in < NG_; g_in++) {
      // Compute node/group leakage to loss ratio
      const double DB2 = this->calc_node_avg_DB2(g_in, m, del_x, del_y, del_z);
      const double LRr = DB2 / xs.Er(g_in);

      // Can now determine fractional change in each group cross section
      const double fD = lc.D(g_in) * LRr;
      const double fEa = lc.Ea(g_in) * LRr;
      const double fEf = lc.Ef(g_in) * LRr;
      const double fvEf = lc.vEf(g_in) * LRr;

      // Update the cross sections
      xs.D_ref(g_in) = (fD + 1.) * sa_xs.D(g_in);
      xs.Ea_ref(g_in) = (fEa + 1.) * sa_xs.Ea(g_in);
      xs.Ef_ref(g_in) = (fEf + 1.) * sa_xs.Ef(g_in);
      xs.vEf_ref(g_in) = (fvEf + 1.) * sa_xs.vEf(g_in);

      for (std::size_t g_out = g_in + 1; g_out < NG_; g_out++) {
        const double fEs = lc.Es(g_in, g_out) * LRr;
        xs.Es_ref(g_in, g_out) = (fEs + 1.) * sa_xs.Es(g_in, g_out);
      }
    }
  }
}

template <NodalMethod NM>
void NodalDiffusionDriver<NM>::fill_loss_matrix(Matrix& M) const {
  const Eigen::Index exp_len = static_cast<Eigen::Index>(NM_ * NG_);
  if (M.rows() != exp_len || M.cols() != exp_len) {
    // Resize the matrix if needed (usually only on first call)
    M.resize(exp_len, exp_len);
    M.reserve(Eigen::VectorX<std::size_t>::Constant(exp_len, 7 + NG_));
  } else {
    // If we are re-building the matrix, we should zero all entries first
    for (int k = 0; k < M.outerSize(); k++) {
      for (Matrix::InnerIterator it(M, k); it; ++it) it.valueRef() = 0.;
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
void NodalDiffusionDriver<NM>::fill_fission_matrix(Matrix& F) const {
  const Eigen::Index exp_len = static_cast<Eigen::Index>(NM_ * NG_);
  if (F.rows() != exp_len || F.cols() != exp_len) {
    // Resize the matrix if needed (usually only on first call)
    F.resize(exp_len, exp_len);
    F.reserve(Eigen::VectorX<std::size_t>::Constant(exp_len, NG_));
  } else {
    // If we are re-building the matrix, we should zero all entries first
    for (int k = 0; k < F.outerSize(); k++) {
      for (Matrix::InnerIterator it(F, k); it; ++it) it.valueRef() = 0.;
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
}

template <NodalMethod NM>
void NodalDiffusionDriver<NM>::solve_keff() {
  Timer sim_timer;
  sim_timer.start();

  // Power Iteration to solve for Keff
  // Initialize flux and source vectors
  Vector new_flux(NG_ * NM_);
  Vector Q(NG_ * NM_);
  Vector Q_err = Q;

  // Initialize a vector for computing keff faster
  Vector VvEf(NG_ * NM_);
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
  Matrix M;
  Matrix F;
  fill_loss_matrix(M);
  fill_fission_matrix(F);

  // Create a solver for the problem
  Solver solver;
  solver.compute(M);
  solver.setTolerance(1.E-10);
  solver.setMaxIterations(
      max_bicgstab_iterations_);  // This is choice used by KOMODO

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
  double D_diff = 100.;
  double prev_src_err = 100.;
  double src_err = 100.;
  std::size_t iteration = 0;
  while (keff_diff > keff_tol_ || flux_diff > flux_tol_ || D_diff > Dnl_tol_) {
    iteration++;
    // Compute source vector
    Q = F * flux_;
    Q *= 1. / keff_;

    // Compute error in the source vector
    prev_src_err = src_err;
    Q_err = Q - Q_err;
    src_err = Q_err.norm();

    // Source extrapolation
    if (iteration > 3 && iteration % source_extrapolation_frequency_ == 0) {
      spdlog::info("-------------------------------------");
      spdlog::info("");
      spdlog::info("Extrapolating source");
      const double dominance_ration = src_err / prev_src_err;
      Q += (dominance_ration / (1. - dominance_ration)) * Q_err;
      spdlog::info("");
    }

    // Solve system
    new_flux = solver.solveWithGuess(Q, flux_);

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

    if ((iteration % nonlinear_update_frequency_ == 0 ||
         (flux_diff < flux_tol_ && D_diff > Dnl_tol_)) &&
        (keff_diff > keff_tol_ || flux_diff > flux_tol_ || D_diff > Dnl_tol_)) {
      spdlog::info("-------------------------------------");
      spdlog::info("");
      spdlog::info("Updating nonlinear diffusion coefficients");
      update_fluxes(flux_);
      if (NM::update_currents) {
        update_currents();
        update_transverse_leakage_coefficients();
      }
      D_diff = update_nonlinear_diffusion_coefficients();
      spdlog::info("Max difference: {:.5E}", D_diff);
      spdlog::info("");
      if constexpr (NM::update_currents) {
        if (leakage_corrections()) {
          update_node_xs();
          // Only need to update F when we update the cross sections
          fill_fission_matrix(F);
        }
      }
      fill_loss_matrix(M);
      solver.compute(M);
    }

    Q_err = Q;
  }

  // We must do one last update of the fluxes, currents, and non-linear
  // diffusion coefficients, as this data is needed for accurate intranodal
  // flux reconstruction.
  update_fluxes(flux_);
  if (NM::update_currents) {
    update_currents();
    update_transverse_leakage_coefficients();
  }
  D_diff = update_nonlinear_diffusion_coefficients();

  solved_ = true;

  sim_timer.stop();
  spdlog::info("");
  spdlog::info("Simulation Time: {:.5E} s", sim_timer.elapsed_time());

  if constexpr (NM::reconstruct_flux) perform_flux_reconstruction();
}

//=============================================================================
// Flux / Power Plotting Methods

template <NodalMethod NM>
inline double NodalDiffusionDriver<NM>::flux(double x, double y, double z,
                                             std::size_t g) const {
  // If problem isn't solved yet, we error
  if (solved_ == false) {
    auto mssg = "Cannot compute flux. Problem has not been solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Check group index
  if (g >= ngroups()) {
    std::stringstream mssg;
    mssg << "Group index g = " << g << " is out of range.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // Get geometry index
  const auto oi = geom_->x_to_i(x);
  const auto oj = geom_->y_to_j(y);
  const auto ok = geom_->z_to_k(z);
  if (oi.has_value() == false || oj.has_value() == false ||
      ok.has_value() == false)
    return 0.;
  const std::size_t i = oi.value();
  const std::size_t j = oj.value();
  const std::size_t k = ok.value();
  const xt::svector<std::size_t> geom_inds{i, j, k};

  // Get material index
  const auto om = geom_->geom_to_mat_indx(geom_inds);
  if (om.has_value() == false) return 0.;
  const std::size_t m = om.value();

  if constexpr (NM::reconstruct_flux) {
    return reconstructed_flux_params_(m, g)(x, y, z);
  } else {
    return flux_(ind(m, g));
  }
}

template <NodalMethod NM>
inline xt::xtensor<double, 4> NodalDiffusionDriver<NM>::flux(
    const xt::xtensor<double, 1>& x, const xt::xtensor<double, 1>& y,
    const xt::xtensor<double, 1>& z) const {
  // If problem isn't solved yet, we error
  if (solved_ == false) {
    auto mssg = "Cannot compute flux. Problem has not been solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure x, y, and z have at least 1 coordinate
  if (x.size() == 0) {
    auto mssg = "Array of x coordinates must have at least one entry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  if (y.size() == 0) {
    auto mssg = "Array of y coordinates must have at least one entry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  if (z.size() == 0) {
    auto mssg = "Array of z coordinates must have at least one entry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  xt::xtensor<double, 4> flux_out;
  flux_out.resize({ngroups(), x.size(), y.size(), z.size()});
  flux_out.fill(0.);

  for (std::size_t g = 0; g < ngroups(); g++) {
#pragma omp parallel for
    for (int ii = 0; ii < static_cast<int>(x.size()); ii++) {
      std::size_t i = static_cast<std::size_t>(ii);
      for (std::size_t j = 0; j < y.size(); j++) {
        for (std::size_t k = 0; k < z.size(); k++) {
          // Get geometry index
          const auto oi = geom_->x_to_i(x[i]);
          const auto oj = geom_->y_to_j(y[j]);
          const auto ok = geom_->z_to_k(z[k]);
          if (oi.has_value() == false || oj.has_value() == false ||
              ok.has_value() == false) {
            continue;
          }
          const std::size_t gi = oi.value();
          const std::size_t gj = oj.value();
          const std::size_t gk = ok.value();
          const xt::svector<std::size_t> geom_inds{gi, gj, gk};

          // Get material index
          const auto om = geom_->geom_to_mat_indx(geom_inds);
          if (om.has_value() == false) continue;
          const std::size_t m = om.value();

          if constexpr (NM::reconstruct_flux) {
            flux_out(g, i, j, k) =
                reconstructed_flux_params_(m, g)(x[i], y[j], z[k]);
          } else {
            flux_out(g, i, j, k) = flux_(ind(m, g));
          }
        }
      }
    }
  }

  return flux_out;
}

template <NodalMethod NM>
inline xt::xtensor<double, 4> NodalDiffusionDriver<NM>::avg_flux() const {
  // If problem isn't solved yet, we error
  if (solved_ == false) {
    auto mssg = "Cannot compute flux. Problem has not been solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  const std::size_t nx = geom_->nx();
  const std::size_t ny = geom_->ny();
  const std::size_t nz = geom_->nz();

  xt::xtensor<double, 4> flux_out;
  flux_out.resize({ngroups(), nx, ny, nz});

  for (std::size_t g = 0; g < ngroups(); g++) {
    for (std::size_t i = 0; i < nx; i++) {
      for (std::size_t j = 0; j < ny; j++) {
        for (std::size_t k = 0; k < nz; k++) {
          const auto om = geom_->geom_to_mat_indx({i, j, k});

          if (om.has_value() == false)
            flux_out(g, i, j, k) = 0.;
          else
            flux_out(g, i, j, k) = flux_(ind(*om, g));
        }
      }
    }
  }

  return flux_out;
}

template <NodalMethod NM>
inline double NodalDiffusionDriver<NM>::power(double x, double y,
                                              double z) const {
  // If problem isn't solved yet, we error
  if (solved_ == false) {
    auto mssg = "Cannot compute power. Problem has not been solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Get geometry index
  const auto oi = geom_->x_to_i(x);
  const auto oj = geom_->y_to_j(y);
  const auto ok = geom_->z_to_k(z);
  if (oi.has_value() == false || oj.has_value() == false ||
      ok.has_value() == false)
    return 0.;
  const std::size_t i = oi.value();
  const std::size_t j = oj.value();
  const std::size_t k = ok.value();
  const xt::svector<std::size_t> geom_inds{i, j, k};

  // Get material index
  const auto om = geom_->geom_to_mat_indx(geom_inds);
  if (om.has_value() == false) return 0.;
  const std::size_t m = om.value();

  const auto& xs = *mats_[m].xs;

  double pwr = 0.;

  for (std::size_t g = 0; g < NG_; g++) {
    if constexpr (NM::reconstruct_flux) {
      pwr += reconstructed_flux_params_(m, g)(x, y, z) * xs.Ef(g);
    } else {
      pwr += flux_(ind(m, g)) * xs.Ef(g);
    }
  }

  return pwr;
}

template <NodalMethod NM>
inline xt::xtensor<double, 3> NodalDiffusionDriver<NM>::power(
    const xt::xtensor<double, 1>& x, const xt::xtensor<double, 1>& y,
    const xt::xtensor<double, 1>& z) const {
  // If problem isn't solved yet, we error
  if (solved_ == false) {
    auto mssg = "Cannot compute power. Problem has not been solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure x, y, and z have at least 1 coordinate
  if (x.size() == 0) {
    auto mssg = "Array of x coordinates must have at least one entry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  if (y.size() == 0) {
    auto mssg = "Array of y coordinates must have at least one entry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  if (z.size() == 0) {
    auto mssg = "Array of z coordinates must have at least one entry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  xt::xtensor<double, 3> pwr_out;
  pwr_out.resize({x.size(), y.size(), z.size()});
  pwr_out.fill(0.);

#pragma omp parallel for
  for (int ii = 0; ii < static_cast<int>(x.size()); ii++) {
    std::size_t i = static_cast<std::size_t>(ii);
    for (std::size_t j = 0; j < y.size(); j++) {
      for (std::size_t k = 0; k < z.size(); k++) {
        // Get geometry index
        const auto oi = geom_->x_to_i(x[i]);
        const auto oj = geom_->y_to_j(y[j]);
        const auto ok = geom_->z_to_k(z[k]);
        if (oi.has_value() == false || oj.has_value() == false ||
            ok.has_value() == false) {
          continue;
        }
        const std::size_t gi = oi.value();
        const std::size_t gj = oj.value();
        const std::size_t gk = ok.value();
        const xt::svector<std::size_t> geom_inds{gi, gj, gk};

        // Get material index
        const auto om = geom_->geom_to_mat_indx(geom_inds);
        if (om.has_value() == false) {
          continue;
        }
        const std::size_t m = om.value();

        const auto& xs = *mats_[m].xs;

        for (std::size_t g = 0; g < NG_; g++) {
          if constexpr (NM::reconstruct_flux) {
            pwr_out(i, j, k) +=
                reconstructed_flux_params_(m, g)(x[i], y[j], z[k]) * xs.Ef(g);
          } else {
            pwr_out(i, j, k) += flux_(ind(m, g)) * xs.Ef(g);
          }
        }
      }
    }
  }

  return pwr_out;
}

template <NodalMethod NM>
inline xt::xtensor<double, 3> NodalDiffusionDriver<NM>::avg_power() const {
  // If problem isn't solved yet, we error
  if (solved_ == false) {
    auto mssg = "Cannot compute power. Problem has not been solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  const std::size_t nx = geom_->nx();
  const std::size_t ny = geom_->ny();
  const std::size_t nz = geom_->nz();

  xt::xtensor<double, 3> pwr_out;
  pwr_out.resize({nx, ny, nz});
  pwr_out.fill(0.);

  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        const auto om = geom_->geom_to_mat_indx({i, j, k});

        if (om.has_value() == false) {
          continue;
        }
        const std::size_t m = om.value();
        const auto& xs = *mats_[m].xs;

        for (std::size_t g = 0; g < NG_; g++) {
          pwr_out(i, j, k) += flux_(ind(m, g)) * xs.Ef(g);
        }
      }
    }
  }

  return pwr_out;
}

//=============================================================================
// Flux Reconstruction Methods

template <NodalMethod NM>
inline IntraNodalFlux NodalDiffusionDriver<NM>::fit_node_recon_params(
    std::size_t g, std::size_t m) const
  requires(NM::reconstruct_flux)
{
  // Get node parameters
  const Node& node = nodes_(m, g);
  const auto geom_indx = geom_->geom_indx(m);
  const double dx = geom_->dx(geom_indx[0]);
  const double dy = geom_->dy(geom_indx[1]);
  const double dz = geom_->dz(geom_indx[2]);
  const auto& xs = *mats_[m].xs;
  const double D = xs.D(g);
  const double Er = xs.Er(g);
  const double eps = std::sqrt(Er / D);

  const double x_low = geom_->x_bounds()[geom_indx[0]];
  const double x_hi = geom_->x_bounds()[geom_indx[0] + 1];
  const double y_low = geom_->y_bounds()[geom_indx[1]];
  const double y_hi = geom_->y_bounds()[geom_indx[1] + 1];
  const double z_low = geom_->z_bounds()[geom_indx[2]];
  const double z_hi = geom_->z_bounds()[geom_indx[2] + 1];

  auto sinhc = [](double x) { return std::sinh(x) / x; };

  IntraNodalFlux nf;
  nf.phi_0 = node.phi0();
  nf.eps = eps;
  nf.xm = 0.5 * (x_low + x_hi);
  nf.ym = 0.5 * (y_low + y_hi);
  nf.zm = 0.5 * (z_low + z_hi);
  nf.invs_dx = 1. / dx;
  nf.invs_dy = 1. / dy;
  nf.invs_dz = 1. / dz;

  // Initial base matrix for finding fx, fy, and fz coefficients
  Eigen::Matrix<double, 4, 4> M{
      {0., 0., 1., 1.}, {0., 0., -1., 1.}, {0., 0., 1., 3.}, {0., 0., 1., -3.}};
  Eigen::Matrix<double, 4, 1> b;
  Eigen::Matrix<double, 4, 1> fu_coeffs;

  // Determine fx coefficients
  const double zeta_x = 0.5 * eps * dx;
  M(0, 0) = std::cosh(zeta_x) - sinhc(zeta_x);
  M(0, 1) = std::sinh(zeta_x);
  M(1, 0) = M(0, 0);
  M(1, 1) = -M(0, 1);
  M(2, 0) = zeta_x * std::sinh(zeta_x);
  M(2, 1) = zeta_x * std::cosh(zeta_x);
  M(3, 0) = -M(2, 0);
  M(3, 1) = M(2, 1);
  b(0) = node.phi_xp() - node.phi0();
  b(1) = node.phi_xn() - node.phi0();
  b(2) = -0.5 * node.J_xp() * dx / D;
  b(3) = -0.5 * node.J_xn() * dx / D;
  fu_coeffs = M.inverse() * b;
  nf.ax1 = fu_coeffs(0);
  nf.ax2 = fu_coeffs(1);
  nf.bx1 = fu_coeffs(2);
  nf.bx2 = fu_coeffs(3);
  nf.ax0 = -nf.ax1 * sinhc(zeta_x);
  nf.zeta_x = zeta_x;

  // Determine fy coefficients
  const double zeta_y = 0.5 * eps * dy;
  M(0, 0) = std::cosh(zeta_y) - sinhc(zeta_y);
  M(0, 1) = std::sinh(zeta_y);
  M(1, 0) = M(0, 0);
  M(1, 1) = -M(0, 1);
  M(2, 0) = zeta_y * std::sinh(zeta_y);
  M(2, 1) = zeta_y * std::cosh(zeta_y);
  M(3, 0) = -M(2, 0);
  M(3, 1) = M(2, 1);
  b(0) = node.phi_yp() - node.phi0();
  b(1) = node.phi_yn() - node.phi0();
  b(2) = -0.5 * node.J_yp() * dy / D;
  b(3) = -0.5 * node.J_yn() * dy / D;
  fu_coeffs = M.inverse() * b;
  nf.ay1 = fu_coeffs(0);
  nf.ay2 = fu_coeffs(1);
  nf.by1 = fu_coeffs(2);
  nf.by2 = fu_coeffs(3);
  nf.ay0 = -nf.ay1 * sinhc(zeta_y);
  nf.zeta_y = zeta_y;

  // Determine fz coefficients
  const double zeta_z = 0.5 * eps * dz;
  M(0, 0) = std::cosh(zeta_z) - sinhc(zeta_z);
  M(0, 1) = std::sinh(zeta_z);
  M(1, 0) = M(0, 0);
  M(1, 1) = -M(0, 1);
  M(2, 0) = zeta_z * std::sinh(zeta_z);
  M(2, 1) = zeta_z * std::cosh(zeta_z);
  M(3, 0) = -M(2, 0);
  M(3, 1) = M(2, 1);
  b(0) = node.phi_zp() - node.phi0();
  b(1) = node.phi_zn() - node.phi0();
  b(2) = -0.5 * node.J_zp() * dz / D;
  b(3) = -0.5 * node.J_zn() * dz / D;
  fu_coeffs = M.inverse() * b;
  nf.az1 = fu_coeffs(0);
  nf.az2 = fu_coeffs(1);
  nf.bz1 = fu_coeffs(2);
  nf.bz2 = fu_coeffs(3);
  nf.az0 = -nf.az1 * sinhc(zeta_z);

  return nf;
}

template <NodalMethod NM>
inline void NodalDiffusionDriver<NM>::fit_node_recon_params_corners(
    std::size_t g, std::size_t m)
  requires(NM::reconstruct_flux)
{
  const auto geom_indx = geom_->geom_indx(m);

  const double x_low = geom_->x_bounds()[geom_indx[0]];
  const double x_hi = geom_->x_bounds()[geom_indx[0] + 1];
  const double y_low = geom_->y_bounds()[geom_indx[1]];
  const double y_hi = geom_->y_bounds()[geom_indx[1] + 1];

  IntraNodalFlux& nf = reconstructed_flux_params_(m, g);

  // If the corner point we are looking at is along an outer boundary,
  // we do not compute the average value of the flux, but instead use
  // the value estimate by the previous node reconstruction, without
  // any cross terms. This allows the use of and f(x,y) term in the flux
  // reconstruction on boundary nodes, without leading to the cusps that
  // would occur when trying to take the average.

  // Determine fxy coefficients
  double flx_pp, flx_pm, flx_mp, flx_mm;
  if (geom_indx[0] != geom_->nx() - 1 && geom_indx[1] != geom_->ny() - 1) {
    flx_pp = avg_xy_corner_flux(g, m, Corner::PP);
  } else {
    flx_pp = nf.flux_xy_no_cross(x_hi, y_hi);
  }

  if (geom_indx[0] != geom_->nx() - 1 && geom_indx[1] != 0) {
    flx_pm = avg_xy_corner_flux(g, m, Corner::PM);
  } else {
    flx_pm = nf.flux_xy_no_cross(x_hi, y_low);
  }

  if (geom_indx[0] != 0 && geom_indx[1] != geom_->ny() - 1) {
    flx_mp = avg_xy_corner_flux(g, m, Corner::MP);
  } else {
    flx_mp = nf.flux_xy_no_cross(x_low, y_hi);
  }

  if (geom_indx[0] != 0 && geom_indx[1] != 0) {
    flx_mm = avg_xy_corner_flux(g, m, Corner::MM);
  } else {
    flx_mm = nf.flux_xy_no_cross(x_low, y_low);
  }

  double pp = flx_pp - nf.flux_xy_no_cross(x_hi, y_hi);
  double pm = flx_pm - nf.flux_xy_no_cross(x_hi, y_low);
  double mp = flx_mp - nf.flux_xy_no_cross(x_low, y_hi);
  double mm = flx_mm - nf.flux_xy_no_cross(x_low, y_low);

  nf.cxy11 = 0.25 * (pp - pm + mm - mp);
  nf.cxy12 = 0.25 * (pp + pm - mm - mp);
  nf.cxy21 = 0.25 * (pp - pm - mm + mp);
  nf.cxy22 = 0.25 * (pp + pm + mm + mp);
}

template <NodalMethod NM>
inline double NodalDiffusionDriver<NM>::eval_heter_xy_corner_flux(
    std::size_t g, std::size_t m, Corner c) const
  requires(NM::reconstruct_flux)
{
  const IntraNodalFlux& nf = reconstructed_flux_params_(m, g);
  const double dx = 1. / nf.invs_dx;
  const double x_hi = nf.xm + 0.5 * dx;
  const double x_low = nf.xm - 0.5 * dx;
  const double dy = 1. / nf.invs_dy;
  const double y_hi = nf.ym + 0.5 * dy;
  const double y_low = nf.ym - 0.5 * dy;

  // Since the definition of the CDF is
  // CDF = flux_het / flux_hom
  // we compute the heterogeneous flux as flux_het = flux_hom * CDF

  switch (c) {
    case Corner::PP:
      return nf.flux_xy_no_cross(x_hi, y_hi) * geom_->cdf_I(m, g);
      break;

    case Corner::PM:
      return nf.flux_xy_no_cross(x_hi, y_low) * geom_->cdf_IV(m, g);
      break;

    case Corner::MP:
      return nf.flux_xy_no_cross(x_low, y_hi) * geom_->cdf_II(m, g);
      break;

    case Corner::MM:
      return nf.flux_xy_no_cross(x_low, y_low) * geom_->cdf_III(m, g);
      break;
  }

  // NEVER GETS HERE
  return 0.;
}

template <NodalMethod NM>
inline double NodalDiffusionDriver<NM>::avg_xy_corner_flux(std::size_t g,
                                                           std::size_t m,
                                                           Corner c) const
  requires(NM::reconstruct_flux)
{
  const auto geom_inds = geom_->geom_indx(m);

  double num = 0.;
  double denom = 0.;

  // First, we add our contribution to the corner flux estimation
  num += eval_heter_xy_corner_flux(g, m, c);
  denom += 1.;

  if (c == Corner::PP) {
    const auto& n_xp = neighbors_(m, Side::XP);
    const auto& n_yp = neighbors_(m, Side::YP);

    if (n_xp.second) {
      num += eval_heter_xy_corner_flux(g, n_xp.second.value(), Corner::MP);
      denom += 1.;
    }
    if (n_yp.second) {
      num += eval_heter_xy_corner_flux(g, n_yp.second.value(), Corner::PM);
      denom += 1.;
    }

    const auto om = geom_->geom_to_mat_indx(
        {geom_inds[0] + 1, geom_inds[1] + 1, geom_inds[2]});
    if (om) {
      num += eval_heter_xy_corner_flux(g, om.value(), Corner::MM);
      denom += 1.;
    }
  } else if (c == Corner::PM) {
    const auto& n_xp = neighbors_(m, Side::XP);
    const auto& n_ym = neighbors_(m, Side::YN);

    if (n_xp.second) {
      num += eval_heter_xy_corner_flux(g, n_xp.second.value(), Corner::MM);
      denom += 1.;
    }
    if (n_ym.second) {
      num += eval_heter_xy_corner_flux(g, n_ym.second.value(), Corner::PP);
      denom += 1.;
    }

    const auto om = geom_->geom_to_mat_indx(
        {geom_inds[0] + 1, geom_inds[1] - 1, geom_inds[2]});
    if (om) {
      num += eval_heter_xy_corner_flux(g, om.value(), Corner::MP);
      denom += 1.;
    }
  } else if (c == Corner::MM) {
    const auto& n_xm = neighbors_(m, Side::XN);
    const auto& n_ym = neighbors_(m, Side::YN);

    if (n_xm.second) {
      num += eval_heter_xy_corner_flux(g, n_xm.second.value(), Corner::PM);
      denom += 1.;
    }
    if (n_ym.second) {
      num += eval_heter_xy_corner_flux(g, n_ym.second.value(), Corner::MP);
      denom += 1.;
    }

    const auto om = geom_->geom_to_mat_indx(
        {geom_inds[0] - 1, geom_inds[1] - 1, geom_inds[2]});
    if (om) {
      num += eval_heter_xy_corner_flux(g, om.value(), Corner::PP);
      denom += 1.;
    }
  } else {  // c = Corner::MP
    const auto& n_xm = neighbors_(m, Side::XN);
    const auto& n_yp = neighbors_(m, Side::YP);

    if (n_xm.second) {
      num += eval_heter_xy_corner_flux(g, n_xm.second.value(), Corner::PP);
      denom += 1.;
    }
    if (n_yp.second) {
      num += eval_heter_xy_corner_flux(g, n_yp.second.value(), Corner::MM);
      denom += 1.;
    }

    const auto om = geom_->geom_to_mat_indx(
        {geom_inds[0] - 1, geom_inds[1] + 1, geom_inds[2]});
    if (om) {
      num += eval_heter_xy_corner_flux(g, om.value(), Corner::PM);
      denom += 1.;
    }
  }

  const double avg_het_flx = num / denom;

  // The homogeneous flux is then flux_hom = flux_het / CDF
  double CDF = 1.;
  switch (c) {
    case Corner::PP:
      CDF = geom_->cdf_I(m, g);
      break;

    case Corner::PM:
      CDF = geom_->cdf_IV(m, g);
      break;

    case Corner::MP:
      CDF = geom_->cdf_II(m, g);
      break;

    case Corner::MM:
      CDF = geom_->cdf_III(m, g);
      break;
  }

  return avg_het_flx / CDF;
}

template <NodalMethod NM>
inline void NodalDiffusionDriver<NM>::perform_flux_reconstruction()
  requires(NM::reconstruct_flux)
{
  Timer fitting_timer;
  fitting_timer.start();
  spdlog::info("Fitting flux reconstruction parameters");
  reconstructed_flux_params_.resize({NM_, NG_});
#pragma omp parallel for
  for (int im = 0; im < static_cast<int>(NM_); im++) {
    std::size_t m = static_cast<std::size_t>(im);
    for (std::size_t g = 0; g < NG_; g++) {
      reconstructed_flux_params_(m, g) = fit_node_recon_params(g, m);
    }
  }
#pragma omp parallel for
  for (int im = 0; im < static_cast<int>(NM_); im++) {
    std::size_t m = static_cast<std::size_t>(im);
    for (std::size_t g = 0; g < NG_; g++) {
      fit_node_recon_params_corners(g, m);
    }
  }
  fitting_timer.stop();
  spdlog::info("Fitting Time: {:.5E} s", fitting_timer.elapsed_time());
}

template <NodalMethod NM>
inline void NodalDiffusionDriver<NM>::save(const std::string& fname) {
  if (std::filesystem::exists(fname)) {
    std::filesystem::remove(fname);
  }

  std::ofstream file(fname, std::ios_base::binary);

  cereal::PortableBinaryOutputArchive arc(file);

  arc(*this);
}

template <NodalMethod NM>
inline std::unique_ptr<NodalDiffusionDriver<NM>> NodalDiffusionDriver<NM>::load(
    const std::string& fname) {
  if (std::filesystem::exists(fname) == false) {
    std::stringstream mssg;
    mssg << "The file \"" << fname << "\" does not exist.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  std::unique_ptr<NodalDiffusionDriver> out(new NodalDiffusionDriver());

  std::ifstream file(fname, std::ios_base::binary);

  cereal::PortableBinaryInputArchive arc(file);

  arc(*out);

  return out;
}

}  // namespace scarabee

#endif
