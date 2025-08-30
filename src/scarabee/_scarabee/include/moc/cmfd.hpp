#ifndef SCARABEE_CMFD_H
#define SCARABEE_CMFD_H

#include <moc/surface.hpp>
#include <moc/vector.hpp>
#include <moc/direction.hpp>
#include <moc/boundary_condition.hpp>
#include <data/diffusion_cross_section.hpp>
#include <utils/simulation_mode.hpp>

#include <xtensor/containers/xtensor.hpp>
#include <Eigen/Sparse>

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/utility.hpp>
#include <utils/serialization.hpp>

#include <array>
#include <memory>
#include <utility>
#include <optional>
#include <variant>
#include <vector>
#include <set>

namespace scarabee {

class MOCDriver;

struct CMFDSurfaceCrossing {
  enum class Type : std::uint8_t { XN, XP, YN, YP, I, II, III, IV };
  std::size_t cell_index{0};
  bool is_valid{false};
  Type crossing;

  constexpr explicit operator bool() const noexcept { return is_valid; }

  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(cell_index), CEREAL_NVP(is_valid), CEREAL_NVP(crossing));
  }
};

class CMFD {
 public:
  CMFD(const std::vector<double>& dx, const std::vector<double>& dy,
       const std::vector<std::pair<std::size_t, std::size_t>>& groups);

  std::size_t nx() const { return nx_; };
  std::size_t ny() const { return ny_; };

  std::size_t nx_surfs() const { return nx_surfs_; }
  std::size_t ny_surfs() const { return ny_surfs_; }

  double x_min() const;
  double x_max() const;
  double y_min() const;
  double y_max() const;

  const std::vector<double>& dx() const { return dx_; }
  const std::vector<double>& dy() const { return dy_; }

  const std::vector<std::pair<std::size_t, std::size_t>>& condensation_scheme()
      const {
    return group_condensation_;
  }

  std::optional<std::array<std::size_t, 2>> get_tile(const Vector& r,
                                                     const Direction& u) const;
  std::size_t tile_to_indx(const std::array<std::size_t, 2>& tile) const;
  std::size_t tile_to_indx(const std::size_t& i, const std::size_t& j) const;

  std::array<std::size_t, 2> indx_to_tile(std::size_t cell_index);

  CMFDSurfaceCrossing get_surface(const Vector& r, const Direction& u) const;

  std::size_t get_x_neg_surf(const std::size_t i, const std::size_t j) const;
  std::size_t get_x_pos_surf(const std::size_t i, const std::size_t j) const;
  std::size_t get_y_neg_surf(const std::size_t i, const std::size_t j) const;
  std::size_t get_y_pos_surf(const std::size_t i, const std::size_t j) const;

  void insert_fsr(const std::array<std::size_t, 2>& tile, std::size_t fsr);
  void insert_fsr(std::size_t tile_indx, std::size_t fsr);
  void pack_fsr_lists();

  const std::vector<std::size_t>& tile_fsr_list(std::size_t i,
                                                std::size_t j) const;

  std::size_t moc_to_cmfd_group(std::size_t g) const;

  const double& current(const std::size_t G, const std::size_t surface) const;

  void tally_current(double aflx, const Direction& u, std::size_t G,
                     const CMFDSurfaceCrossing& surf);

  void zero_currents() {
    surface_currents_.fill(0.);
    surface_currents_normalized_ = false;
  }

  enum class TileSurf : std::uint8_t { XN, XP, YN, YP };

  void solve(MOCDriver& moc, double keff, std::size_t moc_iteration);

  void homogenize_ext_src(const MOCDriver& moc);

  // Setter/Getter functions

  double keff_tolerance() const { return keff_tol_; }
  void set_keff_tolerance(double ktol);

  double flux_tolerance() const { return flux_tol_; }
  void set_flux_tolerance(double ftol);

  double damping() const { return damping_; }
  void set_damping(double wd);

  bool flux_limiting() const { return flux_limiting_; }
  void set_flux_limiting(bool user_pref) { flux_limiting_ = user_pref; }

  bool larsen_correction() const { return larsen_correction_; }
  void set_larsen_correction(bool user_pref);

  bool od_cmfd() const { return od_cmfd_; }
  void set_od_cmfd(bool user_pref);

  bool neutron_balance_check() const { return neutron_balance_check_; }
  void set_neutron_balance_check(bool user_pref) {
    neutron_balance_check_ = user_pref;
  }

  std::size_t num_unbounded_solves() const { return unbounded_cmfd_solves_; }
  void set_num_unbounded_solves(std::size_t num_solves) {
    unbounded_cmfd_solves_ = num_solves;
  }

  std::size_t skip_moc_iterations() const { return skip_moc_iterations_; }
  void set_skip_moc_iterations(std::size_t num_iter) {
    skip_moc_iterations_ = num_iter;
  }

  std::size_t moc_iteration() const { return moc_iteration_; }

  const double& flux(const std::size_t i, const std::size_t j,
                     const std::size_t g) const;
  double keff() const { return keff_; }

  const double& solve_time() const { return solve_time_; }
  bool solved() const { return solved_; }

 private:
  std::vector<double> dx_, dy_;
  std::vector<XPlane> x_bounds_;
  std::vector<YPlane> y_bounds_;
  std::vector<std::size_t> moc_to_cmfd_group_map_;
  std::vector<std::pair<std::size_t, std::size_t>> group_condensation_;
  std::size_t nx_, ny_, ng_;
  std::size_t nx_surfs_, ny_surfs_;

  bool flux_limiting_ = true;
  bool larsen_correction_ = false;
  bool od_cmfd_ = true;
  bool neutron_balance_check_ = false;
  double keff_tol_ = 1E-5;
  double flux_tol_ = 1E-5;
  double damping_ = 0.7;
  std::size_t unbounded_cmfd_solves_ = 1;
  std::size_t cmfd_solves_ = 0;
  std::size_t skip_moc_iterations_ = 0;
  std::size_t moc_iteration_;
  double keff_ = 1.0;
  double solve_time_ = 0.0;
  bool solved_ = false;
  SimulationMode mode_{SimulationMode::Keff};

  // List of flat source region indices for each CMFD cell
  std::vector<std::set<std::size_t>> temp_fsrs_;
  std::vector<std::vector<std::size_t>> fsrs_;

  // This contains the net current for every possible surface in every group.
  // The first index is the CMFG group, and the second is the surface ID.
  // Surfaces are ordered as all x surfaces, then all y surfaces.
  // Number of surfaces is then ny_*x_bounds_.size() + nx_*y_bounds_.size().
  xt::xtensor<double, 2> surface_currents_;  // group, surface
  bool surface_currents_normalized_ = false;

  xt::xtensor<std::shared_ptr<DiffusionCrossSection>, 2> xs_;
  xt::xtensor<double, 3> Et_;             // g, i, j
  xt::xtensor<double, 3> flux_;           // g, x, y
  xt::xtensor<double, 2> D_transp_corr_;  // g, surf

  Eigen::VectorXd flux_cmfd_;      // g*nx_*ny_
  Eigen::VectorXd update_ratios_;  // g*nx_*ny_
  Eigen::VectorXd volumes_;        // nx_*ny_

  Eigen::SparseMatrix<double> M_;   // Loss Matrix
  Eigen::SparseMatrix<double> QM_;  // Source Matrix

  Eigen::VectorXd extern_src_;  // g*nx_*ny_

  void apply_larsen_correction(double& D, const double dx,
                               const MOCDriver& moc) const;
  void optimize_diffusion_coef(double& D, const double dx, const std::size_t i,
                               const std::size_t j, const std::size_t g) const;
  std::pair<double, double> calc_surf_diffusion_coeffs(
      std::size_t i, std::size_t j, std::size_t g, TileSurf surf,
      const MOCDriver& moc) const;
  std::variant<std::array<std::size_t, 2>, BoundaryCondition>
  find_next_cell_or_bc(std::size_t i, std::size_t j, TileSurf surf,
                       const MOCDriver& moc) const;
  double get_cmfd_tile_width(std::size_t i, std::size_t j, TileSurf surf) const;
  double get_current(std::size_t i, std::size_t j, std::size_t g,
                     TileSurf surf) const;
  void create_loss_matrix(const MOCDriver& moc);
  void create_source_matrix();
  void power_iteration(double keff);
  void fixed_source_solve();
  void update_moc_fluxes(MOCDriver& moc);
  void normalize_currents();
  void compute_homogenized_xs_and_flux(const MOCDriver& moc);

  void check_neutron_balance(const std::size_t i, const std::size_t j,
                             std::size_t g, const double keff,
                             const MOCDriver& moc) const;
  double calc_surf_current_cmfd(std::size_t i, std::size_t j, std::size_t g,
                                TileSurf surf, const MOCDriver& moc) const;
  double calc_neutron_balance_residual_cmfd(const std::size_t i,
                                            const std::size_t j, std::size_t g,
                                            const double keff,
                                            const MOCDriver& moc) const;
  double calc_neutron_balance_residual_moc(const std::size_t i,
                                           const std::size_t j, std::size_t g,
                                           const double keff) const;

  // Private default constructor, needed for cereal
  CMFD() = default;

  friend class cereal::access;
  template <class Archive>
  void save(Archive& arc) const {
    arc(CEREAL_NVP(dx_), CEREAL_NVP(dy_), CEREAL_NVP(x_bounds_),
        CEREAL_NVP(y_bounds_), CEREAL_NVP(moc_to_cmfd_group_map_),
        CEREAL_NVP(group_condensation_), CEREAL_NVP(nx_), CEREAL_NVP(ny_),
        CEREAL_NVP(ng_), CEREAL_NVP(nx_surfs_), CEREAL_NVP(ny_surfs_),
        CEREAL_NVP(flux_limiting_), CEREAL_NVP(larsen_correction_),
        CEREAL_NVP(od_cmfd_), CEREAL_NVP(neutron_balance_check_),
        CEREAL_NVP(keff_tol_), CEREAL_NVP(flux_tol_), CEREAL_NVP(damping_),
        CEREAL_NVP(unbounded_cmfd_solves_), CEREAL_NVP(cmfd_solves_),
        CEREAL_NVP(skip_moc_iterations_), CEREAL_NVP(moc_iteration_),
        CEREAL_NVP(keff_), CEREAL_NVP(solve_time_), CEREAL_NVP(solved_),
        CEREAL_NVP(mode_), CEREAL_NVP(fsrs_), CEREAL_NVP(surface_currents_),
        CEREAL_NVP(surface_currents_normalized_), CEREAL_NVP(xs_),
        CEREAL_NVP(Et_), CEREAL_NVP(flux_), CEREAL_NVP(D_transp_corr_));
  }

  template <class Archive>
  void load(Archive& arc) {
    arc(CEREAL_NVP(dx_), CEREAL_NVP(dy_), CEREAL_NVP(x_bounds_),
        CEREAL_NVP(y_bounds_), CEREAL_NVP(moc_to_cmfd_group_map_),
        CEREAL_NVP(group_condensation_), CEREAL_NVP(nx_), CEREAL_NVP(ny_),
        CEREAL_NVP(ng_), CEREAL_NVP(nx_surfs_), CEREAL_NVP(ny_surfs_),
        CEREAL_NVP(flux_limiting_), CEREAL_NVP(larsen_correction_),
        CEREAL_NVP(od_cmfd_), CEREAL_NVP(neutron_balance_check_),
        CEREAL_NVP(keff_tol_), CEREAL_NVP(flux_tol_), CEREAL_NVP(damping_),
        CEREAL_NVP(unbounded_cmfd_solves_), CEREAL_NVP(cmfd_solves_),
        CEREAL_NVP(skip_moc_iterations_), CEREAL_NVP(moc_iteration_),
        CEREAL_NVP(keff_), CEREAL_NVP(solve_time_), CEREAL_NVP(solved_),
        CEREAL_NVP(mode_), CEREAL_NVP(fsrs_), CEREAL_NVP(surface_currents_),
        CEREAL_NVP(surface_currents_normalized_), CEREAL_NVP(xs_),
        CEREAL_NVP(Et_), CEREAL_NVP(flux_), CEREAL_NVP(D_transp_corr_));

    // Must instantiate Eigen bits
    // Set CMFD fluxes to 1
    flux_cmfd_.resize(ng_ * nx_ * ny_);
    flux_cmfd_.setOnes();

    // Allocate flux update ratio array
    update_ratios_.resize(ng_ * nx_ * ny_);
    update_ratios_.setOnes();

    // Allocate external source array
    extern_src_.resize(ng_ * nx_ * ny_);
    extern_src_.setZero();

    // Allocate cell volume array
    volumes_.resize(nx_ * ny_);
    for (std::size_t i = 0; i < nx_; i++) {
      for (std::size_t j = 0; j < ny_; j++) {
        // Store CMFD cell volume
        const auto indx = tile_to_indx(i, j);
        volumes_[indx] = dx_[i] * dy_[j];
      }
    }

    // No need to treat M_ and QM_. These will be reallocated and filled when
    // needed in a CMFD solve.
  }
};

}  // namespace scarabee

#endif
