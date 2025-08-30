#ifndef SCARABEE_FD_DIFFUSION_DRIVER_H
#define SCARABEE_FD_DIFFUSION_DRIVER_H

#include <diffusion/diffusion_data.hpp>
#include <diffusion/diffusion_geometry.hpp>
#include <utils/serialization.hpp>
#include <utils/simulation_mode.hpp>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>

#include <Eigen/Dense>

#include <memory>
#include <optional>
#include <tuple>

namespace scarabee {

class FDDiffusionDriver {
 public:
  FDDiffusionDriver(std::shared_ptr<DiffusionGeometry> geom);

  std::shared_ptr<DiffusionGeometry> geometry() const { return geom_; }

  std::size_t ngroups() const { return geom_->ngroups(); }

  SimulationMode& sim_mode() { return mode_; }
  const SimulationMode& sim_mode() const { return mode_; }

  void solve();
  bool solved() const { return solved_; }

  double keff_tolerance() const { return keff_tol_; }
  void set_keff_tolerance(double ktol);

  double flux_tolerance() const { return flux_tol_; }
  void set_flux_tolerance(double ftol);

  double keff() const { return keff_; }

  double flux(std::size_t i, std::size_t g) const;
  double flux(std::size_t i, std::size_t j, std::size_t g) const;
  double flux(std::size_t i, std::size_t j, std::size_t k, std::size_t g) const;

  double extern_src(std::size_t i, std::size_t g) const;
  double extern_src(std::size_t i, std::size_t j, std::size_t g) const;
  double extern_src(std::size_t i, std::size_t j, std::size_t k,
                    std::size_t g) const;

  void set_extern_src(std::size_t i, std::size_t g, double src);
  void set_extern_src(std::size_t i, std::size_t j, std::size_t g, double src);
  void set_extern_src(std::size_t i, std::size_t j, std::size_t k,
                      std::size_t g, double src);

  std::tuple<xt::xarray<double>, xt::xarray<double>,
             std::optional<xt::xarray<double>>,
             std::optional<xt::xarray<double>>>
  flux() const;

  std::tuple<xt::xarray<double>, xt::xarray<double>,
             std::optional<xt::xarray<double>>,
             std::optional<xt::xarray<double>>>
  power() const;

  void save(const std::string& fname);
  static std::unique_ptr<FDDiffusionDriver> load(const std::string& fname);

 private:
  std::shared_ptr<DiffusionGeometry> geom_;
  Eigen::VectorXd flux_;        // Flux in each MAT tile in each group
  Eigen::VectorXd extern_src_;  // Source in each MAT tile in each group
  SimulationMode mode_;
  double keff_ = 1.;
  double flux_tol_ = 1.E-5;
  double keff_tol_ = 1.E-5;
  bool solved_{false};

  void power_iteration();
  void fixed_source();

  friend class cereal::access;
  FDDiffusionDriver() {}
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(geom_), CEREAL_NVP(flux_), CEREAL_NVP(keff_),
        CEREAL_NVP(flux_tol_), CEREAL_NVP(keff_tol_), CEREAL_NVP(solved_));
  }
};

}  // namespace scarabee

#endif
