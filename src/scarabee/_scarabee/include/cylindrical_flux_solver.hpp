#ifndef SCARABEE_CYLINDRICAL_FLUX_SOLVER_H
#define SCARABEE_CYLINDRICAL_FLUX_SOLVER_H

#include <cylindrical_cell.hpp>
#include <data/cross_section.hpp>
#include <utils/serialization.hpp>
#include <utils/simulation_mode.hpp>

#include <xtensor/containers/xtensor.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

#include <memory>
#include <vector>

struct CylindricalFluxSolverPickler;

namespace scarabee {

class CylindricalFluxSolver {
 public:
  CylindricalFluxSolver(std::shared_ptr<CylindricalCell> cell);

  const std::shared_ptr<CylindricalCell>& cell() const { return cell_; }

  std::size_t ngroups() const { return cell_->ngroups(); }
  std::size_t nregions() const { return cell_->nregions(); }
  std::size_t size() const { return this->nregions(); }
  void solve(bool parallel = false);
  bool solved() const { return solved_; }

  double flux(std::size_t i, std::size_t g) const { return flux_(g, i); }
  double volume(std::size_t i) const { return cell_->volume(i); }
  const std::shared_ptr<CrossSection>& xs(std::size_t i) const {
    return cell_->xs(i);
  }

  double extern_src(std::size_t i, std::size_t g) const {
    return extern_source_(g, i);
  }
  void set_extern_src(std::size_t i, std::size_t g, double src) {
    extern_source_(g, i) = src;
  }

  SimulationMode& sim_mode() { return mode_; }
  const SimulationMode& sim_mode() const { return mode_; }

  std::shared_ptr<CrossSection> homogenize() const;
  std::shared_ptr<CrossSection> homogenize(std::size_t i_max) const;
  std::shared_ptr<CrossSection> homogenize(
      const std::vector<std::size_t>& regions) const;

  xt::xtensor<double, 1> homogenize_flux_spectrum() const;
  xt::xtensor<double, 1> homogenize_flux_spectrum(std::size_t i_max) const;
  xt::xtensor<double, 1> homogenize_flux_spectrum(
      const std::vector<std::size_t>& regions) const;

  double flux_tolerance() const { return flux_tol_; }
  void set_flux_tolerance(double ftol);

  double keff() const { return k_; }
  double keff_tolerance() const { return k_tol_; }
  void set_keff_tolerance(double ktol);

  double albedo() const { return a_; }
  void set_albedo(double a);

  double j_ext(std::size_t g) const { return j_ext_[g]; }
  void set_j_ext(std::size_t g, double j) {
    j_ext_[g] = j;
    solved_ = false;
  }

  double j_neg(std::size_t g) const {
    return (a_ * x_[g] + j_ext_[g]) / (1. - a_ * (1. - cell_->Gamma(g)));
  }

  double j_pos(std::size_t g) const {
    return (x_[g] + (1. - cell_->Gamma(g)) * j_ext_[g]) /
           (1. - a_ * (1. - cell_->Gamma(g)));
  }

  double j(std::size_t g) const {
    return ((1. - a_) * x_[g] - cell_->Gamma(g) * j_ext_[g]) /
           (1. - a_ * (1. - cell_->Gamma(g)));
  }

 private:
  xt::xtensor<double, 2> flux_;
  xt::xtensor<double, 2> extern_source_;
  xt::xtensor<double, 1> j_ext_;
  xt::xtensor<double, 1> x_;
  std::shared_ptr<CylindricalCell> cell_;
  double k_;
  double a_;
  double k_tol_;
  double flux_tol_;
  SimulationMode mode_;
  bool solved_;

  double calc_keff(const xt::xtensor<double, 2>& flux) const;
  double calc_flux_rel_diff(const xt::xtensor<double, 2>& flux,
                            const xt::xtensor<double, 2>& next_flux) const;
  void fill_fission_source(xt::xtensor<double, 2>& source,
                           const xt::xtensor<double, 2>& flux) const;
  void fill_scatter_source(xt::xtensor<double, 2>& source,
                           const xt::xtensor<double, 2>& flux) const;
  double Qscat(std::size_t g, std::size_t i,
               const xt::xtensor<double, 2>& flux) const;
  double Qfiss(std::size_t g, std::size_t i,
               const xt::xtensor<double, 2>& flux) const;

  void solve_single_thread();
  void solve_parallel();

  friend class cereal::access;
  friend struct ::CylindricalFluxSolverPickler;
  CylindricalFluxSolver() {}
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(flux_), CEREAL_NVP(extern_source_), CEREAL_NVP(j_ext_),
        CEREAL_NVP(x_), CEREAL_NVP(cell_), CEREAL_NVP(k_), CEREAL_NVP(a_),
        CEREAL_NVP(k_tol_), CEREAL_NVP(flux_tol_), CEREAL_NVP(mode_),
        CEREAL_NVP(solved_));
  }
};

}  // namespace scarabee

#endif
