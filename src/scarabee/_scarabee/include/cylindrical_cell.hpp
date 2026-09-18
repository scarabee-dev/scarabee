#ifndef SCARABEE_CYLINDRICAL_CELL_H
#define SCARABEE_CYLINDRICAL_CELL_H

#include <data/cross_section.hpp>
#include <utils/constants.hpp>
#include <utils/serialization.hpp>

#include <xtensor/containers/xtensor.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

#include <memory>
#include <vector>

struct CylindricalCellPickler;

namespace scarabee {

class CylindricalCell {
 public:
  CylindricalCell(const std::vector<double>& radii,
                  const std::vector<std::shared_ptr<CrossSection>>& mats);

  bool solved() const { return solved_; }
  void solve(bool parallel = false);

  std::size_t ngroups() const { return ngroups_; }
  std::size_t nregions() const { return radii_.size(); }

  // Surface area of cell
  double Sb() const { return 2. * PI * radii_.back(); }

  // Volume of region i
  double volume(std::size_t i) const { return vols_[i]; }

  // Outer radius of region i
  double radius(std::size_t i) const { return radii_[i]; }

  double Y(double a, std::size_t g, std::size_t i) const {
    const double Yi = Y_(g, i);
    const double Gamma = Gamma_(g);

    return Yi / (1. - a * (1. - Gamma));
  }

  double x(std::size_t g, std::size_t i) const {
    return 0.25 * Sb() * volume(i) * Y_(g, i);
  }

  double X(double a, std::size_t g, std::size_t i, std::size_t k) const {
    const double xk = x(g, k);
    const double Xik = X_(g, i, k);

    return Xik + a * xk * Y(a, g, i);
  }

  double Gamma(std::size_t g) const { return Gamma_(g); }

  double p(std::size_t g, std::size_t i, std::size_t j) const {
    return p_(g, i, j);
  }

  const std::shared_ptr<CrossSection>& xs(std::size_t i) const {
    return mats_[i];
  }

 private:
  xt::xtensor<double, 3> p_;
  xt::xtensor<double, 3> X_;
  xt::xtensor<double, 2> Y_;
  xt::xtensor<double, 1> Gamma_;  // Multicollision blackness in each group
  std::vector<double> radii_;
  std::vector<double> vols_;
  std::vector<std::shared_ptr<CrossSection>> mats_;
  std::size_t ngroups_;
  bool solved_;

  void calculate_collision_probabilities();
  void solve_systems();

  void parallel_calculate_collision_probabilities();
  void parallel_solve_systems();

  double calculate_S_ij(std::size_t i, std::size_t j, std::size_t g) const;

  friend cereal::access;
  friend struct ::CylindricalCellPickler;
  CylindricalCell() {}
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(p_), CEREAL_NVP(X_), CEREAL_NVP(Y_), CEREAL_NVP(Gamma_),
        CEREAL_NVP(radii_), CEREAL_NVP(vols_), CEREAL_NVP(mats_),
        CEREAL_NVP(ngroups_), CEREAL_NVP(solved_));
  }
};

}  // namespace scarabee

#endif
