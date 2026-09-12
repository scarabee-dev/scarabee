#ifndef REFLECTOR_SN_H
#define REFLECTOR_SN_H

#include <data/cross_section.hpp>

#include <xtensor/containers/xtensor.hpp>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <span>
#include <memory>

namespace scarabee {

class ReflectorSN {
 public:
  ReflectorSN(const std::vector<std::shared_ptr<CrossSection>>& xs,
              const xt::xtensor<double, 1>& dx, std::uint32_t nangles,
              bool anisotropic);

  ReflectorSN(py::tuple t);

  std::size_t nangles() const { return mu_.size(); }

  void solve();
  bool solved() const { return solved_; }

  double keff() const { return keff_; }

  double keff_tolerance() const { return keff_tol_; }
  void set_keff_tolerance(double ktol);

  double flux_tolerance() const { return flux_tol_; }
  void set_flux_tolerance(double ftol);

  std::size_t size() const { return xs_.size(); }
  std::size_t nregions() const { return xs_.size(); }
  std::size_t nsurfaces() const { return xs_.size() + 1; }
  std::size_t ngroups() const { return ngroups_; }
  std::size_t max_legendre_order() const { return max_L_; }
  bool anisotropic() const { return anisotropic_; }

  const std::shared_ptr<CrossSection> xs(std::size_t i) const;
  double volume(std::size_t i) const;

  double flux(std::size_t i, std::size_t g, std::size_t l = 0) const;
  double current(std::size_t i, std::size_t g) const;

  std::shared_ptr<CrossSection> homogenize(
      const std::vector<std::size_t>& regions) const;
  xt::xtensor<double, 1> homogenize_flux_spectrum(
      const std::vector<std::size_t>& regions) const;

  py::tuple to_tuple() const;

 private:
  std::vector<std::shared_ptr<CrossSection>> xs_;
  xt::xtensor<double, 1> dx_;
  xt::xtensor<double, 3> flux_;  // group, spatial bin, legendre moment
  xt::xtensor<double, 2> J_;     // group, surface
  xt::xtensor<double, 2> Pnl_;   // direction index, legendre moment
  double keff_{1.};
  double keff_tol_{1.E-5};
  double flux_tol_{1.E-5};
  std::size_t ngroups_;
  std::size_t max_L_ = 0;  // max-legendre-order in scattering moments
  bool solved_{false};
  bool anisotropic_{false};

  void solve_iso();
  void sweep_iso(xt::xtensor<double, 3>& flux,
                 xt::xtensor<double, 2>& incident_angular_flux,
                 const xt::xtensor<double, 3>& Q);
  void fill_source_iso(xt::xtensor<double, 3>& Q,
                       const xt::xtensor<double, 3>& flux) const;

  void solve_aniso();
  void sweep_aniso(xt::xtensor<double, 3>& flux,
                   xt::xtensor<double, 2>& incident_angular_flux,
                   const xt::xtensor<double, 3>& Q);
  void fill_source_aniso(xt::xtensor<double, 3>& Q,
                         const xt::xtensor<double, 3>& flux) const;

  double calc_keff(const xt::xtensor<double, 3>& old_flux,
                   const xt::xtensor<double, 3>& new_flux,
                   const double keff) const;

  std::span<const double> mu_;
  std::span<const double> wgt_;
};

}  // namespace scarabee

#endif
