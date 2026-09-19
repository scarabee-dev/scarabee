#ifndef SCARABEE_ND_LIBRARY_H
#define SCARABEE_ND_LIBRARY_H

#include <data/material.hpp>
#include <data/cross_section.hpp>
#include <data/micro_cross_sections.hpp>
#include <data/depletion_chain.hpp>

#include <xtensor/containers/xtensor.hpp>

#include <highfive/highfive.hpp>

namespace H5 = HighFive;

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace scarabee {

class NDLibrary;

struct NuclideHandle {
  std::string name;
  std::string label;
  std::vector<double> temperatures;
  std::vector<double> dilutions;
  std::vector<double> ir_lambda;
  double awr;
  double potential_xs;
  double fission_energy;
  std::uint32_t ZA;
  bool fissile;
  bool resonant;

  // Packing structure for scattering matrices, both inf and res !
  std::shared_ptr<xt::xtensor<std::uint32_t, 2>> packing;

  // Nu and chi are independent of temperature AND dilution in Scarabée
  std::shared_ptr<xt::xtensor<double, 1>> chi;
  std::shared_ptr<xt::xtensor<double, 1>> nu;

  // Infinite dilution data, only dependent on temperature
  std::shared_ptr<xt::xtensor<double, 2>> inf_absorption;
  std::shared_ptr<xt::xtensor<double, 2>> inf_transport_correction;
  std::shared_ptr<xt::xtensor<double, 2>> inf_scatter;
  std::shared_ptr<xt::xtensor<double, 2>> inf_p1_scatter;
  std::shared_ptr<xt::xtensor<double, 2>> inf_p2_scatter;
  std::shared_ptr<xt::xtensor<double, 2>> inf_p3_scatter;
  std::shared_ptr<xt::xtensor<double, 2>> inf_fission;
  std::shared_ptr<xt::xtensor<double, 2>> inf_n_gamma;
  std::shared_ptr<xt::xtensor<double, 2>> inf_n_2n;
  std::shared_ptr<xt::xtensor<double, 2>> inf_n_3n;
  std::shared_ptr<xt::xtensor<double, 2>> inf_n_a;
  std::shared_ptr<xt::xtensor<double, 2>> inf_n_p;

  // Dilution dependent data
  // First index temperature, second dilution
  std::shared_ptr<xt::xtensor<double, 3>> res_absorption;
  std::shared_ptr<xt::xtensor<double, 3>> res_transport_correction;
  std::shared_ptr<xt::xtensor<double, 3>> res_scatter;
  std::shared_ptr<xt::xtensor<double, 3>> res_p1_scatter;
  std::shared_ptr<xt::xtensor<double, 3>> res_p2_scatter;
  std::shared_ptr<xt::xtensor<double, 3>> res_p3_scatter;
  std::shared_ptr<xt::xtensor<double, 3>> res_fission;
  std::shared_ptr<xt::xtensor<double, 3>> res_n_gamma;

  bool loaded() const { return inf_absorption != nullptr; }
  void load_xs_from_hdf5(const NDLibrary& ndl, std::size_t max_l);
  void load_inf_data(const NDLibrary& ndl, const H5::Group& grp,
                     std::size_t max_l);
  void load_res_data(const H5::Group& grp, std::size_t max_l);
  void unload();
};

class NDLibrary {
 public:
  NDLibrary();
  NDLibrary(const std::string& fname);

  const std::string& fname() const { return fname_; }

  std::size_t ngroups() const { return ngroups_; }

  std::size_t first_resonant_group() const { return first_resonant_group_; }
  std::size_t last_resonant_group() const { return last_resonant_group_; }

  const std::string& library() const { return library_; }

  const std::string& group_structure() const { return group_structure_; }

  const std::vector<double>& group_bounds() const { return group_bounds_; }

  const std::shared_ptr<DepletionChain>& depletion_chain() const {
    return depletion_chain_;
  }

  const std::optional<std::vector<std::pair<std::size_t, std::size_t>>>&
  condensation_scheme() const {
    return condensation_scheme_;
  }

  const std::optional<std::vector<std::pair<std::size_t, std::size_t>>>&
  cmfd_condensation_scheme() const {
    return cmfd_condensation_scheme_;
  }

  NuclideHandle& get_nuclide(const std::string& name);
  const NuclideHandle& get_nuclide(const std::string& name) const;

  std::pair<MicroNuclideXS, MicroDepletionXS> infinite_dilution_xs(
      const std::string& name, const double temp, std::size_t max_l = 1);

  ResonantOneGroupXS dilution_xs(const std::string& name, std::size_t g,
                                 const double temp, const double dil,
                                 std::size_t max_l = 1);

  ResonantOneGroupXS two_term_xs(const std::string& name, std::size_t g,
                                 const double temp, const double b1,
                                 const double b2, const double bg_xs_1,
                                 const double bg_xs_2, std::size_t max_l = 1);

  ResonantOneGroupXS ring_two_term_xs(const std::string& name, std::size_t g,
                                      const double temp, const double a1,
                                      const double a2, const double b1,
                                      const double b2, const double mat_pot_xs,
                                      const double N, const double Rfuel,
                                      const double Rin, const double Rout,
                                      std::size_t max_l = 1);

  const std::shared_ptr<H5::File>& h5() const { return h5_; }

  void unload();

 private:
  std::string fname_;
  std::map<std::string, NuclideHandle> nuclide_handles_;
  std::vector<double> group_bounds_;
  std::optional<std::vector<std::pair<std::size_t, std::size_t>>>
      condensation_scheme_;
  std::optional<std::vector<std::pair<std::size_t, std::size_t>>>
      cmfd_condensation_scheme_;
  std::string library_;
  std::string group_structure_;
  std::size_t ngroups_;
  std::size_t first_resonant_group_;
  std::size_t last_resonant_group_;
  std::shared_ptr<H5::File> h5_;
  std::shared_ptr<DepletionChain> depletion_chain_;

  NDLibrary(const NDLibrary&) = delete;
  NDLibrary& operator=(const NDLibrary&) = delete;

  void init();

  void get_temp_interp_params(double temp, const NuclideHandle& nuc,
                              std::size_t& i, double& f) const;
  void get_dil_interp_params(double dil, const NuclideHandle& nuc,
                             std::size_t& i, double& f) const;

  void interp_temp(xt::xtensor<double, 1>& E, const xt::xtensor<double, 2>& nE,
                   std::size_t it, double f_temp) const;

  double interp_temp_dil(const xt::xtensor<double, 3>& nE, std::size_t g,
                         std::size_t it, double f_temp, std::size_t id,
                         double f_dil) const;

  // E should be a 1D xtensor view
  // nE should be a 3D xtensor view: temp, dil, scat_xs
  void interp_temp_dil_views(auto E, auto nE, std::size_t it, double f_temp,
                             std::size_t id, double f_dil) const {
    if (f_temp > 0.) {
      if (f_dil > 0.) {
        E = (1. - f_temp) * ((1. - f_dil) * xt::view(nE, it, id, xt::all()) +
                             f_dil * xt::view(nE, it, id + 1, xt::all())) +
            f_temp * ((1. - f_dil) * xt::view(nE, it + 1, id, xt::all()) +
                      f_dil * xt::view(nE, it + 1, id + 1, xt::all()));
      } else {
        E = (1. - f_temp) * xt::view(nE, it, id, xt::all()) +
            f_temp * xt::view(nE, it + 1, id, xt::all());
      }
    } else {
      if (f_dil > 0.) {
        E = (1. - f_dil) * xt::view(nE, it, id, xt::all()) +
            f_dil * xt::view(nE, it, id + 1, xt::all());
      } else {
        E = xt::view(nE, it, id, xt::all());
      }
    }
  }

  std::pair<double, double> eta_lm(std::size_t m, double Rfuel, double Rin,
                                   double Rout) const;
};

}  // namespace scarabee

#endif
