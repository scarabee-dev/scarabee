#ifndef SCARABEE_DIFFUSION_DATA_H
#define SCARABEE_DIFFUSION_DATA_H

#include <data/diffusion_cross_section.hpp>
#include <diffusion/leakage_corrections.hpp>
#include <utils/serialization.hpp>

#include <xtensor/containers/xtensor.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <memory>
#include <optional>
#include <string>

namespace scarabee {

class DiffusionData {
 public:
  enum class ADF : std::size_t {
    XN = 0,
    XP = 1,
    YN = 2,
    YP = 3,
    ZN = 4,
    ZP = 5
  };
  enum class CDF : std::size_t { I = 0, II = 1, III = 2, IV = 3 };

  DiffusionData(std::shared_ptr<DiffusionCrossSection> xs);

  DiffusionData(std::shared_ptr<DiffusionCrossSection> xs,
                const xt::xtensor<double, 2>& adf,
                const xt::xtensor<double, 2>& cdf);

  DiffusionData(py::tuple t)
      : xs_(t[0].cast<std::shared_ptr<DiffusionCrossSection>>()),
        adf_(t[1].cast<xt::xtensor<double, 2>>()),
        cdf_(t[2].cast<xt::xtensor<double, 2>>()),
        name_(t[3].cast<std::string>()),
        leakage_corrections_(t[4].cast<std::optional<LeakageCorrections>>()),
        reflector_(t[5].cast<bool>()) {};

  std::size_t ngroups() const { return xs_->ngroups(); }

  bool fissile() const { return xs_->fissile(); }

  double D(std::size_t g) const { return xs_->D(g); }

  double Ea(std::size_t g) const { return xs_->Ea(g); }

  double Ef(std::size_t g) const { return xs_->Ef(g); }

  double vEf(std::size_t g) const { return xs_->vEf(g); }

  double nu(std::size_t g) const { return xs_->nu(g); }

  double Er(std::size_t g) const { return xs_->Er(g); }

  double chi(std::size_t g) const { return xs_->chi(g); }

  double Es(std::size_t gin, std::size_t gout) const {
    return xs_->Es(gin, gout);
  }

  double Es(std::size_t gin) const { return xs_->Es(gin); }

  double adf_xp(std::size_t g) const {
    return adf_.size() > 0 ? adf_(g, ADF::XP) : 1.;
  }

  double adf_xn(std::size_t g) const {
    return adf_.size() > 0 ? adf_(g, ADF::XN) : 1.;
  }

  double adf_yp(std::size_t g) const {
    return adf_.size() > 0 ? adf_(g, ADF::YP) : 1.;
  }

  double adf_yn(std::size_t g) const {
    return adf_.size() > 0 ? adf_(g, ADF::YN) : 1.;
  }

  double adf_zp(std::size_t g) const {
    return adf_.size() > 0 ? adf_(g, ADF::ZP) : 1.;
  }

  double adf_zn(std::size_t g) const {
    return adf_.size() > 0 ? adf_(g, ADF::ZN) : 1.;
  }

  double cdf_I(std::size_t g) const {
    if (cdf_.size() > 0) {
      return cdf_(g, CDF::I);
    } else if (adf_.size() > 0) {
      return 0.5 * (adf_xp(g) + adf_yp(g));
    } else {
      return 1.;
    }
  }

  double cdf_II(std::size_t g) const {
    if (cdf_.size() > 0) {
      return cdf_(g, CDF::II);
    } else if (adf_.size() > 0) {
      return 0.5 * (adf_xn(g) + adf_yp(g));
    } else {
      return 1.;
    }
  }

  double cdf_III(std::size_t g) const {
    if (cdf_.size() > 0) {
      return cdf_(g, CDF::III);
    } else if (adf_.size() > 0) {
      return 0.5 * (adf_xn(g) + adf_yn(g));
    } else {
      return 1.;
    }
  }

  double cdf_IV(std::size_t g) const {
    if (cdf_.size() > 0) {
      return cdf_(g, CDF::IV);
    } else if (adf_.size() > 0) {
      return 0.5 * (adf_xp(g) + adf_yn(g));
    } else {
      return 1.;
    }
  }

  DiffusionData& rotate_clockwise();
  DiffusionData& rotate_counterclockwise();
  DiffusionData& reflect_across_x_axis();
  DiffusionData& reflect_across_y_axis();

  const std::string& xs_name() const { return xs_->name(); }

  std::shared_ptr<DiffusionCrossSection> xs() const { return xs_; }

  const std::string& name() const { return name_; }
  void set_name(const std::string& new_name) { name_ = new_name; }

  const std::optional<LeakageCorrections>& leakage_corrections() const {
    return leakage_corrections_;
  }
  void set_leakage_corrections(const std::optional<LeakageCorrections>& lc);

  bool reflector() const { return reflector_; }
  void set_reflector(bool r) { reflector_ = r; }

  const xt::xtensor<double, 2>& adf() const { return adf_; }
  void set_adf(const xt::xtensor<double, 2>& adf);

  const xt::xtensor<double, 2>& cdf() const { return cdf_; }
  void set_cdf(const xt::xtensor<double, 2>& cdf);

  void save(const std::string& fname) const;
  static std::shared_ptr<DiffusionData> load(const std::string& fname);

  py::tuple to_tuple() const {
    return py::make_tuple(xs_, adf_, cdf_, name_, leakage_corrections_,
                          reflector_);
  }

 private:
  std::shared_ptr<DiffusionCrossSection> xs_;
  xt::xtensor<double, 2> adf_;  // group then ADF direction
  xt::xtensor<double, 2> cdf_;  // group then CDF direction
  std::string name_;
  std::optional<LeakageCorrections> leakage_corrections_;
  bool reflector_{false};

  friend class cereal::access;
  DiffusionData() {}
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(xs_), CEREAL_NVP(adf_), CEREAL_NVP(cdf_), CEREAL_NVP(name_),
        CEREAL_NVP(leakage_corrections_), CEREAL_NVP(reflector_));
  }
};

}  // namespace scarabee

#endif
