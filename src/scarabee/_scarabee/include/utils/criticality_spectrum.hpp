#ifndef SCARABEE_CRITICALITY_SPECTRUM_H
#define SCARABEE_CRITICALITY_SPECTRUM_H

#include <data/cross_section.hpp>
#include <data/diffusion_cross_section.hpp>
#include <utils/serialization.hpp>

#include <xtensor/containers/xtensor.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <memory>

namespace scarabee {

class CriticalitySpectrum {
 public:
  // Need a virtual destructor for RTTI so pybind11 does correctly down casts
  virtual ~CriticalitySpectrum() = default;

  std::size_t ngroups() const { return flux_.size(); }

  double k_inf() const { return k_inf_; }

  double B2() const { return B2_; }
  double buckling() const { return B2(); }

  const xt::xtensor<double, 1>& flux() const { return flux_; }
  const xt::xtensor<double, 1>& diff_coeff() const { return diff_coeff_; }

  const std::shared_ptr<CrossSection>& cross_section() const { return xs_; }
  double flux(std::size_t g) const { return flux_(g); }
  double diff_coeff(std::size_t g) const { return diff_coeff_(g); }

  std::shared_ptr<DiffusionCrossSection> make_diffusion_cross_section() const;

 protected:
  xt::xtensor<double, 1> flux_;
  xt::xtensor<double, 1> diff_coeff_;
  std::shared_ptr<CrossSection> xs_;
  double k_inf_, B2_;

  CriticalitySpectrum() : flux_(), diff_coeff_(), xs_(), k_inf_(), B2_() {}

  CriticalitySpectrum(py::tuple t);
  py::tuple to_tuple() const;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(flux_), CEREAL_NVP(diff_coeff_), CEREAL_NVP(xs_),
        CEREAL_NVP(k_inf_), CEREAL_NVP(B2_));
  }
};

class FundamentalModeCriticalitySpectrum : public CriticalitySpectrum {
 public:
  FundamentalModeCriticalitySpectrum(std::shared_ptr<CrossSection> xs);
  FundamentalModeCriticalitySpectrum(std::shared_ptr<CrossSection> xs,
                                     double B2);
  FundamentalModeCriticalitySpectrum(py::tuple t);

  py::tuple to_tuple() const;

 private:
  friend class cereal::access;
  FundamentalModeCriticalitySpectrum() {}
  template <class Archive>
  void serialize(Archive& arc) {
    arc(cereal::base_class<CriticalitySpectrum>(this));
  }
};

class CriticalitySpectrumWithCurrent : public CriticalitySpectrum {
 public:
  const xt::xtensor<double, 1>& current() const { return current_; }
  double current(std::size_t g) const { return current_(g); }

 protected:
  xt::xtensor<double, 1> current_;

  CriticalitySpectrumWithCurrent() : current_() {}

  CriticalitySpectrumWithCurrent(py::tuple t);
  py::tuple to_tuple() const;

  void P1_B1_spectrum_search(bool B1);
  void P1_B1_provided_buckling(bool B1);

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(current_), cereal::base_class<CriticalitySpectrum>(this));
  }
};

class P1CriticalitySpectrum : public CriticalitySpectrumWithCurrent {
 public:
  P1CriticalitySpectrum(std::shared_ptr<CrossSection> xs);
  P1CriticalitySpectrum(std::shared_ptr<CrossSection> xs, double B2);
  P1CriticalitySpectrum(py::tuple t);

  py::tuple to_tuple() const;

 private:
  friend class cereal::access;
  P1CriticalitySpectrum() {}
  template <class Archive>
  void serialize(Archive& arc) {
    arc(cereal::base_class<CriticalitySpectrum>(this));
  }
};

class B1CriticalitySpectrum : public CriticalitySpectrumWithCurrent {
 public:
  B1CriticalitySpectrum(std::shared_ptr<CrossSection> xs);
  B1CriticalitySpectrum(std::shared_ptr<CrossSection> xs, double B2);
  B1CriticalitySpectrum(py::tuple t);

  py::tuple to_tuple() const;

 private:
  friend class cereal::access;
  B1CriticalitySpectrum() {}
  template <class Archive>
  void serialize(Archive& arc) {
    arc(cereal::base_class<CriticalitySpectrum>(this));
  }
};

}  // namespace scarabee

#endif
