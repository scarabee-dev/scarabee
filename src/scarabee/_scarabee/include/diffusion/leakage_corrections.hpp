#ifndef SCARABEE_LEAKAGE_CORRECTIONS_H
#define SCARABEE_LEAKAGE_CORRECTIONS_H

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <vector>

namespace scarabee {

class LeakageCorrections {
 public:
  LeakageCorrections() = default;  // Only here for serialization !

  LeakageCorrections(std::size_t ngroups)
      : ngroups_(ngroups),
        data_(4 * ngroups_ + (ngroups_ * (ngroups_ - 1) / 2)) {}

  LeakageCorrections(py::tuple t)
      : ngroups_(t[0].cast<std::size_t>()),
        data_(t[1].cast<std::vector<double>>()) {}

  std::size_t ngroups() const { return ngroups_; }

  double D(std::size_t g) const;
  void set_D(std::size_t g, double val);

  double Ea(std::size_t g) const;
  void set_Ea(std::size_t g, double val);

  double Ef(std::size_t g) const;
  void set_Ef(std::size_t g, double val);

  double vEf(std::size_t g) const;
  void set_vEf(std::size_t g, double val);

  double Es(std::size_t g_in, std::size_t g_out) const;
  void set_Es(std::size_t g_in, std::size_t g_out, double val);

  py::tuple to_tuple() const { return py::make_tuple(ngroups_, data_); }

 private:
  std::size_t ngroups_;
  std::vector<double> data_;

  // Inside data_, first 4*ngroups_ entries store Coeff for D, Ea, Ef, vEf for
  // each group. The next entries store all down scattering coefficients.

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(ngroups_), CEREAL_NVP(data_));
  }
};

}  // namespace scarabee

#endif
