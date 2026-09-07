#ifndef SCARABEE_LEAKAGE_CORRECTIONS_H
#define SCARABEE_LEAKAGE_CORRECTIONS_H

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>

#include <vector>

struct LeakageCorrectionsPickler;

namespace scarabee {

class LeakageCorrections {
 public:
  LeakageCorrections() = default;  // Only here for serialization !

  LeakageCorrections(std::size_t ngroups)
      : ngroups_(ngroups),
        data_(4 * ngroups_ + (ngroups_ * (ngroups_ - 1) / 2)) {}

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

 private:
  std::size_t ngroups_;
  std::vector<double> data_;

  // Inside data_, first 4*ngroups_ entries store Coeff for D, Ea, Ef, vEf for
  // each group. The next entries store all down scattering coefficients.

  friend class cereal::access;
  friend struct ::LeakageCorrectionsPickler;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(ngroups_), CEREAL_NVP(data_));
  }
};

}  // namespace scarabee

#endif
