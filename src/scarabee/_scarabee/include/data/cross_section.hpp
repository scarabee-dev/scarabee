#ifndef SCARABEE_CROSS_SECTIONS_H
#define SCARABEE_CROSS_SECTIONS_H

#include <data/xs1d.hpp>
#include <data/xs2d.hpp>
#include <data/diffusion_cross_section.hpp>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>

#include <cstdint>
#include <string>
#include <memory>
#include <utility>

namespace scarabee {

class CrossSection {
 public:
  CrossSection(const xt::xtensor<double, 1>& Etr,
               const xt::xtensor<double, 1>& Ea,
               const xt::xtensor<double, 2>& Es_tr,
               const xt::xtensor<double, 1>& Ef,
               const xt::xtensor<double, 1>& vEf,
               const xt::xtensor<double, 1>& chi, const std::string& name = "");

  CrossSection(const XS1D& Etr, const XS1D& Ea, const XS2D& Es_tr,
               const XS1D& Ef, const XS1D& vEf, const XS1D& chi,
               const std::string& name = "");

  CrossSection(const xt::xtensor<double, 1>& Et,
               const xt::xtensor<double, 1>& Dtr,
               const xt::xtensor<double, 1>& Ea,
               const xt::xtensor<double, 3>& Es,
               const xt::xtensor<double, 1>& Ef,
               const xt::xtensor<double, 1>& vEf,
               const xt::xtensor<double, 1>& chi, const std::string& name = "");

  CrossSection(const XS1D& Et, const XS1D& Dtr, const XS1D& Ea, const XS2D& Es,
               const XS1D& Ef, const XS1D& vEf, const XS1D& chi,
               const std::string& name = "");

  CrossSection(const xt::xtensor<double, 1>& Etr,
               const xt::xtensor<double, 1>& Ea,
               const xt::xtensor<double, 2>& Es_tr,
               const std::string& name = "");

  CrossSection(const XS1D& Etr, const XS1D& Ea, const XS2D& Es_tr,
               const std::string& name = "");

  CrossSection(const xt::xtensor<double, 1>& Et,
               const xt::xtensor<double, 1>& Dtr,
               const xt::xtensor<double, 1>& Ea,
               const xt::xtensor<double, 3>& Es, const std::string& name = "");

  CrossSection(const XS1D& Et, const XS1D& Dtr, const XS1D& Ea, const XS2D& Es,
               const std::string& name = "");

  void set(const CrossSection& other) { (*this) = other; }

  std::size_t ngroups() const { return Etr_.ngroups(); }

  const std::string& name() const { return name_; }
  void set_name(const std::string& new_name) { name_ = new_name; }

  bool fissile() const { return fissile_; }

  bool anisotropic() const { return Es_.anisotropic(); }

  std::size_t max_legendre_order() const { return Es_.max_legendre_order(); }

  double Etr(std::size_t g) const { return Etr_(g); }

  double Dtr(std::size_t g) const { return Dtr_(g); }

  double Et(std::size_t g) const { return Etr_(g) + Dtr_(g); }

  double Ea(std::size_t g) const { return Ea_(g); }

  double Ef(std::size_t g) const { return Ef_(g); }

  double vEf(std::size_t g) const { return vEf_(g); }

  double nu(std::size_t g) const {
    const double Efg = Ef_(g);
    if (Efg > 0.) {
      return vEf_(g) / Efg;
    }
    return 0.;
  }

  double Er(std::size_t g) const { return Ea(g) + Es_tr(g) - Es_tr(g, g); }

  double chi(std::size_t g) const { return chi_(g); }

  double Es_tr(std::size_t gin, std::size_t gout) const {
    return Es_(0, gin, gout);
  }

  double Es_tr(std::size_t gin) const { return Es_(0, gin); }

  double Es(std::size_t l, std::size_t gin, std::size_t gout) const {
    if (l > this->max_legendre_order()) return 0.;

    double Es_l_gin_gout = Es_(l, gin, gout);
    if (l == 0 && gin == gout) {
      Es_l_gin_gout += Dtr_(gin);
    }
    return Es_l_gin_gout;
  }

  double Es(std::size_t l, std::size_t gin) const {
    if (l > this->max_legendre_order()) return 0.;

    double Es = Es_(l, gin);
    if (l == 0) {
      Es += Dtr_(gin);
    }
    return Es;
  }

  std::shared_ptr<CrossSection> condense(
      const std::vector<std::pair<std::size_t, std::size_t>>& groups,
      const xt::xtensor<double, 1>& flux) const;

  std::shared_ptr<DiffusionCrossSection> diffusion_xs() const;

  const XS1D& Etr_XS1D() const { return Etr_; }
  const XS1D& Dtr_XS1D() const { return Dtr_; }
  const XS1D& Ea_XS1D() const { return Ea_; }
  const XS1D& Ef_XS1D() const { return Ef_; }
  const XS1D& vEf_XS1D() const { return vEf_; }
  const XS1D& chi_XS1D() const { return chi_; }
  const XS2D& Es_XS2D() const { return Es_; }

  // Operators for constructing compound cross sections
  CrossSection operator+(const CrossSection& R) const;
  CrossSection operator*(double N) const;
  CrossSection& operator+=(const CrossSection& R);
  CrossSection& operator*=(double N);

  void save(const std::string& fname) const;
  static std::shared_ptr<CrossSection> load(const std::string& fname);

 private:
  XS1D Etr_;  // Transport xs
  XS1D Dtr_;  // Transport Correction xs
  XS1D Ea_;   // Absorption xs
  XS1D Ef_;   // Fission xs
  XS1D vEf_;  // Fission xs * yield
  XS1D chi_;  // Fission spectrum
  // Scattering Matrices. [0,:,:] is the TRANSPORT corrected P0 scatter matrix
  // while [1,:,:] is the P1 scatter matrix, [2,:,:] is the P2 matrix, etc.
  XS2D Es_;
  std::string name_;
  bool fissile_;

  void check_xs();

  friend class cereal::access;

  CrossSection() {}

  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(Etr_), CEREAL_NVP(Dtr_), CEREAL_NVP(Ea_), CEREAL_NVP(Ef_),
        CEREAL_NVP(vEf_), CEREAL_NVP(chi_), CEREAL_NVP(Es_), CEREAL_NVP(name_),
        CEREAL_NVP(fissile_));
  }
};

}  // namespace scarabee

#endif
