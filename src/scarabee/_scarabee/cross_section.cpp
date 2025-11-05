#include <data/cross_section.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>
#include <utils/constants.hpp>

#include <xtensor/generators/xbuilder.hpp>

#include <cereal/archives/portable_binary.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace scarabee {

CrossSection::CrossSection(const xt::xtensor<double, 1>& Etr,
                           const xt::xtensor<double, 1>& Ea,
                           const xt::xtensor<double, 2>& Es_tr,
                           const xt::xtensor<double, 1>& Ef,
                           const xt::xtensor<double, 1>& vEf,
                           const xt::xtensor<double, 1>& chi,
                           const std::string& name)
    : Etr_(Etr),
      Dtr_(xt::zeros<double>({Etr_.ngroups()})),
      Ea_(Ea),
      Ef_(Ef),
      vEf_(vEf),
      chi_(chi),
      Es_(Es_tr),
      name_(name),
      fissile_(false) {
  // We were provided with transport corrected data, so Dtr is 0 here
  this->check_xs();
}

CrossSection::CrossSection(const XS1D& Etr, const XS1D& Ea, const XS2D& Es_tr,
                           const XS1D& Ef, const XS1D& vEf, const XS1D& chi,
                           const std::string& name)
    : Etr_(Etr),
      Dtr_(xt::zeros<double>({Etr_.ngroups()})),
      Ea_(Ea),
      Ef_(Ef),
      vEf_(vEf),
      chi_(chi),
      Es_(Es_tr),
      name_(name),
      fissile_(false) {
  // We were provided with transport corrected data, so Dtr is 0 here
  this->check_xs();
}

CrossSection::CrossSection(
    const xt::xtensor<double, 1>& Et, const xt::xtensor<double, 1>& Dtr,
    const xt::xtensor<double, 1>& Ea, const xt::xtensor<double, 3>& Es,
    const xt::xtensor<double, 1>& Ef, const xt::xtensor<double, 1>& vEf,
    const xt::xtensor<double, 1>& chi, const std::string& name)
    : Etr_(Et),
      Dtr_(Dtr),
      Ea_(Ea),
      Ef_(Ef),
      vEf_(vEf),
      chi_(chi),
      Es_(Es),
      name_(name),
      fissile_(false) {
  // Check xs first, to make sure the size of everything is okay
  this->check_xs();

  // Apply the transport correction
  for (std::size_t g = 0; g < ngroups(); g++) {
    Etr_.set_value(g, Etr_(g) - Dtr_(g));
    Es_.set_value(0, g, g, Es_(0, g, g) - Dtr_(g));
  }
}

CrossSection::CrossSection(const XS1D& Et, const XS1D& Dtr, const XS1D& Ea,
                           const XS2D& Es, const XS1D& Ef, const XS1D& vEf,
                           const XS1D& chi, const std::string& name)
    : Etr_(Et),
      Dtr_(Dtr),
      Ea_(Ea),
      Ef_(Ef),
      vEf_(vEf),
      chi_(chi),
      Es_(Es),
      name_(name),
      fissile_(false) {
  // Check xs first, to make sure the size of everything is okay
  this->check_xs();

  // Apply the transport correction
  for (std::size_t g = 0; g < ngroups(); g++) {
    Etr_.set_value(g, Etr_(g) - Dtr_(g));
    Es_.set_value(0, g, g, Es_(0, g, g) - Dtr_(g));
  }
}

CrossSection::CrossSection(const xt::xtensor<double, 1>& Etr,
                           const xt::xtensor<double, 1>& Ea,
                           const xt::xtensor<double, 2>& Es_tr,
                           const std::string& name)
    : Etr_(Etr),
      Dtr_(xt::zeros<double>({Etr_.ngroups()})),
      Ea_(Ea),
      Ef_(xt::zeros<double>({Etr_.ngroups()})),
      vEf_(xt::zeros<double>({Etr_.ngroups()})),
      chi_(xt::zeros<double>({Etr_.ngroups()})),
      Es_(Es_tr),
      name_(name),
      fissile_(false) {
  this->check_xs();
}

CrossSection::CrossSection(const XS1D& Etr, const XS1D& Ea, const XS2D& Es_tr,
                           const std::string& name)
    : Etr_(Etr),
      Dtr_(xt::zeros<double>({Etr_.ngroups()})),
      Ea_(Ea),
      Ef_(xt::zeros<double>({Etr_.ngroups()})),
      vEf_(xt::zeros<double>({Etr_.ngroups()})),
      chi_(xt::zeros<double>({Etr_.ngroups()})),
      Es_(Es_tr),
      name_(name),
      fissile_(false) {
  this->check_xs();
}

CrossSection::CrossSection(const xt::xtensor<double, 1>& Et,
                           const xt::xtensor<double, 1>& Dtr,
                           const xt::xtensor<double, 1>& Ea,
                           const xt::xtensor<double, 3>& Es,
                           const std::string& name)
    : Etr_(Et),
      Dtr_(Dtr),
      Ea_(Ea),
      Ef_(xt::zeros<double>({Etr_.ngroups()})),
      vEf_(xt::zeros<double>({Etr_.ngroups()})),
      chi_(xt::zeros<double>({Etr_.ngroups()})),
      Es_(Es),
      name_(name),
      fissile_(false) {
  // Apply the transport correction
  for (std::size_t g = 0; g < ngroups(); g++) {
    Etr_.set_value(g, Etr_(g) - Dtr_(g));
    Es_.set_value(0, g, g, Es_(0, g, g) - Dtr_(g));
  }

  this->check_xs();
}

CrossSection::CrossSection(const XS1D& Et, const XS1D& Dtr, const XS1D& Ea,
                           const XS2D& Es, const std::string& name)
    : Etr_(Et),
      Dtr_(Dtr),
      Ea_(Ea),
      Ef_(xt::zeros<double>({Etr_.ngroups()})),
      vEf_(xt::zeros<double>({Etr_.ngroups()})),
      chi_(xt::zeros<double>({Etr_.ngroups()})),
      Es_(Es),
      name_(name),
      fissile_(false) {
  // Apply the transport correction
  for (std::size_t g = 0; g < ngroups(); g++) {
    Etr_.set_value(g, Etr_(g) - Dtr_(g));
    Es_.set_value(0, g, g, Es_(0, g, g) - Dtr_(g));
  }

  this->check_xs();
}

std::shared_ptr<CrossSection> CrossSection::condense(
    const std::vector<std::pair<std::size_t, std::size_t>>& groups,
    const xt::xtensor<double, 1>& flux) const {
  const std::size_t NGOUT = groups.size();
  const std::size_t NG = ngroups();

  // First, we need to check the condensation scheme
  if (groups.size() == 0) {
    auto mssg = "Empty energy condensation scheme provided.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (groups.front().first != 0) {
    auto mssg = "The energy condensation scheme does not start with 0.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (groups.back().second != NG - 1) {
    std::stringstream mssg;
    mssg << "The energy condensation scheme does not end with " << NG - 1
         << ".";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  for (std::size_t i = 0; i < NGOUT - 1; i++) {
    if (groups[i].second + 1 != groups[i + 1].first) {
      std::stringstream mssg;
      mssg << "Condensed groups " << i << " and " << i + 1
           << " are not continuous.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
  }

  if (flux.size() != NG) {
    auto mssg =
        "The number of provided flux values diagrees with the number of energy "
        "groups.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Everything checks out, we can start the condensation.

  xt::xtensor<double, 1> Et = xt::zeros<double>({NGOUT});
  xt::xtensor<double, 1> Dtr = xt::zeros<double>({NGOUT});
  xt::xtensor<double, 1> Ea = xt::zeros<double>({NGOUT});
  xt::xtensor<double, 1> Ef = xt::zeros<double>({NGOUT});
  xt::xtensor<double, 1> vEf = xt::zeros<double>({NGOUT});
  xt::xtensor<double, 1> chi = xt::zeros<double>({NGOUT});
  xt::xtensor<double, 3> Es =
      xt::zeros<double>({max_legendre_order() + 1, NGOUT, NGOUT});

  for (std::size_t G = 0; G < NGOUT; G++) {  // Incoming macro groups
    const std::size_t g_min = groups[G].first;
    const std::size_t g_max = groups[G].second;

    // First, we get the sum of all flux values in the macro group
    double flux_G = 0.;
    for (std::size_t g = g_min; g <= g_max; g++) flux_G += flux(g);
    const double invs_flux_G = 1. / flux_G;

    // First we do all of the 1D cross sections
    for (std::size_t g = g_min; g <= g_max; g++) {
      const double fluxg_fluxG = flux(g) * invs_flux_G;
      Et(G) += fluxg_fluxG * this->Et(g);
      Dtr(G) += fluxg_fluxG * this->Dtr(g);
      Ea(G) += fluxg_fluxG * this->Ea(g);
      Ef(G) += fluxg_fluxG * this->Ef(g);
      vEf(G) += fluxg_fluxG * this->vEf(g);
      chi(G) += this->chi(g);  // chi doesn't need to be weighted
    }

    // Next, we do all 2D cross sections
    for (std::size_t l = 0; l <= max_legendre_order(); l++) {
      for (std::size_t GG = 0; GG < NGOUT; GG++) {  // Outgoing macro groups
        const std::size_t gg_min = groups[GG].first;
        const std::size_t gg_max = groups[GG].second;

        for (std::size_t g = g_min; g <= g_max; g++) {  // Incoming micro groups
          const double fluxg_fluxG = flux(g) * invs_flux_G;
          for (std::size_t gg = gg_min; gg <= gg_max;
               gg++) {  // Outgoing micro groups
            Es(l, G, GG) += fluxg_fluxG * this->Es(l, g, gg);
          }
        }
      }
    }
  }

  // If we don't have a P1 matrix, then Et actually contains Etr and Es
  // contains Es_tr. We can therefore use that constructor.
  return std::make_shared<CrossSection>(Et, Dtr, Ea, Es, Ef, vEf, chi,
                                        this->name_);
}

CrossSection& CrossSection::operator+=(const CrossSection& R) {
  // Perform dimensionality checks
  if (ngroups() != R.ngroups()) {
    auto mssg = "Disagreement in number of groups.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // First, calculate/update fission spectrum
  double L_vEf_sum = 0.;
  double R_vEf_sum = 0.;
  for (std::size_t g = 0; g < ngroups(); g++) {
    L_vEf_sum += vEf(g);
    R_vEf_sum += R.vEf(g);
  }

  // Resize chi if needed
  if (this->chi_.ngroups() < R.chi_.ngroups()) {
    this->chi_.resize(R.chi_.ngroups());
  }

  double chi_sum = 0.;
  if (L_vEf_sum + R_vEf_sum > 0.) {
    for (std::size_t g = 0; g < ngroups(); g++) {
      const double new_chi =
          (chi(g) * L_vEf_sum + R.chi(g) * R_vEf_sum) / (L_vEf_sum + R_vEf_sum);
      chi_.set_value(g, new_chi);
      chi_sum += chi_(g);
    }

    // Renormalize chi
    if (chi_sum > 0.) chi_ /= chi_sum;
  } else {
    // With no entries, all groups have chi(g) = 0
    for (std::size_t g = 0; g < chi_.ngroups(); g++) chi_.set_value(g, 0.);
  }

  // Make sure that we are fissile if the other was also fissile
  if (R.fissile()) fissile_ = true;

  // Add all other cross sections which aren't averaged
  Etr_ += R.Etr_;
  Dtr_ += R.Dtr_;
  Ea_ += R.Ea_;
  Ef_ += R.Ef_;
  vEf_ += R.vEf_;
  Es_ += R.Es_;

  this->check_xs();

  return *this;
}

std::shared_ptr<DiffusionCrossSection> CrossSection::diffusion_xs() const {
  const std::size_t NG = Etr_.ngroups();
  xt::xtensor<double, 1> D = xt::zeros<double>({NG});
  xt::xtensor<double, 1> Ea = xt::zeros<double>({NG});
  xt::xtensor<double, 1> Ef = xt::zeros<double>({NG});
  xt::xtensor<double, 1> vEf = xt::zeros<double>({NG});
  xt::xtensor<double, 1> chi = xt::zeros<double>({NG});
  xt::xtensor<double, 2> Es = xt::zeros<double>({NG, NG});

  for (std::size_t g = 0; g < NG; g++) {
    D(g) = 1. / (3. * Etr_(g));
    Ea(g) = Ea_(g);
    Ef(g) = Ef_(g);
    vEf(g) = vEf_(g);
    chi(g) = chi_(g);

    for (std::size_t gg = 0; gg < NG; gg++) {
      Es(g, gg) = Es_(0, g, gg);
    }
  }

  return std::make_shared<DiffusionCrossSection>(D, Ea, Es, Ef, vEf, chi,
                                                 this->name_);
}

CrossSection& CrossSection::operator*=(double N) {
  // Scale Es_
  Es_ *= N;

  // Scale Etr_
  Etr_ *= N;

  // Scale Dtr_
  Dtr_ *= N;

  // Scale Ea_
  Ea_ *= N;

  // Scale Ef_
  Ef_ *= N;

  // Scale vEf_
  vEf_ *= N;

  this->check_xs();

  return *this;
}

CrossSection CrossSection::operator+(const CrossSection& R) const {
  CrossSection out = *this;
  out += R;
  return out;
}

CrossSection CrossSection::operator*(double N) const {
  CrossSection out = *this;
  out *= N;
  return out;
}

void CrossSection::check_xs() {
  // Make correct number of groups
  if (ngroups() == 0) {
    auto mssg = "Must have at least 1 energy group.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (Dtr_.ngroups() != ngroups()) {
    auto mssg = "Dtr is not the same size as Et.";
    spdlog::error(mssg);
    spdlog::error("Dtr_.ngroups() = {:}", Dtr_.ngroups());
    spdlog::error("ngroups() = {:}", ngroups());
    throw ScarabeeException(mssg);
  }

  if (Ea_.ngroups() != ngroups()) {
    auto mssg = "Ea is not the same size as Et.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (Ef_.ngroups() != ngroups()) {
    auto mssg = "Ef is not the same size as Et.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (vEf_.ngroups() != ngroups()) {
    auto mssg = "vEf is not the same size as Et.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (chi_.ngroups() != ngroups()) {
    auto mssg = "chi is not the same size as Et.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (Es_.ngroups() != ngroups()) {
    auto mssg = "Es is not the same size as Et.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Check values for each group
  double chi_sum = 0.;
  for (std::size_t g = 0; g < ngroups(); g++) {
    if (std::abs((Etr(g) - Ea(g) - Es_tr(g)) / Etr(g)) > 1.E-3) {
      std::stringstream mssg;
      mssg << "Ea + Es_tr != Etr in group " << g << ".";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    if (vEf(g) < 0.) {
      std::stringstream mssg;
      mssg << "vEf is negative in group " << g << ".";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    if (vEf(g) > 0.) fissile_ = true;

    if (chi(g) < 0.) {
      std::stringstream mssg;
      mssg << "chi is negative in group " << g << ".";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
    chi_sum += chi(g);
  }

  if (fissile_ && chi_sum <= 0.) {
    std::stringstream mssg;
    mssg << "Material is fissile, but all chi in all groups is zero.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // Make sure no NaN values
  for (std::size_t gin = 0; gin < ngroups(); gin++) {
    if (std::isnan(Dtr_(gin))) {
      std::stringstream mssg;
      mssg << "Dtr has NaN value in group " << gin << ".";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    if (std::isnan(Ea_(gin))) {
      std::stringstream mssg;
      mssg << "Ea has NaN value in group " << gin << ".";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    if (std::isnan(Ef_(gin))) {
      std::stringstream mssg;
      mssg << "Ef has NaN value in group " << gin << ".";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    if (std::isnan(Etr_(gin))) {
      std::stringstream mssg;
      mssg << "Etr has NaN value in group " << gin << ".";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    if (Etr_(gin) == 0.) {
      std::stringstream mssg;
      mssg << "Etr value of zero in group " << gin << ".";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    if (std::isnan(vEf_(gin))) {
      std::stringstream mssg;
      mssg << "vEf has NaN value in group " << gin << ".";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    if (std::isnan(chi_(gin))) {
      std::stringstream mssg;
      mssg << "chi has NaN value in group " << gin << ".";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    for (std::size_t l = 0; l <= max_legendre_order(); l++) {
      for (std::size_t gout = 0; gout < ngroups(); gout++) {
        if (std::isnan(Es_(l, gin, gout))) {
          std::stringstream mssg;
          mssg << "Es_ has NaN value in l = " << l << ", " << gin << " -> "
               << gout << ".";
          spdlog::error(mssg.str());
          throw ScarabeeException(mssg.str());
        }
      }
    }
  }
}

void CrossSection::save(const std::string& fname) const {
  if (std::filesystem::exists(fname)) {
    std::filesystem::remove(fname);
  }

  std::ofstream file(fname, std::ios_base::binary);

  cereal::PortableBinaryOutputArchive arc(file);

  arc(*this);
}

std::shared_ptr<CrossSection> CrossSection::load(const std::string& fname) {
  if (std::filesystem::exists(fname) == false) {
    std::stringstream mssg;
    mssg << "The file \"" << fname << "\" does not exist.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  std::shared_ptr<CrossSection> out(new CrossSection());

  std::ifstream file(fname, std::ios_base::binary);

  cereal::PortableBinaryInputArchive arc(file);

  arc(*out);

  return out;
}

}  // namespace scarabee
