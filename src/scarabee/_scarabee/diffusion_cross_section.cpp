#include <data/diffusion_cross_section.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>
#include <utils/constants.hpp>

#include <cmath>
#include <sstream>

namespace scarabee {

DiffusionCrossSection::DiffusionCrossSection(const xt::xtensor<double, 1>& D,
                                             const xt::xtensor<double, 1>& Ea,
                                             const xt::xtensor<double, 2>& Es,
                                             const xt::xtensor<double, 1>& Ef,
                                             const xt::xtensor<double, 1>& vEf,
                                             const xt::xtensor<double, 1>& chi,
                                             const std::string& name)
    : Es_(Es),
      D_(D),
      Ea_(Ea),
      Ef_(Ef),
      vEf_(vEf),
      chi_(chi),
      name_(name),
      fissile_(false) {
  this->check_xs();
}

DiffusionCrossSection::DiffusionCrossSection(const xt::xtensor<double, 1>& D,
                                             const xt::xtensor<double, 1>& Ea,
                                             const xt::xtensor<double, 2>& Es,
                                             const std::string& name)
    : Es_(Es), D_(D), Ea_(Ea), vEf_(), chi_(), name_(name), fissile_(false) {
  Ef_ = xt::zeros<double>({ngroups()});
  vEf_ = xt::zeros<double>({ngroups()});
  chi_ = xt::zeros<double>({ngroups()});
  this->check_xs();
}

void DiffusionCrossSection::check_xs() {
  // Make correct number of groups
  if (ngroups() == 0) {
    auto mssg = "Must have at least 1 energy group.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (Ea_.size() != ngroups()) {
    auto mssg = "Ea is not the same size of Et.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (Ef_.size() != ngroups()) {
    auto mssg = "Ef is not the same size of Et.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (vEf_.size() != ngroups()) {
    auto mssg = "vEf is not the same size of Et.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (chi_.size() != ngroups()) {
    auto mssg = "chi is not the same size of Et.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (Es_.shape(0) != ngroups() || Es_.shape(1) != ngroups()) {
    auto mssg = "Es is not the same size of Et.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Check values for each group
  double chi_sum = 0.;
  for (std::size_t g = 0; g < ngroups(); g++) {
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

    if (std::isnan(D_(gin))) {
      std::stringstream mssg;
      mssg << "D has NaN value in group " << gin << ".";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    if (D_(gin) == 0.) {
      std::stringstream mssg;
      mssg << "D value of zero in group " << gin << ".";
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

    for (std::size_t gout = 0; gout < ngroups(); gout++) {
      if (std::isnan(Es_(gin, gout))) {
        std::stringstream mssg;
        mssg << "Es has NaN value in transfer " << gin << " -> " << gout << ".";
        spdlog::error(mssg.str());
        throw ScarabeeException(mssg.str());
      }
    }
  }
}

std::shared_ptr<DiffusionCrossSection> DiffusionCrossSection::condense(
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

  xt::xtensor<double, 1> D = xt::zeros<double>({NGOUT});
  xt::xtensor<double, 1> Ea = xt::zeros<double>({NGOUT});
  xt::xtensor<double, 2> Es = xt::zeros<double>({NGOUT, NGOUT});
  xt::xtensor<double, 1> Ef = xt::zeros<double>({NGOUT});
  xt::xtensor<double, 1> vEf = xt::zeros<double>({NGOUT});
  xt::xtensor<double, 1> chi = xt::zeros<double>({NGOUT});

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
      D(G) += fluxg_fluxG * this->D(g);
      Ea(G) += fluxg_fluxG * this->Ea(g);
      Ef(G) += fluxg_fluxG * this->Ef(g);
      vEf(G) += fluxg_fluxG * this->vEf(g);
      chi(G) += this->chi(g);  // chi doesn't need to be weighted
    }

    // Next, we do all 2D cross sections
    for (std::size_t GG = 0; GG < NGOUT; GG++) {  // Outgoing macro groups
      const std::size_t gg_min = groups[GG].first;
      const std::size_t gg_max = groups[GG].second;

      for (std::size_t g = g_min; g <= g_max; g++) {  // Incoming micro groups
        const double fluxg_fluxG = flux(g) * invs_flux_G;
        for (std::size_t gg = gg_min; gg <= gg_max;
             gg++) {  // Outgoing micro groups
          Es(G, GG) += fluxg_fluxG * this->Es(g, gg);
        }
      }
    }
  }

  return std::make_shared<DiffusionCrossSection>(D, Ea, Es, Ef, vEf, chi,
                                                 this->name_);
}

}  // namespace scarabee
