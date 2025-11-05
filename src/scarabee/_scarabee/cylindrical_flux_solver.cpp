#include <cylindrical_flux_solver.hpp>
#include <utils/scarabee_exception.hpp>
#include <utils/logging.hpp>
#include <utils/timer.hpp>

#include <xtensor/generators/xbuilder.hpp>

namespace scarabee {

CylindricalFluxSolver::CylindricalFluxSolver(
    std::shared_ptr<CylindricalCell> cell)
    : flux_(),
      extern_source_(),
      j_ext_(),
      x_(),
      cell_(cell),
      k_(1.),
      a_(1.),
      k_tol_(1.E-5),
      flux_tol_(1.E-5),
      mode_(SimulationMode::Keff),
      solved_(false) {
  if (cell_ == nullptr) {
    auto mssg = "Provided CylindricalCell was a nullptr.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Initialize flux with 1's everywhere
  flux_.resize({ngroups(), nregions()});
  flux_.fill(1.);
  j_ext_.resize({ngroups()});
  j_ext_.fill(0.);
  x_.resize({ngroups()});
  x_.fill(0.);

  extern_source_ = xt::zeros<double>({ngroups(), nregions()});
}

void CylindricalFluxSolver::set_albedo(double a) {
  if (a < 0. || a > 1.) {
    auto mssg = "Albedo must be in the interval [0,1].";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  a_ = a;
  solved_ = false;
}

void CylindricalFluxSolver::set_flux_tolerance(double ftol) {
  if (ftol <= 0.) {
    auto mssg = "Tolerance for flux must be in the interval (0., 0.1).";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (ftol >= 0.1) {
    auto mssg = "Tolerance for flux must be in the interval (0., 0.1).";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  flux_tol_ = ftol;
  solved_ = false;
}

void CylindricalFluxSolver::set_keff_tolerance(double ktol) {
  if (ktol <= 0.) {
    auto mssg = "Tolerance for keff must be in the interval (0., 0.1).";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (ktol >= 0.1) {
    auto mssg = "Tolerance for keff must be in the interval (0., 0.1).";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  k_tol_ = ktol;
  solved_ = false;
}

double CylindricalFluxSolver::Qscat(std::size_t g, std::size_t i,
                                    const xt::xtensor<double, 2>& flux) const {
  double Qout = 0.;
  const auto& xs = cell_->xs(i);

  for (std::size_t gg = 0; gg < ngroups(); gg++) {
    const double flux_gg_i = flux(gg, i);

    // Scattering into group g, excluding g -> g
    if (gg != g) Qout += xs->Es_tr(gg, g) * flux_gg_i;
  }

  return Qout;
}

double CylindricalFluxSolver::Qfiss(std::size_t g, std::size_t i,
                                    const xt::xtensor<double, 2>& flux) const {
  double Qout = 0.;
  const double inv_k = 1. / k_;
  const auto& xs = cell_->xs(i);
  const double chi_g = xs->chi(g);

  for (std::size_t gg = 0; gg < ngroups(); gg++) {
    Qout += xs->vEf(gg) * flux(gg, i);
  }

  Qout *= inv_k * chi_g;

  return Qout;
}

double CylindricalFluxSolver::calc_keff(
    const xt::xtensor<double, 2>& flux) const {
  double keff = 0.;
  for (std::size_t r = 0; r < nregions(); r++) {
    const double Vr = cell_->volume(r);
    const auto& xs = cell_->xs(r);
    for (std::size_t g = 0; g < ngroups(); g++) {
      keff += Vr * xs->vEf(g) * flux(g, r);
    }
  }

  return keff;
}

double CylindricalFluxSolver::calc_flux_rel_diff(
    const xt::xtensor<double, 2>& flux,
    const xt::xtensor<double, 2>& next_flux) const {
  double max_rel_diff = 0.;

  for (std::size_t g = 0; g < ngroups(); g++) {
    for (std::size_t r = 0; r < nregions(); r++) {
      const double rel_diff =
          std::abs(next_flux(g, r) - flux(g, r)) / flux(g, r);

      if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;
    }
  }

  return max_rel_diff;
}

void CylindricalFluxSolver::fill_fission_source(
    xt::xtensor<double, 2>& source, const xt::xtensor<double, 2>& flux) const {
  for (std::size_t g = 0; g < ngroups(); g++) {
    for (std::size_t r = 0; r < nregions(); r++) {
      source(g, r) = Qfiss(g, r, flux);
    }
  }
}

void CylindricalFluxSolver::fill_scatter_source(
    xt::xtensor<double, 2>& source, const xt::xtensor<double, 2>& flux) const {
  for (std::size_t g = 0; g < ngroups(); g++) {
    for (std::size_t r = 0; r < nregions(); r++) {
      source(g, r) = Qscat(g, r, flux);
    }
  }
}

void CylindricalFluxSolver::solve(bool parallel) {
  Timer sim_timer;
  sim_timer.start();

  if (parallel == false) {
    solve_single_thread();
  } else {
    solve_parallel();
  }

  sim_timer.stop();
  spdlog::info("");
  spdlog::info("Simulation Time: {:.5E} s", sim_timer.elapsed_time());
}

void CylindricalFluxSolver::solve_single_thread() {
  spdlog::info("Solving for keff.");
  spdlog::info("keff tolerance: {:.5E}", k_tol_);
  spdlog::info("Flux tolerance: {:.5E}", flux_tol_);

  // Make sure the cell is solved
  if (cell_->solved() == false) {
    auto mssg = "Cannot solve for flux if cell is not solved.";
    throw ScarabeeException(mssg);
  }

  // Create a new array to hold the source according to the current flux, and
  // another for the next generation flux.
  xt::xtensor<double, 2> scat_source(flux_.shape());
  xt::xtensor<double, 2> fiss_source(flux_.shape());
  xt::xtensor<double, 2> next_flux(flux_.shape());
  xt::xtensor<double, 2> old_outer_flux(flux_.shape());
  double old_keff = 100.;
  if (mode_ == SimulationMode::Keff) {
    k_ = calc_keff(flux_);
  } else {
    k_ = 1.;
    old_keff = k_;
  }

  std::size_t outer_iter = 0;

  // Outer Generations
  double max_outer_flux_diff = 100.;
  while (std::abs((old_keff - k_) / k_) > k_tol_ ||
         max_outer_flux_diff > flux_tol_) {
    outer_iter++;
    old_outer_flux = flux_;

    // At the begining of a generation, we calculate the fission source
    fill_fission_source(fiss_source, flux_);

    // Copy flux into next_flux, so that we can continually use next_flux
    // in the inner iterations for the scattering source.
    next_flux = flux_;

    double max_inner_flux_diff = 100.;
    std::size_t inner_iter = 0;
    // Inner Iterations
    while (max_inner_flux_diff > flux_tol_) {
      inner_iter++;

      // At the begining of an inner iteration, we calculate the fission source
      fill_scatter_source(scat_source, next_flux);

      // From the sources, we calculate the new flux values
      for (std::size_t g = 0; g < ngroups(); g++) {
        for (std::size_t r = 0; r < nregions(); r++) {
          const double Yr = cell_->Y(a_, g, r);

          double Xr = 0.;
          for (std::size_t k = 0; k < nregions(); k++) {
            Xr +=
                (fiss_source(g, k) + scat_source(g, k) + extern_source_(g, k)) *
                cell_->X(a_, g, r, k);
          }

          next_flux(g, r) = Xr + j_ext_[g] * Yr;
        }
      }

      // Calculate the max difference in the flux
      max_inner_flux_diff = calc_flux_rel_diff(flux_, next_flux);
      if (mode_ == SimulationMode::Keff) {
        spdlog::debug("Inner iteration {} flux diff {:.5E}", inner_iter,
                      max_inner_flux_diff);
      } else if (mode_ == SimulationMode::Keff) {
        spdlog::info("-------------------------------------");
        spdlog::info("Iteration {:>4d}     max flux difference: {:.5E}",
                     inner_iter, max_inner_flux_diff);
      }

      // Copy next_flux into flux for calculating next relative difference
      flux_ = next_flux;
    }  // End of Inner Iterations
    max_outer_flux_diff = calc_flux_rel_diff(old_outer_flux, flux_);

    if (mode_ == SimulationMode::Keff) {
      // Calculate keff
      old_keff = k_;
      k_ = calc_keff(next_flux);
      spdlog::info("-------------------------------------");
      spdlog::info("Iteration {:>4d}          keff: {:.5f}", outer_iter, k_);
      spdlog::info("     keff difference:     {:.5E}",
                   std::abs((old_keff - k_) / k_));
      spdlog::info("     max flux difference: {:.5E}", max_outer_flux_diff);
    }

    // Assign next_flux to be the flux
    std::swap(next_flux, flux_);
  }  // End of Outer Generations

  // Now that we have the solution, we need to get the number of source
  // neutrons reaching the boundary, x, from Stamm'ler and Abbate.
  // These are used when calculating the currents.
  for (std::size_t g = 0; g < ngroups(); g++) {
    for (std::size_t r = 0; r < nregions(); r++) {
      x_[g] +=
          (Qfiss(g, r, flux_) + Qscat(g, r, flux_) + extern_source_(g, r)) *
          cell_->x(g, r);
    }
  }

  solved_ = true;
}

void CylindricalFluxSolver::solve_parallel() {
  spdlog::info("Solving for keff.");
  spdlog::info("keff tolerance: {:.5E}", k_tol_);
  spdlog::info("Flux tolerance: {:.5E}", flux_tol_);

  // Make sure the cell is solved
  if (cell_->solved() == false) {
    auto mssg = "Cannot solve for flux if cell is not solved.";
    throw ScarabeeException(mssg);
  }

  // Create a new array to hold the source according to the current flux, and
  // another for the next generation flux.
  xt::xtensor<double, 2> scat_source(flux_.shape());
  xt::xtensor<double, 2> fiss_source(flux_.shape());
  xt::xtensor<double, 2> next_flux(flux_.shape());
  xt::xtensor<double, 2> old_outer_flux(flux_.shape());
  double old_keff = 100.;
  if (mode_ == SimulationMode::Keff) {
    k_ = calc_keff(flux_);
  } else {
    k_ = 1.;
    old_keff = k_;
  }

  std::size_t outer_iter = 0;

  // Outer Generations
  double max_outer_flux_diff = 100.;
  while (std::abs((old_keff - k_) / k_) > k_tol_ ||
         max_outer_flux_diff > flux_tol_) {
    outer_iter++;
    old_outer_flux = flux_;

    // At the begining of a generation, we calculate the fission source
    fill_fission_source(fiss_source, flux_);

    // Copy flux into next_flux, so that we can continually use next_flux
    // in the inner iterations for the scattering source.
    next_flux = flux_;

    double max_inner_flux_diff = 100.;
    std::size_t inner_iter = 0;
    // Inner Iterations
    while (max_inner_flux_diff > flux_tol_) {
      inner_iter++;

      // At the begining of an inner iteration, we calculate the fission source
      fill_scatter_source(scat_source, next_flux);

      // From the sources, we calculate the new flux values
#pragma omp parallel for
      for (int ig = 0; ig < static_cast<int>(ngroups()); ig++) {
        std::size_t g = static_cast<std::size_t>(ig);
        for (std::size_t r = 0; r < nregions(); r++) {
          const double Yr = cell_->Y(a_, g, r);

          double Xr = 0.;
          for (std::size_t k = 0; k < nregions(); k++) {
            Xr +=
                (fiss_source(g, k) + scat_source(g, k) + extern_source_(g, k)) *
                cell_->X(a_, g, r, k);
          }

          next_flux(g, r) = Xr + j_ext_[g] * Yr;
        }
      }

      // Calculate the max difference in the flux
      max_inner_flux_diff = calc_flux_rel_diff(flux_, next_flux);
      if (mode_ == SimulationMode::Keff) {
        spdlog::debug("Inner iteration {} flux diff {:.5E}", inner_iter,
                      max_inner_flux_diff);
      } else if (mode_ == SimulationMode::Keff) {
        spdlog::info("-------------------------------------");
        spdlog::info("Iteration {:>4d}     max flux difference: {:.5E}",
                     inner_iter, max_inner_flux_diff);
      }

      // Copy next_flux into flux for calculating next relative difference
      flux_ = next_flux;
    }  // End of Inner Iterations
    max_outer_flux_diff = calc_flux_rel_diff(old_outer_flux, flux_);

    if (mode_ == SimulationMode::Keff) {
      // Calculate keff
      old_keff = k_;
      k_ = calc_keff(next_flux);
      spdlog::info("-------------------------------------");
      spdlog::info("Iteration {:>4d}          keff: {:.5f}", outer_iter, k_);
      spdlog::info("     keff difference:     {:.5E}",
                   std::abs((old_keff - k_) / k_));
      spdlog::info("     max flux difference: {:.5E}", max_outer_flux_diff);
    }

    // Assign next_flux to be the flux
    std::swap(next_flux, flux_);
  }  // End of Outer Generations

  // Now that we have the solution, we need to get the number of source
  // neutrons reaching the boundary, x, from Stamm'ler and Abbate.
  // These are used when calculating the currents.
  for (std::size_t g = 0; g < ngroups(); g++) {
    for (std::size_t r = 0; r < nregions(); r++) {
      x_[g] +=
          (Qfiss(g, r, flux_) + Qscat(g, r, flux_) + extern_source_(g, r)) *
          cell_->x(g, r);
    }
  }

  solved_ = true;
}

std::shared_ptr<CrossSection> CylindricalFluxSolver::homogenize() const {
  const std::size_t NR = this->nregions();
  std::vector<std::size_t> regions(NR, 0);
  for (std::size_t i = 0; i < NR; i++) {
    regions[i] = i;
  }

  return this->homogenize(regions);
}

std::shared_ptr<CrossSection> CylindricalFluxSolver::homogenize(
    std::size_t i_max) const {
  if (i_max >= this->nregions()) {
    std::stringstream mssg;
    mssg << "Desired max region index " << i_max << " is invalid.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  const std::size_t NR = i_max;
  std::vector<std::size_t> regions(NR, 0);
  for (std::size_t i = 0; i <= NR; i++) {
    regions[i] = i;
  }

  return this->homogenize(regions);
}

std::shared_ptr<CrossSection> CylindricalFluxSolver::homogenize(
    const std::vector<std::size_t>& regions) const {
  // We can only perform a homogenization if we have a flux spectrum
  if (solved() == false) {
    auto mssg =
        "Cannot perform homogenization when problem has not been solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure we were actually provided with regions
  if (regions.empty()) {
    const auto mssg = "No regions were provided for homogenization.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Check all regions are valid
  if (regions.size() > this->nregions()) {
    auto mssg =
        "The number of provided regions is greater than the number of regions.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  for (const auto m : regions) {
    if (m >= this->nregions()) {
      auto mssg = "Invalid region index in homogenization list.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }
  }

  // We now begin homogenization
  const std::size_t NR = regions.size();
  const std::size_t NG = this->ngroups();

  std::size_t max_l = 0;
  for (const auto m : regions) {
    const auto m_max_l = this->xs(m)->max_legendre_order();
    if (m_max_l > max_l) {
      max_l = m_max_l;
    }
  }

  xt::xtensor<double, 1> Et = xt::zeros<double>({NG});
  xt::xtensor<double, 1> Dtr = xt::zeros<double>({NG});
  xt::xtensor<double, 1> Ea = xt::zeros<double>({NG});
  xt::xtensor<double, 3> Es = xt::zeros<double>({max_l + 1, NG, NG});
  xt::xtensor<double, 1> Ef = xt::zeros<double>({NG});
  xt::xtensor<double, 1> vEf = xt::zeros<double>({NG});
  xt::xtensor<double, 1> chi = xt::zeros<double>({NG});

  // We need to calculate the total fission production in each volume for
  // generating the homogenized fission spectrum.
  std::vector<double> fiss_prod(NR, 0.);
  std::size_t j = 0;
  for (const auto i : regions) {
    const auto& mat = this->xs(i);
    const double V = this->volume(i);
    for (std::size_t g = 0; g < NG; g++) {
      fiss_prod[j] += mat->vEf(g) * flux(i, g) * V;
    }
    j++;
  }
  const double sum_fiss_prod =
      std::accumulate(fiss_prod.begin(), fiss_prod.end(), 0.);
  const double invs_sum_fiss_prod = sum_fiss_prod > 0.
                                        ? 1. / sum_fiss_prod
                                        : 1.;  // Do this to avoid nan errors

  for (std::size_t g = 0; g < NG; g++) {
    // Get the sum of flux*volume for this group
    double sum_fluxV = 0.;
    for (const auto i : regions) {
      sum_fluxV += this->flux(i, g) * this->volume(i);
    }
    const double invs_sum_fluxV = 1. / sum_fluxV;

    j = 0;
    for (const auto i : regions) {
      const auto& mat = this->xs(i);
      const double V = volume(i);
      const double flx = flux(i, g);
      const double coeff = invs_sum_fluxV * flx * V;
      Et(g) += coeff * mat->Et(g);
      Dtr(g) += coeff * mat->Dtr(g);
      Ea(g) += coeff * mat->Ea(g);
      Ef(g) += coeff * mat->Ef(g);
      vEf(g) += coeff * mat->vEf(g);

      chi(g) += invs_sum_fiss_prod * fiss_prod[j] * mat->chi(g);

      for (std::size_t l = 0; l <= max_l; l++) {
        for (std::size_t gg = 0; gg < NG; gg++) {
          Es(l, g, gg) += coeff * mat->Es(l, g, gg);
        }
      }

      j++;
    }
  }

  return std::make_shared<CrossSection>(Et, Dtr, Ea, Es, Ef, vEf, chi);
}

xt::xtensor<double, 1> CylindricalFluxSolver::homogenize_flux_spectrum() const {
  const std::size_t NR = this->nregions();
  std::vector<std::size_t> regions(NR, 0);
  for (std::size_t i = 0; i < NR; i++) {
    regions[i] = i;
  }

  return this->homogenize_flux_spectrum(regions);
}

xt::xtensor<double, 1> CylindricalFluxSolver::homogenize_flux_spectrum(
    std::size_t i_max) const {
  if (i_max >= this->nregions()) {
    std::stringstream mssg;
    mssg << "Desired max region index " << i_max << " is invalid.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  const std::size_t NR = i_max;
  std::vector<std::size_t> regions(NR, 0);
  for (std::size_t i = 0; i <= NR; i++) {
    regions[i] = i;
  }

  return this->homogenize_flux_spectrum(regions);
}

xt::xtensor<double, 1> CylindricalFluxSolver::homogenize_flux_spectrum(
    const std::vector<std::size_t>& regions) const {
  // We can only perform a homogenization if we have a flux spectrum
  if (solved() == false) {
    auto mssg =
        "Cannot perform spectrum homogenization when problem has not been "
        "solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure we were actually provided with regions
  if (regions.empty()) {
    const auto mssg =
        "No regions were provided for homogenization of flux spectrum.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Check all regions are valid
  if (regions.size() > this->nregions()) {
    auto mssg =
        "The number of provided regions is greater than the number of regions.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  for (const auto m : regions) {
    if (m >= this->nregions()) {
      auto mssg = "Invalid region index in homogenization list.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }
  }

  const std::size_t NG = this->ngroups();

  // First, calculate the sum of the volumes
  double sum_V = 0.;
  for (const auto i : regions) {
    sum_V += this->volume(i);
  }
  const double invs_sum_V = 1. / sum_V;

  xt::xtensor<double, 1> spectrum = xt::zeros<double>({NG});
  for (std::size_t g = 0; g < NG; g++) {
    for (const auto i : regions) {
      spectrum(g) += invs_sum_V * this->volume(i) * this->flux(i, g);
    }
  }

  return spectrum;
}

}  // namespace scarabee
