#include <reflector_sn.hpp>
#include <utils/gauss_legendre.hpp>
#include <utils/math.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>
#include <utils/timer.hpp>

#include <xtensor/io/xio.hpp>

#include <sstream>

namespace scarabee {

ReflectorSN::ReflectorSN(const std::vector<std::shared_ptr<CrossSection>>& xs,
                         const xt::xtensor<double, 1>& dx,
                         std::uint32_t nangles, bool anisotropic)
    : xs_(xs), dx_(dx), anisotropic_(anisotropic) {
  if (xs_.size() != dx_.size()) {
    auto mssg = "Number of cross sections and regions do not agree.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (xs_.size() == 0) {
    auto mssg = "Number of regions must be > 0.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  for (std::size_t i = 0; i < dx_.size(); i++) {
    if (dx_[i] <= 0.) {
      std::stringstream mssg;
      mssg << "Region " << i << " is <= 0.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
  }

  ngroups_ = xs_.front()->ngroups();
  for (std::size_t i = 0; i < xs_.size(); i++) {
    if (xs_[i]->ngroups() != ngroups_) {
      auto mssg = "Not all regions have the same number of groups.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }
  }

  // get the max-legendre-order for given scattering moments
  if (anisotropic_) {
    max_L_ = 0;
    for (std::size_t i = 0; i < xs_.size(); i++) {
      const auto& mat = *xs_[i];
      const std::size_t l = mat.max_legendre_order();
      if (l > max_L_) max_L_ = l;
    }
  }

  // Must allocate with zeros in case someone calls the flux method
  flux_ = xt::zeros<double>({ngroups_, xs_.size(), max_L_ + 1});
  J_ = xt::zeros<double>({ngroups_, xs_.size() + 1});

  // Now we need to set the spans for the angular quadrature
  switch (nangles) {
    case 2:
      mu_ = std::span<const double>(gl_2_abscissa.begin(), gl_2_abscissa.end());
      wgt_ = std::span<const double>(gl_2_weights.begin(), gl_2_weights.end());
      break;
    case 4:
      mu_ = std::span<const double>(gl_4_abscissa.begin(), gl_4_abscissa.end());
      wgt_ = std::span<const double>(gl_4_weights.begin(), gl_4_weights.end());
      break;
    case 6:
      mu_ = std::span<const double>(gl_6_abscissa.begin(), gl_6_abscissa.end());
      wgt_ = std::span<const double>(gl_6_weights.begin(), gl_6_weights.end());
      break;
    case 8:
      mu_ = std::span<const double>(gl_8_abscissa.begin(), gl_8_abscissa.end());
      wgt_ = std::span<const double>(gl_8_weights.begin(), gl_8_weights.end());
      break;
    case 10:
      mu_ =
          std::span<const double>(gl_10_abscissa.begin(), gl_10_abscissa.end());
      wgt_ =
          std::span<const double>(gl_10_weights.begin(), gl_10_weights.end());
      break;
    case 12:
      mu_ =
          std::span<const double>(gl_12_abscissa.begin(), gl_12_abscissa.end());
      wgt_ =
          std::span<const double>(gl_12_weights.begin(), gl_12_weights.end());
      break;
    case 14:
      mu_ =
          std::span<const double>(gl_14_abscissa.begin(), gl_14_abscissa.end());
      wgt_ =
          std::span<const double>(gl_14_weights.begin(), gl_14_weights.end());
      break;
    case 16:
      mu_ =
          std::span<const double>(gl_16_abscissa.begin(), gl_16_abscissa.end());
      wgt_ =
          std::span<const double>(gl_16_weights.begin(), gl_16_weights.end());
      break;
    case 32:
      mu_ =
          std::span<const double>(gl_32_abscissa.begin(), gl_32_abscissa.end());
      wgt_ =
          std::span<const double>(gl_32_weights.begin(), gl_32_weights.end());
      break;
    case 64:
      mu_ =
          std::span<const double>(gl_64_abscissa.begin(), gl_64_abscissa.end());
      wgt_ =
          std::span<const double>(gl_64_weights.begin(), gl_64_weights.end());
      break;
    case 128:
      mu_ = std::span<const double>(gl_128_abscissa.begin(),
                                    gl_128_abscissa.end());
      wgt_ =
          std::span<const double>(gl_128_weights.begin(), gl_128_weights.end());
      break;
    default: {
      std::stringstream mssg;
      mssg << "Invalid nangles argument of " << nangles << " .\n";
      mssg << "Please use one of the following: 2, 4, 6, 8, 10, 12, 14, 16, "
              "32, 64, 128.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    } break;
  }
}

void ReflectorSN::set_flux_tolerance(double ftol) {
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
}

void ReflectorSN::set_keff_tolerance(double ktol) {
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

  keff_tol_ = ktol;
}

void ReflectorSN::solve() {
  Timer sim_timer;
  sim_timer.start();

  if (anisotropic() && max_legendre_order() > 0) {
    solve_aniso();
  } else {
    solve_iso();
  }

  sim_timer.stop();
  spdlog::info("");
  spdlog::info("Simulation Time: {:.5E} s", sim_timer.elapsed_time());
}

void ReflectorSN::solve_iso() {
  const std::size_t NG = xs_[0]->ngroups();
  const std::size_t NR = xs_.size();

  if (max_legendre_order() != 0) {
    const auto mssg =
        "Should not use solve_iso if the maximum legendre order is greater "
        "than 0.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  flux_ = xt::ones<double>({NG, NR, max_L_ + 1});
  xt::xtensor<double, 3> next_flux = xt::zeros<double>({NG, NR, max_L_ + 1});
  xt::xtensor<double, 3> old_flux = xt::zeros<double>({NG, NR, max_L_ + 1});
  xt::xtensor<double, 3> Q = xt::zeros<double>({NG, NR, max_L_ + 1});

  keff_ = 1.;

  // Initialize stabalization matrix (see [1])
  xt::xtensor<double, 2> D;
  D.resize({NG, NR});
  D.fill(0.);
  for (std::size_t i = 0; i < NR; i++) {
    const auto xs = this->xs(i);
    for (std::size_t g = 0; g < NG; g++) {
      const double Estr_g_g = xs->Es_tr(g, g);
      if (Estr_g_g < 0.) {
        D(g, i) = -Estr_g_g / xs->Etr(g);
      }
    }
  }

  // Array to hold the incident flux at the reflective boundary. We create
  // this here instead of in the sweep to avoid making memory allocations.
  xt::xtensor<double, 2> incident_angular_flux =
      xt::zeros<double>({NG, mu_.size()});

  // Outer Iterations
  double keff_diff = 100.;
  double flux_diff = 100.;
  std::size_t iteration = 0;
  Timer iteration_timer;
  while (keff_diff > keff_tol_ || flux_diff > flux_tol_) {
    iteration_timer.reset();
    iteration_timer.start();
    iteration++;

    old_flux = flux_;

    fill_source_iso(Q, flux_);

    // Check for negative source values at beginning of simulation
    bool set_neg_src_to_zero = false;
    if (iteration <= 20) {
      for (std::size_t i = 0; i < Q.size(); i++) {
        if (Q.flat(i) < 0.) {
          Q.flat(i) = 0.;
          set_neg_src_to_zero = true;
        }
      }
    }

    next_flux.fill(0.);
    J_.fill(0.);
    sweep_iso(next_flux, incident_angular_flux, Q);

    // Apply stabalization (see [1])
    for (std::size_t i = 0; i < D.size(); i++) {
      if (D.flat(i) != 0.) {
        next_flux.flat(i) += flux_.flat(i) * D.flat(i);
        next_flux.flat(i) /= (1. + D.flat(i));
      }
    }

    // Get difference
    flux_diff = xt::amax(xt::abs(next_flux - flux_) / next_flux)();

    // Make sure that the flux is positive everywhere !
    bool set_neg_flux_to_zero = false;
    for (std::size_t i = 0; i < next_flux.size(); i++) {
      if (next_flux.flat(i) < 0.) {
        next_flux.flat(i) = 0.;
        set_neg_flux_to_zero = true;
      }
    }

    flux_ = next_flux;

    const double old_keff = keff_;
    keff_ = calc_keff(old_flux, flux_, keff_);
    keff_diff = std::abs((old_keff - keff_) / keff_);

    iteration_timer.stop();
    spdlog::info("-------------------------------------");
    spdlog::info("Iteration {:>6d}        keff: {:.5f}", iteration, keff_);
    spdlog::info("     keff difference:     {:.5E}", keff_diff);
    spdlog::info("     max flux difference: {:.5E}", flux_diff);
    spdlog::info("     iteration time: {:.5E} s",
                 iteration_timer.elapsed_time());

    // Write warnings about negative flux and source
    if (set_neg_src_to_zero) {
      spdlog::info("Negative source values set to zero");
    }
    if (set_neg_flux_to_zero) {
      spdlog::info("Negative flux values set to zero");
    }
  }

  solved_ = true;
}

void ReflectorSN::sweep_iso(xt::xtensor<double, 3>& flux,
                            xt::xtensor<double, 2>& incident_angular_flux,
                            const xt::xtensor<double, 3>& Q) {
  const std::size_t NG = xs_.front()->ngroups();
  const int iNG = static_cast<int>(NG);

  incident_angular_flux.fill(0.);

#pragma omp parallel for
  for (int ig = 0; ig < iNG; ig++) {
    const std::size_t g = static_cast<std::size_t>(ig);

    for (std::size_t n = 0; n < mu_.size(); n++) {
      const double mu_n = mu_[n];
      const double wgt_n = wgt_[n];
      double flux_in = 0.;
      double flux_out = 0.;
      double flux_bin = 0.;
      std::size_t s = 0;  // surface index

      if (mu_n < 0.) {
        s = this->nsurfaces() - 1;  // Start at far right (last) surface

        // Track from right to left (negative direction)
        flux_in = 0.;

        // Tally incident current
        J_(g, s) += wgt_n * mu_n * flux_in;
        s--;

        for (int ii = static_cast<int>(xs_.size()) - 1; ii >= 0; ii--) {
          const std::size_t i = static_cast<std::size_t>(ii);
          const double dx = dx_[i];
          const double Etr = xs_[i]->Etr(g);
          const double Qni = Q(g, i, 0);

          // Calculate outgoing flux and bin flux
          flux_out =
              (2. * dx * Qni + (2. * std::abs(mu_n) - dx * Etr) * flux_in) /
              (dx * Etr + 2. * std::abs(mu_n));
          flux_bin = 0.5 * (flux_in + flux_out);

          // Contribute to flux legendre moments
          flux(g, i, 0) += wgt_n * flux_bin;

          // Save outgoing flux as an incident flux
          if (i == 0) {
            incident_angular_flux(g, mu_.size() - 1 - n) = flux_out;
          }

          // Tally current at out surface
          J_(g, s) += wgt_n * mu_n * flux_out;
          s--;

          flux_in = flux_out;
        }
      } else {
        s = 0;  // Start at far left (first) surface

        // Track from left to right (positive direction)
        flux_in = incident_angular_flux(g, n);

        // Tally incident current
        J_(g, s) += wgt_n * mu_n * flux_in;
        s++;

        for (std::size_t i = 0; i < xs_.size(); i++) {
          const double dx = dx_[i];
          const double Etr = xs_[i]->Etr(g);
          const double Qni = Q(g, i, 0);

          // Calculate outgoing flux and bin flux
          flux_out =
              (2. * dx * Qni + (2. * std::abs(mu_n) - dx * Etr) * flux_in) /
              (dx * Etr + 2. * std::abs(mu_n));
          flux_bin = 0.5 * (flux_in + flux_out);

          // Contribute to flux legendre moments
          flux(g, i, 0) += wgt_n * flux_bin;

          // Tally current at out surface
          J_(g, s) += wgt_n * mu_n * flux_out;
          s++;

          flux_in = flux_out;
        }
      }
    }  // for all mu
    xt::view(incident_angular_flux, g, xt::all()) = 0.;
  }  // for all groups
}

void ReflectorSN::fill_source_iso(xt::xtensor<double, 3>& Q,
                                  const xt::xtensor<double, 3>& flux) const {
  const double invs_keff = 1. / keff_;
  Q.fill(0.);

#pragma omp parallel for
  for (int ii = 0; ii < static_cast<int>(xs_.size()); ii++) {
    const std::size_t i = static_cast<std::size_t>(ii);
    const auto& mat = xs_[i];

    for (std::size_t g = 0; g < xs_[0]->ngroups(); g++) {
      const double chi_g = mat->chi(g);
      for (std::size_t gg = 0; gg < xs_[0]->ngroups(); gg++) {
        const double flx_gg = flux(gg, i, 0);
        Q(g, i, 0) += 0.5 * mat->Es_tr(gg, g) * flx_gg;
        Q(g, i, 0) += 0.5 * chi_g * mat->vEf(gg) * flx_gg * invs_keff;
      }
    }
  }
}

void ReflectorSN::solve_aniso() {
  const std::size_t NG = xs_[0]->ngroups();
  const std::size_t NR = xs_.size();

  if (max_legendre_order() == 0) {
    const auto mssg =
        "Should not use solve_aniso if the maximum legendre order is 0.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  flux_ = xt::ones<double>({NG, NR, max_L_ + 1});
  xt::xtensor<double, 3> next_flux = xt::zeros<double>({NG, NR, max_L_ + 1});
  xt::xtensor<double, 3> old_flux = xt::zeros<double>({NG, NR, max_L_ + 1});
  xt::xtensor<double, 3> Q = xt::zeros<double>({NG, NR, max_L_ + 1});

  // Evaluate the Legendre function P_l(mu_n) for all mu and all l
  Pnl_ = xt::zeros<double>({mu_.size(), max_L_ + 1});
  for (std::size_t n = 0; n < mu_.size(); n++) {
    for (std::size_t l = 0; l <= max_legendre_order(); l++) {
      Pnl_(n, l) = legendre(static_cast<unsigned int>(l), mu_[n]);
    }
  }

  keff_ = 1.;

  // Array to hold the incident flux at the reflective boundary. We create
  // this here instead of in the sweep to avoid making memory allocations.
  xt::xtensor<double, 2> incident_angular_flux =
      xt::zeros<double>({NG, mu_.size()});

  // Outer Iterations
  double keff_diff = 100.;
  double flux_diff = 100.;
  std::size_t iteration = 0;
  Timer iteration_timer;
  while (keff_diff > keff_tol_ || flux_diff > flux_tol_) {
    iteration_timer.reset();
    iteration_timer.start();
    iteration++;

    old_flux = flux_;

    // We assume the fission source is only isotropic
    fill_source_aniso(Q, flux_);

    // Check for negative P0 source values at beginning of simulation
    bool set_neg_src_to_zero = false;
    if (iteration <= 20) {
      for (std::size_t g = 0; g < NG; g++) {
        for (std::size_t i = 0; i < NR; i++) {
          if (Q(g, i, 0) < 0.) {
            Q(g, i, 0) = 0.;
            set_neg_src_to_zero = true;
          }
        }
      }
    }

    next_flux.fill(0.);
    J_.fill(0.);
    sweep_aniso(next_flux, incident_angular_flux, Q);

    // Get max difference in the scalar flux
    flux_diff = 0.;
    bool set_neg_flux_to_zero = false;
    for (std::size_t g = 0; g < NG; g++) {
      for (std::size_t i = 0; i < NR; i++) {
        const double nf = next_flux(g, i, 0);
        const double f = flux_(g, i, 0);
        const double diff = std::abs(nf - f) / nf;
        if (diff > flux_diff) {
          flux_diff = diff;
        }

        // Make sure that the SCALAR flux is positive everywhere !
        if (nf < 0.) {
          next_flux(g, i, 0) = 0.;
          set_neg_flux_to_zero = true;
        }
      }
    }

    flux_ = next_flux;

    const double old_keff = keff_;
    keff_ = calc_keff(old_flux, flux_, keff_);
    keff_diff = std::abs((old_keff - keff_) / keff_);

    iteration_timer.stop();
    spdlog::info("-------------------------------------");
    spdlog::info("Iteration {:>6d}        keff: {:.5f}", iteration, keff_);
    spdlog::info("     keff difference:     {:.5E}", keff_diff);
    spdlog::info("     max flux difference: {:.5E}", flux_diff);
    spdlog::info("     iteration time: {:.5E} s",
                 iteration_timer.elapsed_time());

    // Write warnings about negative flux and source
    if (set_neg_src_to_zero) {
      spdlog::info("Negative source values set to zero");
    }
    if (set_neg_flux_to_zero) {
      spdlog::info("Negative flux values set to zero");
    }
  }

  solved_ = true;

  // We can unallocate Pnl_ now to save memory
  Pnl_.resize({0, 0});
}

void ReflectorSN::sweep_aniso(xt::xtensor<double, 3>& flux,
                              xt::xtensor<double, 2>& incident_angular_flux,
                              const xt::xtensor<double, 3>& Q) {
  const std::size_t NG = xs_.front()->ngroups();
  const int iNG = static_cast<int>(NG);

  incident_angular_flux.fill(0.);

#pragma omp parallel for
  for (int ig = 0; ig < iNG; ig++) {
    const std::size_t g = static_cast<std::size_t>(ig);

    for (std::size_t n = 0; n < mu_.size(); n++) {
      const double mu_n = mu_[n];
      const double wgt_n = wgt_[n];
      double flux_in = 0.;
      double flux_out = 0.;
      double flux_avg = 0.;
      std::size_t s = 0;  // surface index

      if (mu_n < 0.) {
        s = this->nsurfaces() - 1;  // Start at far right (last) surface

        // Track from right to left (negative direction)
        flux_in = 0.;

        // Tally incident current
        J_(g, s) += wgt_n * mu_n * flux_in;
        s--;

        for (int ii = static_cast<int>(xs_.size()) - 1; ii >= 0; ii--) {
          const std::size_t i = static_cast<std::size_t>(ii);
          const double dx = dx_[i];
          const double Et = xs_[i]->Et(g);

          double Qni = 0.;
          for (std::size_t l = 0; l <= max_legendre_order(); l++) {
            Qni += Q(g, i, l) * Pnl_(n, l);
          }

          // Calculate outgoing flux and average flux
          flux_out =
              (2. * dx * Qni + (2. * std::abs(mu_n) - dx * Et) * flux_in) /
              (dx * Et + 2. * std::abs(mu_n));
          flux_avg = 0.5 * (flux_in + flux_out);

          // Contribute to flux legendre moments
          for (std::size_t l = 0; l <= max_legendre_order(); l++) {
            flux(g, i, l) += wgt_n * flux_avg * Pnl_(n, l);
          }

          // Save outgoing flux as an incident flux
          if (i == 0) {
            incident_angular_flux(g, mu_.size() - 1 - n) = flux_out;
          }

          // Tally current at out surface
          J_(g, s) += wgt_n * mu_n * flux_out;
          s--;

          flux_in = flux_out;
        }
      } else {
        s = 0;  // Start at far left (first) surface

        // Track from left to right (positive direction)
        flux_in = incident_angular_flux(g, n);

        // Tally incident current
        J_(g, s) += wgt_n * mu_n * flux_in;
        s++;

        for (std::size_t i = 0; i < xs_.size(); i++) {
          const double dx = dx_[i];
          const double Et = xs_[i]->Et(g);

          double Qni = 0.;
          for (std::size_t l = 0; l <= max_legendre_order(); l++) {
            Qni += Q(g, i, l) * Pnl_(n, l);
          }

          // Calculate outgoing flux and average flux
          flux_out =
              (2. * dx * Qni + (2. * std::abs(mu_n) - dx * Et) * flux_in) /
              (dx * Et + 2. * std::abs(mu_n));
          flux_avg = 0.5 * (flux_in + flux_out);

          // Contribute to flux legendre moments
          for (std::size_t l = 0; l <= max_legendre_order(); l++) {
            flux(g, i, l) += wgt_n * flux_avg * Pnl_(n, l);
          }

          // Tally current at out surface
          J_(g, s) += wgt_n * mu_n * flux_out;
          s++;

          flux_in = flux_out;
        }
      }
    }  // for all mu
    xt::view(incident_angular_flux, g, xt::all()) = 0.;
  }  // for all groups
}

void ReflectorSN::fill_source_aniso(xt::xtensor<double, 3>& Q,
                                    const xt::xtensor<double, 3>& flux) const {
  const double invs_keff = 1. / keff_;
  Q.fill(0.);

#pragma omp parallel for
  for (int ii = 0; ii < static_cast<int>(xs_.size()); ii++) {
    const std::size_t i = static_cast<std::size_t>(ii);
    const auto& mat = xs_[i];

    for (std::size_t g = 0; g < xs_[0]->ngroups(); g++) {
      const double chi_g = mat->chi(g);
      for (std::size_t gg = 0; gg < xs_[0]->ngroups(); gg++) {
        for (std::size_t l = 0; l <= max_legendre_order(); l++) {
          const double flx_gg_l = flux(gg, i, l);
          Q(g, i, l) += 0.5 * (2. * static_cast<double>(l) + 1.) *
                        mat->Es(l, gg, g) * flx_gg_l;

          if (l == 0) {
            Q(g, i, 0) += 0.5 * invs_keff * chi_g * mat->vEf(gg) * flx_gg_l;
          }
        }
      }
    }
  }
}

double ReflectorSN::calc_keff(const xt::xtensor<double, 3>& old_flux,
                              const xt::xtensor<double, 3>& new_flux,
                              const double keff) const {
  double num = 0.;
  double denom = 0.;

#pragma omp parallel
  {
    double thrd_num = 0.;
    double thrd_denom = 0.;

#pragma omp for
    for (int ii = 0; ii < static_cast<int>(xs_.size()); ii++) {
      const std::size_t i = static_cast<std::size_t>(ii);
      const auto& mat = xs_[i];
      const double dx = dx_[i];
      for (std::size_t g = 0; g < mat->ngroups(); g++) {
        thrd_num += dx * mat->vEf(g) * new_flux(g, i, 0);
        thrd_denom += dx * mat->vEf(g) * old_flux(g, i, 0);
      }
    }
#pragma omp atomic
    num += thrd_num;

#pragma omp atomic
    denom += thrd_denom;
  }

  return keff * num / denom;
}

double ReflectorSN::flux(std::size_t i, std::size_t g, std::size_t l) const {
  if (i >= this->size()) {
    std::stringstream mssg;
    mssg << "Region index i =" << i << " is out of range.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (g >= this->ngroups()) {
    std::stringstream mssg;
    mssg << "Energy group index g =" << g << " is out of range.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (l > max_L_) {
    std::stringstream mssg;
    mssg << "Legendre order l =" << l << " is out of range.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  return flux_(g, i, l);
}

double ReflectorSN::current(std::size_t i, std::size_t g) const {
  if (i >= this->size() + 1) {
    std::stringstream mssg;
    mssg << "Surface index i =" << i << " is out of range.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (g >= this->ngroups()) {
    std::stringstream mssg;
    mssg << "Energy group index g =" << g << " is out of range.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  return J_(g, i);
}

const std::shared_ptr<CrossSection> ReflectorSN::xs(std::size_t i) const {
  if (i >= this->size()) {
    std::stringstream mssg;
    mssg << "Region index i =" << i << " is out of range.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  return xs_[i];
}

double ReflectorSN::volume(std::size_t i) const {
  if (i >= this->size()) {
    std::stringstream mssg;
    mssg << "Region index i =" << i << " is out of range.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  return dx_[i];
}

std::shared_ptr<CrossSection> ReflectorSN::homogenize(
    const std::vector<std::size_t>& regions) const {
  // We can only perform a homogenization if we have a flux spectrum
  if (solved() == false) {
    auto mssg =
        "Cannot perform homogenization when problem has not been solved.";
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
  const double invs_sum_fiss_prod =
      sum_fiss_prod > 0. ? 1. / sum_fiss_prod : 1.;

  for (std::size_t g = 0; g < NG; g++) {
    // Get the sum of flux*volume for this group
    double sum_fluxV = 0.;
    for (const auto i : regions) {
      sum_fluxV += this->flux(i, g) * dx_[i];
    }
    const double invs_sum_fluxV = 1. / sum_fluxV;

    j = 0;
    for (const auto i : regions) {
      const auto& mat = this->xs(i);
      const double V = this->volume(i);
      const double flx = flux(i, g);
      const double coeff = invs_sum_fluxV * flx * V;
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

    // Reconstruct total xs from absorption and scattering
    Et(g) = Ea(g) + xt::sum(xt::view(Es, 0, g, xt::all()))();
  }

  return std::make_shared<CrossSection>(Et, Dtr, Ea, Es, Ef, vEf, chi);
}

xt::xtensor<double, 1> ReflectorSN::homogenize_flux_spectrum(
    const std::vector<std::size_t>& regions) const {
  // We can only perform a homogenization if we have a flux spectrum
  if (solved() == false) {
    auto mssg =
        "Cannot perform spectrum homogenization when problem has not been "
        "solved.";
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

// REFERENCES
// [1] G. Gunow, B. Forget, and K. Smith, “Stabilization of multi-group neutron
//     transport with transport-corrected cross-sections,” Ann. Nucl. Energy,
//     vol. 126, pp. 211–219, 2019, doi: 10.1016/j.anucene.2018.10.036.
