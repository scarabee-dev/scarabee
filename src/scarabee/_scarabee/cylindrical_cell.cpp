#include <cylindrical_cell.hpp>
#include <utils/math.hpp>
#include <utils/constants.hpp>
#include <utils/gauss_kronrod.hpp>
#include <utils/scarabee_exception.hpp>
#include <utils/logging.hpp>

#include <xtensor/generators/xbuilder.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace scarabee {

CylindricalCell::CylindricalCell(
    const std::vector<double>& radii,
    const std::vector<std::shared_ptr<CrossSection>>& mats)
    : p_(),
      X_(),
      Y_(),
      Gamma_(),
      radii_(radii),
      vols_(),
      mats_(mats),
      ngroups_(0),
      solved_(false) {
  // Perform all sanity checks
  if (radii_.size() != mats_.size()) {
    auto mssg = "Number of radii does not match the number of materials.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (radii_.size() < 2) {
    auto mssg = "Must have at least 2 regions.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (std::is_sorted(radii_.begin(), radii_.end()) == false) {
    auto mssg = "Radii are not sorted.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (radii_.front() <= 0.) {
    auto mssg = "All radii must be > 0.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  for (const auto& mat : mats_) {
    if (mat == nullptr) {
      auto mssg = "Nullptr found in materials.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }
  }

  ngroups_ = mats_[0]->ngroups();
  if (ngroups_ == 0) {
    auto mssg = "Must have at least 1 energy group.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  for (const auto& mat : mats_) {
    if (mat->ngroups() != ngroups_) {
      auto mssg = "Not all materials have the same number of energy groups.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }
  }

  // Calculate the volumes
  vols_.resize(radii_.size(), 0.);
  for (std::size_t i = 0; i < radii_.size(); i++) {
    vols_[i] = PI * radii_[i] * radii_[i];

    if (i != 0) {
      vols_[i] -= PI * radii_[i - 1] * radii_[i - 1];
    }
  }
}

void CylindricalCell::solve(bool parallel) {
  spdlog::info("Solving cylindrical cell.");

  if (parallel == false) {
    this->calculate_collision_probabilities();
    this->solve_systems();
  } else {
    this->parallel_calculate_collision_probabilities();
    this->parallel_solve_systems();
  }

  solved_ = true;
}

void CylindricalCell::calculate_collision_probabilities() {
  spdlog::info("Calculating collision probabilities.");

  // First, ensure we have a matrix of the proper size
  p_ = xt::zeros<double>({ngroups(), nregions(), nregions()});

  // Create a matrix to temporarily hold the S_ij factors. We do one group at
  // a time, so we don't need a third axis
  xt::xtensor<double, 2> S = xt::zeros<double>({nregions(), nregions()});

  // Calculate the matrix for each energy group
  for (std::uint32_t g = 0; g < ngroups(); g++) {
    // Load S
    for (std::size_t i = 0; i < nregions(); i++) {
      for (std::size_t j = i; j < nregions(); j++) {
        S(i, j) = calculate_S_ij(i, j, g);

        // These are symmetric, so we can assign the diagonal term too
        if (j != i) S(j, i) = S(i, j);
      }
    }

    // S has now been filled. We can now load p
    for (std::size_t i = 0; i < nregions(); i++) {
      for (std::size_t j = i; j < nregions(); j++) {
        p_(g, i, j) = 0.;
        if (i == j) p_(g, i, j) += vols_[i] * mats_[i]->Etr(g);

        p_(g, i, j) += 2. * S(i, j);
        if (i > 0 && j > 0) p_(g, i, j) += 2. * S(i - 1, j - 1);
        if (i > 0) p_(g, i, j) -= 2. * S(i - 1, j);
        if (j > 0) p_(g, i, j) -= 2. * S(i, j - 1);

        if (i != j) p_(g, j, i) = p_(g, i, j);
      }
    }

    S.fill(0.);
  }  // for groups
}

void CylindricalCell::parallel_calculate_collision_probabilities() {
  spdlog::info("Calculating collision probabilities.");

  // First, ensure we have a matrix of the proper size
  p_ = xt::zeros<double>({ngroups(), nregions(), nregions()});

  // Calculate the matrix for each energy group
#pragma omp parallel for
  for (int ig = 0; ig < static_cast<int>(ngroups()); ig++) {
    std::size_t g = static_cast<std::size_t>(ig);
    // Create a matrix to temporarily hold the S_ij factors. We do one group at
    // a time, so we don't need a third axis
    xt::xtensor<double, 2> S = xt::zeros<double>({nregions(), nregions()});

    // Load S
    for (std::size_t i = 0; i < nregions(); i++) {
      for (std::size_t j = i; j < nregions(); j++) {
        S(i, j) = calculate_S_ij(i, j, g);

        // These are symmetric, so we can assign the diagonal term too
        if (j != i) S(j, i) = S(i, j);
      }
    }

    // S has now been filled. We can now load p
    for (std::size_t i = 0; i < nregions(); i++) {
      for (std::size_t j = i; j < nregions(); j++) {
        p_(g, i, j) = 0.;
        if (i == j) p_(g, i, j) += vols_[i] * mats_[i]->Etr(g);

        p_(g, i, j) += 2. * S(i, j);
        if (i > 0 && j > 0) p_(g, i, j) += 2. * S(i - 1, j - 1);
        if (i > 0) p_(g, i, j) -= 2. * S(i - 1, j);
        if (j > 0) p_(g, i, j) -= 2. * S(i, j - 1);

        if (i != j) p_(g, j, i) = p_(g, i, j);
      }
    }
  }  // for groups
}

void CylindricalCell::solve_systems() {
  spdlog::info("Solving system of equations for cylindrical cell.");

  // First, we allocate the needed memory to hold all of the X and Y terms
  X_ = xt::zeros<double>({ngroups(), nregions(), nregions()});
  Y_ = xt::zeros<double>({ngroups(), nregions()});
  Gamma_ = xt::zeros<double>({ngroups()});

  // The Y and X terms all depend on the energy group, and are described by
  // the same matrix. We therefore load the matrix once, and solve it for
  // all equations.
  for (std::size_t g = 0; g < ngroups(); g++) {
    // Initialize and fill matrix for group g
    Eigen::MatrixXd M(nregions(), nregions());
    for (long i = 0; i < static_cast<long>(nregions()); i++) {
      for (long j = 0; j < static_cast<long>(nregions()); j++) {
        const auto& mat = mats_[static_cast<std::size_t>(j)];
        const double Etr = mat->Etr(g);
        const double Es_tr = mat->Es_tr(g, g);
        const double c_j = Es_tr / Etr;

        M(i, j) = -c_j * p_(g, j, i);

        if (j == i) {
          M(i, j) += Etr * vols_[static_cast<std::size_t>(i)];
        }
      }
    }

    // Now that the matrix has been loaded for this group, we need to
    // initialize a solver for it.
    const auto solver = M.colPivHouseholderQr();

    // There are nregions systems for solve for X
    for (long k = 0; k < static_cast<long>(nregions()); k++) {
      // Load the b vector
      Eigen::VectorXd b(nregions());
      const double Etr_k = mats_[static_cast<std::size_t>(k)]->Etr(g);
      for (long i = 0; i < static_cast<long>(nregions()); i++) {
        b(i) = p_(g, k, i) / Etr_k;
      }

      // Solve for this set of X_ik
      auto X_k = solver.solve(b);

      // Save the X_ik in the array
      for (long i = 0; i < static_cast<long>(nregions()); i++) {
        X_(g, i, k) = X_k(i);
      }
    }

    // We can now solve the equation for Y_i
    Eigen::VectorXd b(nregions());
    const double coeff = 4. / Sb();
    for (long i = 0; i < static_cast<long>(nregions()); i++) {
      const std::size_t indx = static_cast<std::size_t>(i);
      const double Etr_i = mats_[indx]->Etr(g);
      const double Vol_i = vols_[indx];
      double sum_p = 0.;
      for (std::size_t j = 0; j < nregions(); j++) {
        sum_p += p_(g, i, j);
      }

      b(i) = coeff * (Etr_i * Vol_i - sum_p);
    }
    auto Y_i = solver.solve(b);
    for (long i = 0; i < static_cast<long>(nregions()); i++) {
      Y_(g, i) = Y_i(i);
    }

    // Calculate multicollision blackness for this group
    Gamma_(g) = 0.;
    for (std::size_t i = 0; i < nregions(); i++) {
      Gamma_(g) += mats_[i]->Er(g) * vols_[i] * Y_(g, i);
    }
  }  // For all groups
}

void CylindricalCell::parallel_solve_systems() {
  spdlog::info("Solving system of equations for cylindrical cell.");

  // First, we allocate the needed memory to hold all of the X and Y terms
  X_ = xt::zeros<double>({ngroups(), nregions(), nregions()});
  Y_ = xt::zeros<double>({ngroups(), nregions()});
  Gamma_ = xt::zeros<double>({ngroups()});

  // The Y and X terms all depend on the energy group, and are described by
  // the same matrix. We therefore load the matrix once, and solve it for
  // all equations.
#pragma omp parallel for
  for (int ig = 0; ig < static_cast<int>(ngroups()); ig++) {
    std::size_t g = static_cast<std::size_t>(ig);
    // Initialize and fill matrix for group g
    Eigen::MatrixXd M(nregions(), nregions());
    for (long i = 0; i < static_cast<long>(nregions()); i++) {
      for (long j = 0; j < static_cast<long>(nregions()); j++) {
        const auto& mat = mats_[static_cast<std::size_t>(j)];
        const double Etr = mat->Etr(g);
        const double Es_tr = mat->Es_tr(g, g);
        const double c_j = Es_tr / Etr;

        M(i, j) = -c_j * p_(g, j, i);

        if (j == i) {
          M(i, j) += Etr * vols_[static_cast<std::size_t>(i)];
        }
      }
    }

    // Now that the matrix has been loaded for this group, we need to
    // initialize a solver for it.
    const auto solver = M.colPivHouseholderQr();

    // There are nregions systems for solve for X
    for (long k = 0; k < static_cast<long>(nregions()); k++) {
      // Load the b vector
      Eigen::VectorXd b(nregions());
      const double Etr_k = mats_[static_cast<std::size_t>(k)]->Etr(g);
      for (long i = 0; i < static_cast<long>(nregions()); i++) {
        b(i) = p_(g, k, i) / Etr_k;
      }

      // Solve for this set of X_ik
      auto X_k = solver.solve(b);

      // Save the X_ik in the array
      for (long i = 0; i < static_cast<long>(nregions()); i++) {
        X_(g, i, k) = X_k(i);
      }
    }

    // We can now solve the equation for Y_i
    Eigen::VectorXd b(nregions());
    const double coeff = 4. / Sb();
    for (long i = 0; i < static_cast<long>(nregions()); i++) {
      const std::size_t indx = static_cast<std::size_t>(i);
      const double Etr_i = mats_[indx]->Etr(g);
      const double Vol_i = vols_[indx];
      double sum_p = 0.;
      for (std::size_t j = 0; j < nregions(); j++) {
        sum_p += p_(g, i, j);
      }

      b(i) = coeff * (Etr_i * Vol_i - sum_p);
    }
    auto Y_i = solver.solve(b);
    for (long i = 0; i < static_cast<long>(nregions()); i++) {
      Y_(g, i) = Y_i(i);
    }

    // Calculate multicollision blackness for this group
    Gamma_(g) = 0.;
    for (std::size_t i = 0; i < nregions(); i++) {
      Gamma_(g) += mats_[i]->Er(g) * vols_[i] * Y_(g, i);
    }
  }  // For all groups
}

double CylindricalCell::calculate_S_ij(std::size_t i, std::size_t j,
                                       std::size_t g) const {
  if (i > j) std::swap(i, j);

  /*
   * We must solve S_{i,j} = int_{0}^{R_i} (Ki3(Tmax(y)) - Ki3(Tmin(y))) dy.
   * This integral is performed by doing the integral out to each annular ring
   * and adding it to a sum.
   * */

  double S_ij = 0.;

  // Initialize a vector for the x coordinates
  std::vector<double> x;
  x.reserve(j + 1);

  for (std::size_t k = 0; k <= i; k++) {
    // We do the integral from radii_[k-1] to radii_[k]. If k == 0, then we
    // start the integral from the center of the cylinder.
    // Get min and max radii for determining y points
    const double Rmin = (k == 0 ? 0. : radii_[k - 1]);
    const double Rmax = radii_[k];

    // Initialize a vector for the x coordinates
    x.clear();
    x.resize(j + 1 - k, 0.);

    // Create a lambda for the integrand
    auto integrand = [this, &x, k, i, j, g](double y) {
      const double y2 = y * y;

      // Get the x coordinates for the given y
      for (std::size_t n = 0; n < x.size(); n++) {
        x[n] = std::sqrt(radii_[k + n] * radii_[k + n] - y2);
      }

      // Calculate tau_pls and tau_min by iterating through all segments
      double tau_pls = 0.;
      double tau_min = 0.;
      for (std::size_t s = 0; s < x.size(); s++) {
        // Get the length of the segment
        double t = (s == 0 ? x[s] : x[s] - x[s - 1]);

        // Get the material
        const auto& mat = mats_[s + k];

        // Get optical depth constribution
        const double dtau = t * mat->Etr(g);

        // Add to the tau variables
        if (s + k <= i) {
          tau_pls += 2. * dtau;
        } else {
          tau_pls += dtau;
          tau_min += dtau;
        }
      }
      if (i == j) tau_min = 0.0;

      return Ki3(tau_pls) - Ki3(tau_min);
    };

    // We now integrate from Rmin to Rmax
    GaussKronrodQuadrature<15> gk;
    auto integral = gk.integrate(integrand, Rmin, Rmax);

    // TODO check integral

    S_ij += integral.first;
  }

  return S_ij;
}

}  // namespace scarabee
