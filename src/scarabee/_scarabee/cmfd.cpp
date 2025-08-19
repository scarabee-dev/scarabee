#include <moc/cmfd.hpp>
#include <moc/moc_driver.hpp>
#include <utils/constants.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>
#include <utils/simulation_mode.hpp>
#include <utils/timer.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include "spdlog/spdlog.h"

namespace scarabee {

CMFD::CMFD(const std::vector<double>& dx, const std::vector<double>& dy,
           const std::vector<std::pair<std::size_t, std::size_t>>& groups)
    : dx_(dx),
      dy_(dy),
      x_bounds_(),
      y_bounds_(),
      moc_to_cmfd_group_map_(),
      group_condensation_(groups),
      nx_(dx_.size()),
      ny_(dy_.size()),
      ng_(groups.size()),
      nx_surfs_(),
      ny_surfs_(),
      temp_fsrs_(),
      fsrs_() {
  // Make sure we have at least 1 bin in each direction
  if (dx_.size() == 0) {
    auto mssg = "Must provide at least 1 x width.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (dy_.size() == 0) {
    auto mssg = "Must provide at least 1 y width.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure all widths > 0
  double dx_tot = 0.;
  for (std::size_t i = 0; i < dx_.size(); i++) {
    if (dx_[i] <= 0.) {
      std::stringstream mssg;
      mssg << "dx at index " << i << " is <= 0.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    dx_tot += dx_[i];
  }

  double dy_tot = 0.;
  for (std::size_t j = 0; j < dy_.size(); j++) {
    if (dy_[j] <= 0.) {
      std::stringstream mssg;
      mssg << "dy at index " << j << " is <= 0.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    dy_tot += dy_[j];
  }

  // Create surfaces
  x_bounds_.reserve(dx_.size() + 1);
  x_bounds_.emplace_back(-0.5 * dx_tot);
  for (const auto& d : dx_) {
    const double new_x0 = x_bounds_.back().x0() + d;

    x_bounds_.emplace_back(new_x0);
  }

  y_bounds_.reserve(dy_.size() + 1);
  y_bounds_.emplace_back(-0.5 * dy_tot);
  for (const auto& d : dy_) {
    const double new_y0 = y_bounds_.back().y0() + d;

    y_bounds_.emplace_back(new_y0);
  }

  nx_surfs_ = x_bounds_.size() * (y_bounds_.size() - 1);
  ny_surfs_ = y_bounds_.size() * (x_bounds_.size() - 1);

  // Allocate temp_fsrs_
  temp_fsrs_.resize(nx_ * ny_);

  // Check group condensation scheme
  if (group_condensation_.size() == 0) {
    auto mssg = "Empty energy condensation scheme provided.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (group_condensation_.front().first != 0) {
    auto mssg = "The energy condensation scheme does not start with 0.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  for (std::size_t i = 0; i < group_condensation_.size() - 1; i++) {
    if (group_condensation_[i].second + 1 != group_condensation_[i + 1].first) {
      std::stringstream mssg;
      mssg << "Condensed groups " << i << " and " << i + 1
           << " are not continuous.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
  }

  moc_to_cmfd_group_map_.resize(group_condensation_.back().second + 1);
  for (std::size_t g = 0; g < moc_to_cmfd_group_map_.size(); g++) {
    for (std::size_t G = 0; G < group_condensation_.size(); G++) {
      if (group_condensation_[G].first <= g &&
          g <= group_condensation_[G].second) {
        moc_to_cmfd_group_map_[g] = G;
        break;
      }
    }
  }

  // Allocate surfaces array
  surface_currents_ =
      xt::zeros<double>({group_condensation_.size(), nx_surfs_ + ny_surfs_});

  // Allocate the xs array
  xs_.resize({nx_, ny_});
  xs_.fill(nullptr);

  // Allocate cell volume array
  volumes_.resize(nx_ * ny_);
  for (std::size_t i = 0; i < nx_; i++) {
    for (std::size_t j = 0; j < ny_; j++) {
      // Store CMFD cell volume
      const auto indx = tile_to_indx(i, j);
      volumes_[indx] = dx_[i] * dy_[j];
    }
  }

  // Set CMFD fluxes to 1
  flux_cmfd_.resize(ng_ * nx_ * ny_);
  flux_cmfd_.setOnes();

  // Allocate flux update ratio array
  update_ratios_.resize(ng_ * nx_ * ny_);
  update_ratios_.setOnes();

  // Allocate external source array
  extern_src_.resize(ng_ * nx_ * ny_);
  extern_src_.setZero();

  // Allocate the flux, Et, and D_transp_corr arrays
  flux_ = xt::zeros<double>({ng_, nx_, ny_});
  Et_ = xt::zeros<double>({ng_, nx_, ny_});
  D_transp_corr_ = xt::zeros<double>({ng_, nx_surfs_ + ny_surfs_});
}

std::optional<std::array<std::size_t, 2>> CMFD::get_tile(
    const Vector& r, const Direction& u) const {
  for (std::size_t i = 0; i < nx(); i++) {
    for (std::size_t j = 0; j < ny(); j++) {
      // Get the surfaces that make up our tile
      const auto& xl = x_bounds_[i];
      const auto& xh = x_bounds_[i + 1];
      const auto& yl = y_bounds_[j];
      const auto& yh = y_bounds_[j + 1];

      // Check if we are in the tile. If so, return index
      if (xl.side(r, u) == Surface::Side::Positive &&
          xh.side(r, u) == Surface::Side::Negative &&
          yl.side(r, u) == Surface::Side::Positive &&
          yh.side(r, u) == Surface::Side::Negative) {
        return std::array<std::size_t, 2>{i, j};
      }
    }
  }

  // If we get here, we weren't in any tile.
  // Return nullopt
  return std::nullopt;
}

CMFDSurfaceCrossing CMFD::get_surface(const Vector& r,
                                      const Direction& u) const {
  CMFDSurfaceCrossing surface;

  // First, we try to get a tile
  auto otile = this->get_tile(r, u);
  if (otile.has_value() == false) {
    const auto mssg =
        "Could not find a CMFD tile when looking for CMFD surface.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  const std::size_t i = (*otile)[0];
  const std::size_t j = (*otile)[1];

  surface.cell_index = tile_to_indx(*otile);

  // Now we get our surfaces for this tile
  const auto& x_n = x_bounds_[i];
  const auto& x_p = x_bounds_[i + 1];
  const auto& y_n = y_bounds_[j];
  const auto& y_p = y_bounds_[j + 1];

  // Get the distances
  const double dx_n = std::abs(x_n.x0() - r.x());
  const double dx_p = std::abs(x_p.x0() - r.x());
  const double dy_n = std::abs(y_n.y0() - r.y());
  const double dy_p = std::abs(y_p.y0() - r.y());

  // start with corners, then sides
  // top right
  if (dx_p < SURFACE_COINCIDENT && dy_p < SURFACE_COINCIDENT) {
    surface.crossing = CMFDSurfaceCrossing::Type::I;
  } else if (dx_p < SURFACE_COINCIDENT && dy_n < SURFACE_COINCIDENT) {
    surface.crossing = CMFDSurfaceCrossing::Type::IV;
  } else if (dx_n < SURFACE_COINCIDENT && dy_n < SURFACE_COINCIDENT) {
    surface.crossing = CMFDSurfaceCrossing::Type::III;
  } else if (dx_n < SURFACE_COINCIDENT && dy_p < SURFACE_COINCIDENT) {
    surface.crossing = CMFDSurfaceCrossing::Type::II;
  } else if (dx_p < SURFACE_COINCIDENT) {
    surface.crossing = CMFDSurfaceCrossing::Type::XP;
  } else if (dx_n < SURFACE_COINCIDENT) {
    surface.crossing = CMFDSurfaceCrossing::Type::XN;
  } else if (dy_p < SURFACE_COINCIDENT) {
    surface.crossing = CMFDSurfaceCrossing::Type::YP;
  } else if (dy_n < SURFACE_COINCIDENT) {
    surface.crossing = CMFDSurfaceCrossing::Type::YN;
  } else {
    surface.is_valid = false;
    return surface;
  }

  surface.is_valid = true;
  return surface;
}

std::size_t CMFD::tile_to_indx(const std::array<std::size_t, 2>& tile) const {
  return this->tile_to_indx(tile[0], tile[1]);
}

std::size_t CMFD::tile_to_indx(const std::size_t& i,
                               const std::size_t& j) const {
  return j * nx_ + i;
}

std::array<std::size_t, 2> CMFD::indx_to_tile(std::size_t cell_index) {
  std::array<std::size_t, 2> tile;
  tile[0] = cell_index % nx_;
  tile[1] = (cell_index - tile[0]) / nx_;
  return tile;
}

double CMFD::x_min() const {
  double width_x = std::accumulate(dx_.begin(), dx_.end(), 0.0);
  return -0.5 * width_x;
}
double CMFD::x_max() const {
  double width_x = std::accumulate(dx_.begin(), dx_.end(), 0.0);
  return 0.5 * width_x;
}
double CMFD::y_min() const {
  double width_y = std::accumulate(dy_.begin(), dy_.end(), 0.0);
  return -0.5 * width_y;
}
double CMFD::y_max() const {
  double width_y = std::accumulate(dy_.begin(), dy_.end(), 0.0);
  return 0.5 * width_y;
}

void CMFD::insert_fsr(const std::array<std::size_t, 2>& tile, std::size_t fsr) {
  // Compute linear index
  const std::size_t i = this->tile_to_indx(tile);
  temp_fsrs_.at(i).insert(fsr);
}

void CMFD::insert_fsr(std::size_t tile_indx, std::size_t fsr) {
  temp_fsrs_.at(tile_indx).insert(fsr);
}

void CMFD::pack_fsr_lists() {
  fsrs_.resize(nx_ * ny_, std::vector<std::size_t>());

  for (std::size_t i = 0; i < fsrs_.size(); i++) {
    fsrs_[i].insert(fsrs_[i].begin(), temp_fsrs_[i].begin(),
                    temp_fsrs_[i].end());
  }

  temp_fsrs_.clear();
  temp_fsrs_.shrink_to_fit();
}

std::size_t CMFD::moc_to_cmfd_group(std::size_t g) const {
  if (g >= moc_to_cmfd_group_map_.size()) {
    auto mssg = "Group index is out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  return moc_to_cmfd_group_map_[g];
}

const double& CMFD::flux(const std::size_t i, const std::size_t j,
                         const std::size_t g) const {
  if (g > ng_) {
    auto mssg = "Group index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  if (i >= nx_) {
    auto mssg = "Cell x index is out of range";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  if (j >= nx_) {
    auto mssg = "Cell y index is out of range";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  const std::size_t cell_index = tile_to_indx(i, j);

  return flux_cmfd_(g * nx_ * ny_ + cell_index);
}

const double& CMFD::current(const std::size_t G,
                            const std::size_t surface) const {
  if (G >= surface_currents_.shape()[0]) {
    auto mssg = "Group index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (surface >= surface_currents_.shape()[1]) {
    auto mssg = "Surface index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  return surface_currents_(G, surface);
}

std::size_t CMFD::get_x_neg_surf(const std::size_t i,
                                 const std::size_t j) const {
  if (i >= nx_) {
    const auto mssg = "Index i along x axis is out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (j >= ny_) {
    const auto mssg = "Index j along y axis is out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  return (nx_ + 1) * j + i;
}

std::size_t CMFD::get_x_pos_surf(const std::size_t i,
                                 const std::size_t j) const {
  if (i >= nx_) {
    const auto mssg = "Index i along x axis is out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (j >= ny_) {
    const auto mssg = "Index j along y axis is out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  return get_x_neg_surf(i, j) + 1;
}

std::size_t CMFD::get_y_neg_surf(const std::size_t i,
                                 const std::size_t j) const {
  if (i >= nx_) {
    const auto mssg = "Index i along x axis is out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (j >= ny_) {
    const auto mssg = "Index j along y axis is out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  return nx_surfs_ + (ny_ + 1) * i + j;
}

std::size_t CMFD::get_y_pos_surf(const std::size_t i,
                                 const std::size_t j) const {
  if (i >= nx_) {
    const auto mssg = "Index i along x axis is out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (j >= ny_) {
    const auto mssg = "Index j along y axis is out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  return get_y_neg_surf(i, j) + 1;
}

void CMFD::tally_current(double aflx, const Direction& u, std::size_t G,
                         const CMFDSurfaceCrossing& surf) {
  if (G >= surface_currents_.shape()[0]) {
    auto mssg = "Group index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  const auto tile = indx_to_tile(surf.cell_index);
  std::size_t i = tile[0];
  std::size_t j = tile[1];

  // Get surface index(s) from CMFDSurfaceCrossing
  htl::static_vector<std::size_t, 4> surf_indexes;
  bool is_corner = false;
  if (surf.crossing == CMFDSurfaceCrossing::Type::XN) {
    surf_indexes.push_back(get_x_neg_surf(i, j));
  } else if (surf.crossing == CMFDSurfaceCrossing::Type::XP) {
    surf_indexes.push_back(get_x_pos_surf(i, j));
  } else if (surf.crossing == CMFDSurfaceCrossing::Type::YN) {
    surf_indexes.push_back(get_y_neg_surf(i, j));
  } else if (surf.crossing == CMFDSurfaceCrossing::Type::YP) {
    surf_indexes.push_back(get_y_pos_surf(i, j));
  } else if (surf.crossing == CMFDSurfaceCrossing::Type::I) {
    is_corner = true;
    surf_indexes.push_back(get_x_pos_surf(i, j));
    surf_indexes.push_back(get_y_pos_surf(i, j));
    if (i + 1 < nx_) {
      surf_indexes.push_back(get_y_pos_surf(i + 1, j));
    }
    if (j + 1 < ny_) {
      surf_indexes.push_back(get_x_pos_surf(i, j + 1));
    }
  } else if (surf.crossing == CMFDSurfaceCrossing::Type::IV) {
    is_corner = true;
    surf_indexes.push_back(get_x_pos_surf(i, j));
    surf_indexes.push_back(get_y_neg_surf(i, j));
    if (i + 1 < nx_) {
      surf_indexes.push_back(get_y_neg_surf(i + 1, j));
    }
    if (j != 0) {
      surf_indexes.push_back(get_x_pos_surf(i, j - 1));
    }
  } else if (surf.crossing == CMFDSurfaceCrossing::Type::III) {
    is_corner = true;
    surf_indexes.push_back(get_x_neg_surf(i, j));
    surf_indexes.push_back(get_y_neg_surf(i, j));
    if (i != 0) {
      surf_indexes.push_back(get_y_neg_surf(i - 1, j));
    }
    if (j != 0) {
      surf_indexes.push_back(get_x_neg_surf(i, j - 1));
    }
  } else if (surf.crossing == CMFDSurfaceCrossing::Type::II) {
    is_corner = true;
    surf_indexes.push_back(get_x_neg_surf(i, j));
    surf_indexes.push_back(get_y_pos_surf(i, j));
    if (i != 0) {
      surf_indexes.push_back(get_y_pos_surf(i - 1, j));
    }
    if (j + 1 < ny_) {
      surf_indexes.push_back(get_x_neg_surf(i, j + 1));
    }
  }

  if (is_corner) {
    // Split flux between two surfaces evenly
    aflx *= 0.5;
  }

  for (auto si : surf_indexes) {
    if (si < nx_surfs_) {
#pragma omp atomic
      surface_currents_(G, si) += std::copysign(aflx, u.x());
    } else {
#pragma omp atomic
      surface_currents_(G, si) += std::copysign(aflx, u.y());
    }
  }
}

void CMFD::normalize_currents() {
  // We must normalize the currents by the lengths of each surface

  // First, go through all y-levels, and normalize the x surfaces
  std::size_t s = 0;
  for (std::size_t j = 0; j < ny_; j++) {
    const double dy = y_bounds_[j + 1].y0() - y_bounds_[j].y0();
    const double invs_dy = 1. / dy;

    for (std::size_t xs = 0; xs < x_bounds_.size(); xs++) {
      xt::view(surface_currents_, xt::all(), s) *= invs_dy;
      s++;
    }
  }

  // Now go through all x-levels, and normalize the y surfaces
  for (std::size_t i = 0; i < nx_; i++) {
    const double dx = x_bounds_[i + 1].x0() - x_bounds_[i].x0();
    const double invs_dx = 1. / dx;

    for (std::size_t ys = 0; ys < y_bounds_.size(); ys++) {
      xt::view(surface_currents_, xt::all(), s) *= invs_dx;
      s++;
    }
  }

  surface_currents_normalized_ = true;
}

void CMFD::compute_homogenized_xs_and_flux(const MOCDriver& moc) {
#pragma omp parallel for
  for (int ii = 0; ii < static_cast<int>(nx_); ii++) {
    std::size_t i = static_cast<std::size_t>(ii);

    for (std::size_t j = 0; j < ny_; j++) {
      const auto indx = this->tile_to_indx(i, j);
      const auto fg_xs = moc.homogenize(fsrs_[indx]);
      const auto fg_dxs = fg_xs->diffusion_xs();
      const auto flux_spec = moc.homogenize_flux_spectrum(fsrs_[indx]);
      auto& xs = xs_(i, j);
      if (xs)
        *xs = *(fg_dxs->condense(group_condensation_, flux_spec));
      else
        xs = fg_dxs->condense(group_condensation_, flux_spec);

      // Generate the flux and Et values for the tile
      xt::view(flux_, xt::all(), i, j) = 0.;
      xt::view(Et_, xt::all(), i, j) = 0.;
      for (std::size_t g = 0; g < moc_to_cmfd_group_map_.size(); g++) {
        flux_(moc_to_cmfd_group_map_[g], i, j) += flux_spec(g);
        Et_(moc_to_cmfd_group_map_[g], i, j) += fg_xs->Etr(g) * flux_spec(g);
      }
      xt::view(Et_, xt::all(), i, j) /= xt::view(flux_, xt::all(), i, j);
    }
  }
}

void CMFD::homogenize_ext_src(const MOCDriver& moc) {
  // Reset external source array
  extern_src_.setZero();
  const std::size_t tot_cells = nx_ * ny_;
  // Loop over all cells
  for (std::size_t l = 0; l < tot_cells; l++) {
    const double invs_V = 1. / volumes_[l];
    // Loop over MOC groups
    for (std::size_t g = 0; g < moc_to_cmfd_group_map_.size(); g++) {
      // Loop over FSRs in cell l
      for (auto fsr : fsrs_[l]) {
        extern_src_[moc_to_cmfd_group_map_[g] * tot_cells + l] +=
            invs_V * moc.volume(fsr) * moc.extern_src(fsr, g);
      }
    }
  }
}

void CMFD::check_neutron_balance(const std::size_t i, const std::size_t j,
                                 std::size_t g, const double keff) const {
  // First, we get the spacing of the tile
  const double dx = dx_[i];
  const double dy = dy_[j];

  // Next, get the surfaces for out tile
  const double J_xn = surface_currents_.at(g, j * x_bounds_.size() + i);
  const double J_xp = surface_currents_.at(g, j * x_bounds_.size() + i + 1);
  const double J_yn =
      surface_currents_.at(g, nx_surfs_ + i * y_bounds_.size() + j);
  const double J_yp =
      surface_currents_.at(g, nx_surfs_ + i * y_bounds_.size() + j + 1);

  // Get the xs for our tile
  const auto& xs = *xs_(i, j);
  const double chi_g_keff = xs.chi(g) / keff;

  // Compute the fission and scatter sources
  double fiss_source = 0.;
  double scat_source = 0.;
  for (std::size_t gg = 0; gg < ng_; gg++) {
    const double flx_gg = flux_.at(gg, i, j);
    fiss_source += chi_g_keff * xs.vEf(gg) * flx_gg;
    scat_source += xs.Es(gg, g) * flx_gg;
  }

  // Compute the total leakage from the currents
  const double leak_rate = ((J_xp - J_xn) / dx) + ((J_yp - J_yn) / dy);

  // Now compute the removal reaction rate
  const double tot_reac_rate = Et_(g, i, j) * flux_.at(g, i, j);

  // Compute the residual of the balance equation
  const double residual =
      leak_rate + tot_reac_rate - (scat_source + fiss_source);

  if (std::abs(residual) >= 1.E-5) {
    spdlog::error(
        "CMFD tile ({:d}, {:d}) in group {:d} has a neutron balance residual "
        "of "
        "{:.5E}.",
        i, j, g, residual);
  }
}

void CMFD::set_damping(double wd) {
  if (wd < 0.0 || wd > 1.0) {
    auto mssg = "Damping factor for CMFD must be in interval [0.0, 1.0].";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  damping_ = wd;
}

void CMFD::set_flux_tolerance(double ftol) {
  if (ftol <= 0. || ftol >= 0.1) {
    auto mssg = "Tolerance for flux must be in the interval (0., 0.1).";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  flux_tol_ = ftol;
}

void CMFD::set_keff_tolerance(double ktol) {
  if (ktol <= 0. || ktol >= 0.1) {
    auto mssg = "Tolerance for keff must be in the interval (0., 0.1).";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  keff_tol_ = ktol;
}

void CMFD::set_larsen_correction(bool user_pref) {
  if (od_cmfd_ && user_pref) {
    auto mssg =
        "odCMFD and Larsen Correction options are mutally exclusive. Disable "
        "odCMFD with 'od_cmfd = False'.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  larsen_correction_ = user_pref;
}

void CMFD::set_od_cmfd(bool user_pref) {
  if (larsen_correction_ && user_pref) {
    auto mssg =
        "odCMFD and Larsen Correction options are mutally exclusive. Disable "
        "Larsen Correction with 'larsen_correction = False'.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  od_cmfd_ = user_pref;
}

std::variant<std::array<std::size_t, 2>, BoundaryCondition>
CMFD::find_next_cell_or_bc(std::size_t i, std::size_t j, CMFD::TileSurf surf,
                           const MOCDriver& moc) const {
  switch (surf) {
    case CMFD::TileSurf::XP:
      if (i + 1 == nx_) {
        const auto bc = moc.x_max_bc();
        if (bc == BoundaryCondition::Periodic) {
          return std::array<std::size_t, 2>{0, j};
        }
        return bc;
      }
      return std::array<std::size_t, 2>{i + 1, j};
      break;

    case CMFD::TileSurf::XN:
      if (i == 0) {
        const auto bc = moc.x_min_bc();
        if (bc == BoundaryCondition::Periodic) {
          return std::array<std::size_t, 2>{nx_ - 1, j};
        }
        return bc;
      }
      return std::array<std::size_t, 2>{i - 1, j};
      break;

    case CMFD::TileSurf::YP:
      if (j + 1 == ny_) {
        const auto bc = moc.y_max_bc();
        if (bc == BoundaryCondition::Periodic) {
          return std::array<std::size_t, 2>{i, 0};
        }
        return bc;
      }
      return std::array<std::size_t, 2>{i, j + 1};
      break;

    case CMFD::TileSurf::YN:
      if (j == 0) {
        const auto bc = moc.y_min_bc();
        if (bc == BoundaryCondition::Periodic) {
          return std::array<std::size_t, 2>{i, ny_ - 1};
        }
        return bc;
      }
      return std::array<std::size_t, 2>{i, j - 1};
      break;
  }

  // NEVER GETS HERE
  return std::array<std::size_t, 2>{std::numeric_limits<std::size_t>::max(),
                                    std::numeric_limits<std::size_t>::max()};
}

double CMFD::get_cmfd_tile_width(std::size_t i, std::size_t j,
                                 CMFD::TileSurf surf) const {
  switch (surf) {
    case CMFD::TileSurf::XP:
    case CMFD::TileSurf::XN:
      return dx_[i];
      break;
    case CMFD::TileSurf::YP:
    case CMFD::TileSurf::YN:
      return dy_[j];
      break;
  }

  // NEVER GETS HERE
  return 0.;
}

double CMFD::get_current(std::size_t i, std::size_t j, std::size_t g,
                         CMFD::TileSurf surf) const {
  switch (surf) {
    case CMFD::TileSurf::XP:
      return surface_currents_(g, get_x_pos_surf(i, j));
      break;
    case CMFD::TileSurf::XN:
      return surface_currents_(g, get_x_neg_surf(i, j));
      break;
    case CMFD::TileSurf::YP:
      return surface_currents_(g, get_y_pos_surf(i, j));
      break;
    case CMFD::TileSurf::YN:
      return surface_currents_(g, get_y_neg_surf(i, j));
      break;
  }

  // NEVER GETS HERE
  return 0.;
}

void CMFD::apply_larsen_correction(double& D, const double dx,
                                   const MOCDriver& moc) const {
  // Correct Diffusion Coefficient to agree with infinite medium
  // for optically thick mesh cells. Based on the derivation in [1].
  const PolarQuadrature& moc_polq = moc.polar_quadrature();
  const auto& moc_azmq = moc.azimuthal_quadrature();
  const std::size_t n_pol_angles = moc_polq.sin().size();
  const auto& pol_wgts = moc_polq.wgt();
  const auto& polar_angles = moc_polq.polar_angle();

  double rho = 0.;
  for (std::size_t a = 0; a < moc_azmq.size(); a++) {
    const double& wa = moc_azmq[a].wgt;
    // Loop over polar angles
    for (std::size_t p = 0; p < n_pol_angles; p++) {
      const double f = dx / (3. * D * std::cos(polar_angles[p]));
      const double alpha =
          ((1. + std::exp(-f)) / (1. - std::exp(-f))) - (2. / (f));
      rho += 2.0 * std::cos(polar_angles[p]) * pol_wgts[p] * wa * alpha;
    }
  }
  // Calculate effective diffusion coefficient
  const double D_eff = D * (1. + ((dx * rho) / (2. * D)));

  // Set D to effective diffusion coefficient
  D = D_eff;
}

void CMFD::optimize_diffusion_coef(double& D, const double dx,
                                   const std::size_t i, const std::size_t j,
                                   const std::size_t g) const {
  // This method adds a term to the diffusion coefficient which generalizes CMFD
  // and pCMFD to be the same based on the optimal diffusion coefficient for a
  // given cell's optical thickness. This method is based on the result from
  // [2].
  const double EtDx = Et_(g, i, j) * dx;

  if (EtDx < 1.) {
    return;
  } else if (EtDx < 14 && EtDx >= 1) {
    const double EtDx2 = EtDx * EtDx;
    const double EtDx3 = EtDx2 * EtDx;
    const double EtDx4 = EtDx3 * EtDx;
    const double EtDx5 = EtDx4 * EtDx;
    const double EtDx6 = EtDx5 * EtDx;
    const double theta = -5.542780E-02 + 8.740501E-02 * EtDx +
                         -2.152599E-02 * EtDx2 + 3.145553E-03 * EtDx3 +
                         -2.683648E-04 * EtDx4 + 1.222516E-05 * EtDx5 +
                         -2.284879E-07 * EtDx6;

    D += theta * dx;
  } else {
    D += 0.127 * dx;
  }
}

std::pair<double, double> CMFD::calc_surf_diffusion_coeffs(
    std::size_t i, std::size_t j, std::size_t g, CMFD::TileSurf surf,
    const MOCDriver& moc) const {
  // Get flux and diffusion coefficient for current cell
  const double flx_ij = flux_(g, i, j);
  double D_ij = xs_(i, j)->D(g);
  const double dx_ij = get_cmfd_tile_width(i, j, surf);
  const double current = get_current(i, j, g, surf);

  // Modify material diffusion coefficient for cell i,j by Larsen's correction
  // or odCMFD
  if (larsen_correction_) {
    apply_larsen_correction(D_ij, dx_ij, moc);
  } else if (od_cmfd_) {
    optimize_diffusion_coef(D_ij, dx_ij, i, j, g);
  }

  const auto apply_flux_limiting = [current, surf, flx_ij](
                                       double& D_surf, double& D_nl,
                                       const double flx_next) {
    // Flux limiting condition
    if (std::abs(D_nl) > std::abs(D_surf)) {
      if (surf == CMFD::TileSurf::XP || surf == CMFD::TileSurf::YP) {
        D_surf = current / (-2. * flx_next);
      } else {
        D_surf = current / (-2. * flx_ij);
      }
      D_nl = D_surf;
    }
  };

  // Get the next CMFD tile or BC (only Reflective or Vacuum).
  // Periodic is treated like an interior cell.
  const auto next_tile_or_bc = find_next_cell_or_bc(i, j, surf, moc);

  // If we have a BC, handle that here
  if (std::holds_alternative<BoundaryCondition>(next_tile_or_bc)) {
    const auto bc = std::get<BoundaryCondition>(next_tile_or_bc);

    if (bc == BoundaryCondition::Reflective) {
      return {0., 0.};
    }

    // Vacuum BC
    double D_surf = 2. * D_ij / (4. * D_ij + dx_ij);
    double D_nl = 0.;
    if (surf == CMFD::TileSurf::XP || surf == CMFD::TileSurf::YP) {
      D_nl = -(current - D_surf * flx_ij) / flx_ij;
    } else {
      D_nl = -(current + D_surf * flx_ij) / flx_ij;
    }

    if (flux_limiting_) {
      if (surf == CMFD::TileSurf::XN || surf == CMFD::TileSurf::YN) {
        apply_flux_limiting(D_surf, D_nl, 0.);
      }
    }

    if (moc_iteration_ == 0) {
      D_nl = 0.0;
    }

    return {D_surf, D_nl};
  }

  // Get indices to next CMFD tile
  const auto [ii, jj] = std::get<std::array<std::size_t, 2>>(next_tile_or_bc);

  // Get flux, diffusion coefficient, and length for next tile
  const double flx_iijj = flux_(g, ii, jj);
  double D_iijj = xs_(ii, jj)->D(g);
  const double dx_iijj = get_cmfd_tile_width(ii, jj, surf);

  // Modify material diffusion coefficient for cell ii, jj by Larsen's
  // correction or odCMFD
  if (larsen_correction_) {
    apply_larsen_correction(D_iijj, dx_iijj, moc);
  } else if (od_cmfd_) {
    optimize_diffusion_coef(D_iijj, dx_iijj, ii, jj, g);
  }

  // First, compute normal surface diffusion coefficient
  double D_surf = (2 * D_ij * D_iijj) / (D_ij * dx_iijj + D_iijj * dx_ij);

  // Compute non-linear diffusion coefficient
  double D_nl = 0.;
  if (surf == CMFD::TileSurf::XP || surf == CMFD::TileSurf::YP) {
    D_nl = (-D_surf * (flx_iijj - flx_ij) - current) / (flx_iijj + flx_ij);
  } else {
    D_nl = (D_surf * (flx_iijj - flx_ij) - current) / (flx_iijj + flx_ij);
  }

  if (flux_limiting_) {
    apply_flux_limiting(D_surf, D_nl, flx_iijj);
  }

  if (moc_iteration_ == 0) {
    D_nl = 0.0;
  }

  // Return coefficients
  return {D_surf, D_nl};
}

void CMFD::create_loss_matrix(const MOCDriver& moc) {
  const std::size_t tot_cells = nx_ * ny_;
  M_.resize(ng_ * tot_cells, ng_ * tot_cells);
  M_.reserve(Eigen::VectorXi::Constant(ng_ * tot_cells, 5));
  // Initialize array to store n+1 diffusion coefficients
  xt::xtensor<double, 2> D_transp_corr_new =
      xt::zeros<double>({ng_, nx_surfs_ + ny_surfs_});

  // Loop over all cells and groups, cell index changes fastest
  for (std::size_t g = 0; g < ng_; ++g) {
    for (std::size_t l = 0; l < nx_ * ny_; l++) {
      const auto [i, j] = indx_to_tile(l);
      const double dx = dx_[i];
      const double dy = dy_[j];

      const double invs_dx = 1. / dx;
      const double invs_dy = 1. / dy;

      const auto xpsurf = get_x_pos_surf(i, j);
      const auto xnsurf = get_x_neg_surf(i, j);
      const auto ypsurf = get_y_pos_surf(i, j);
      const auto ynsurf = get_y_neg_surf(i, j);

      // Get surface diffusion coefficients for Cell i,j
      auto [Dxp, Dnl_xp] =
          calc_surf_diffusion_coeffs(i, j, g, CMFD::TileSurf::XP, moc);
      auto [Dyp, Dnl_yp] =
          calc_surf_diffusion_coeffs(i, j, g, CMFD::TileSurf::YP, moc);
      auto [Dxn, Dnl_xn] =
          calc_surf_diffusion_coeffs(i, j, g, CMFD::TileSurf::XN, moc);
      auto [Dyn, Dnl_yn] =
          calc_surf_diffusion_coeffs(i, j, g, CMFD::TileSurf::YN, moc);

      if (Dnl_xp > Dxp || Dnl_xn > Dxn || Dnl_yp > Dyp || Dnl_yn > Dyn) {
        auto mssg =
            "At least one transport corrected diffusion coefficient is greater "
            "than its non-corrected counterpart";
        spdlog::debug(mssg);
      }

      // Calculate n+1 diffusion coefficients from n-1 and n+1/2
      Dnl_xp = (1 - damping_) * D_transp_corr_(g, xpsurf) + damping_ * Dnl_xp;
      Dnl_xn = (1 - damping_) * D_transp_corr_(g, xnsurf) + damping_ * Dnl_xn;
      Dnl_yp = (1 - damping_) * D_transp_corr_(g, ypsurf) + damping_ * Dnl_yp;
      Dnl_yn = (1 - damping_) * D_transp_corr_(g, ynsurf) + damping_ * Dnl_yn;

      // Store the current CMFD iteration's diffusion coefficients
      D_transp_corr_new(g, xpsurf) = Dnl_xp;
      D_transp_corr_new(g, xnsurf) = Dnl_xn;
      D_transp_corr_new(g, ypsurf) = Dnl_yp;
      D_transp_corr_new(g, ynsurf) = Dnl_yn;

      // Streaming to adjacent X cells
      if (i != 0) {
        M_.coeffRef(g * tot_cells + l,
                    g * tot_cells + tile_to_indx(i - 1, j)) +=
            (Dnl_xn - Dxn) * invs_dx;
      }
      if (i != nx_ - 1) {
        M_.coeffRef(g * tot_cells + l,
                    g * tot_cells + tile_to_indx(i + 1, j)) +=
            (-Dxp - Dnl_xp) * invs_dx;
      }
      M_.coeffRef(g * tot_cells + l, g * tot_cells + l) +=
          (Dxn + Dxp + Dnl_xn - Dnl_xp) * invs_dx;

      // Streaming to adjacent Y cells
      if (j != 0) {
        M_.coeffRef(g * tot_cells + l,
                    g * tot_cells + tile_to_indx(i, j - 1)) +=
            (Dnl_yn - Dyn) * invs_dy;
      }
      if (j != ny_ - 1) {
        M_.coeffRef(g * tot_cells + l,
                    g * tot_cells + tile_to_indx(i, j + 1)) +=
            (-Dyp - Dnl_yp) * invs_dy;
      }
      M_.coeffRef(g * tot_cells + l, g * tot_cells + l) +=
          (Dyn + Dyp + Dnl_yn - Dnl_yp) * invs_dy;

      // Handle periodic BC
      // X direction
      if (i == 0 && moc.x_min_bc() == BoundaryCondition::Periodic) {
        M_.coeffRef(g * tot_cells + l,
                    g * tot_cells + tile_to_indx(nx_ - 1, j)) +=
            (Dnl_xn - Dxn) * invs_dx;
      }
      if (i == nx_ - 1 && moc.x_max_bc() == BoundaryCondition::Periodic) {
        M_.coeffRef(g * tot_cells + l, g * tot_cells + tile_to_indx(0, j)) +=
            (-Dxp - Dnl_xp) * invs_dx;
      }
      // Y direction
      if (j == 0 && moc.y_min_bc() == BoundaryCondition::Periodic) {
        M_.coeffRef(g * tot_cells + l,
                    g * tot_cells + tile_to_indx(i, ny_ - 1)) +=
            (Dnl_yn - Dyn) * invs_dy;
      }
      if (j == nx_ - 1 && moc.y_max_bc() == BoundaryCondition::Periodic) {
        M_.coeffRef(g * tot_cells + l, g * tot_cells + tile_to_indx(i, 0)) +=
            (-Dyp - Dnl_yp) * invs_dy;
      }

      // Add removal xs along diagonal
      M_.coeffRef(g * tot_cells + l, g * tot_cells + l) += xs_(i, j)->Er(g);

      // Remove scattering sources
      for (std::size_t gg = 0; gg < ng_; ++gg) {
        if (gg != g) {
          M_.coeffRef(g * tot_cells + l, gg * tot_cells + l) -=
              xs_(i, j)->Es(gg, g);
        }
      }
    }
  }
  D_transp_corr_ = D_transp_corr_new;
  M_.makeCompressed();
}

void CMFD::create_source_matrix() {
  const std::size_t tot_cells = nx_ * ny_;
  QM_.resize(ng_ * tot_cells, ng_ * tot_cells);
  QM_.reserve(Eigen::VectorX<std::size_t>::Constant(ng_ * tot_cells, ng_));

  // Loop over all cells and groups, cell index changes fastest
  for (std::size_t g = 0; g < ng_; g++) {
    for (std::size_t l = 0; l < nx_ * ny_; l++) {
      auto [i, j] = indx_to_tile(l);
      const double chi_g = xs_(i, j)->chi(g);
      // Loop over all groups again for fission source
      for (std::size_t gg = 0; gg < ng_; gg++) {
        const double vEf_gg = xs_(i, j)->vEf(gg);
        QM_.coeffRef(g * tot_cells + l, gg * tot_cells + l) = chi_g * vEf_gg;
      }
    }
  }
  QM_.makeCompressed();
}

void CMFD::power_iteration(double keff) {
  // Power Iteration to solve for Keff
  // Initialize flux and source vectors
  Eigen::VectorXd new_flux(ng_ * nx_ * ny_);
  Eigen::VectorXd Q(ng_ * nx_ * ny_);

  // Initialize a vector for computing keff faster
  Eigen::VectorXd VvEf(ng_ * nx_ * ny_);
  for (std::size_t l = 0; l < nx_ * ny_; l++) {
    auto [i, j] = indx_to_tile(l);
    for (std::size_t g = 0; g < ng_; g++) {
      double vEf = xs_(i, j)->vEf(g);
      VvEf(l + g * ny_ * nx_) = volumes_[l] * vEf;
    }
  }

  // Create a solver for the problem
  Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
  solver.compute(M_);
  solver.setTolerance(1.E-10);
  if (solver.info() != Eigen::Success) {
    std::stringstream mssg;
    mssg << "Could not initialize CMFD iterative solver";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // Begin power iteration
  double keff_diff = 100.;
  double flux_diff = 100.;
  std::size_t iteration = 0;
  while (keff_diff > keff_tol_ || flux_diff > flux_tol_) {
    iteration++;
    // Compute source vector
    Q = (1. / keff) * QM_ * flux_cmfd_;

    // Get new flux
    new_flux = solver.solveWithGuess(Q, flux_cmfd_);
    // For some reason, this doesn't seem to be working with the new versions
    // of Eigen, despite clearly succeeding. Just commenting it out for now.
    // if (solver.info() != Eigen::Success) {
    //   spdlog::error("Solution impossible.");
    //   throw ScarabeeException("Solution impossible");
    // }

    // Estimate keff
    double prev_keff = keff;
    keff = prev_keff * (VvEf.dot(new_flux) / VvEf.dot(flux_cmfd_));
    keff_diff = std::abs(keff - prev_keff) / keff;

    // Normalize our new flux
    new_flux *= prev_keff / keff;

    // Find the max flux error
    flux_diff = 0.;
    for (std::size_t i = 0; i < ng_ * nx_ * ny_; i++) {
      double flux_diff_i = std::abs(new_flux(i) - flux_cmfd_(i)) / new_flux(i);
      if (flux_diff_i > flux_diff) flux_diff = flux_diff_i;
    }
    flux_cmfd_ = new_flux;
  }
  keff_ = keff;
}

void CMFD::fixed_source_solve() {
  // Solve fixed source problem
  // Subtract fission source from loss matrix
  Eigen::SparseMatrix<double> L = M_ - QM_;

  Eigen::VectorXd new_flux(ng_ * nx_ * ny_);

  // Create a solver for the problem
  Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
  solver.compute(L);
  solver.setTolerance(1.E-10);

  if (solver.info() != Eigen::Success) {
    std::stringstream mssg;
    mssg << "Could not initialize CMFD iterative solver";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // Get new flux
  new_flux = solver.solveWithGuess(extern_src_, flux_cmfd_);
  // For some reason, this doesn't seem to be working with the new versions
  // of Eigen, despite clearly succeeding. Just commenting it out for now.
  // if (solver.info() != Eigen::Success) {
  //   spdlog::error("Solution impossible.");
  //   throw ScarabeeException("Solution impossible");
  // }

  flux_cmfd_ = new_flux;
}

void CMFD::update_moc_fluxes(MOCDriver& moc) {
  // Update MOC FSR scalar fluxes
  const std::size_t tot_cells = ny_ * nx_;
  bool flux_update_warning = false;

  // Normalize homogenized moc flux and cmfd flux
  flux_ /= xt::sum(flux_)();
  flux_cmfd_ /= flux_cmfd_.sum();

  // Precompute update ratio in each CMFD cell
  std::size_t i = 0;
  std::size_t j = 0;
  for (std::size_t l = 0; l < tot_cells; l++) {
    for (std::size_t G = 0; G < ng_; G++) {
      const std::size_t linear_indx = G * tot_cells + l;
      const double invs_flx = 1. / flux_(G, i, j);
      double ratio = flux_cmfd_(linear_indx) * invs_flx;
      // Don't warn on first moc iteration or if ratio is exactly 0.0
      if (ratio != 0.0 && moc_iteration_ > 1 &&
          (ratio < 0.05 || ratio > 20.0)) {
        flux_update_warning = true;
      }
      // Check that the update ratio is valid
      if (ratio < 0.0) {
        auto mssg =
            "Negative CMFD flux update ratio. Try using Larsen correction";
        spdlog::error(mssg);
        throw ScarabeeException(mssg);
      }
      if (ratio == 0.0) {
        auto mssg = "CMFD flux update is zero.";
        spdlog::error(mssg);
        throw ScarabeeException(mssg);
      }
      if (std::isinf(ratio)) {
        auto mssg = "CMFD flux update ratio is inf.";
        spdlog::error(mssg);
        throw ScarabeeException(mssg);
      }
      if (std::isnan(ratio)) {
        auto mssg = "CMFD flux update ratio is NaN.";
        spdlog::error(mssg);
        throw ScarabeeException(mssg);
      }
      // Clamp CMFD flux ratio after a certain # of solves
      if (cmfd_solves_ > unbounded_cmfd_solves_) {
        ratio = std::clamp(ratio, 0.05, 20.0);
      }
      update_ratios_(linear_indx) = ratio;
    }
    i++;
    if (i == nx_) {
      i = 0;
      j++;
    }
  }

  // Loop over MOC groups
  for (std::size_t g = 0; g < moc_to_cmfd_group_map_.size(); g++) {
    // Get CMFD group G from MOC group g
    const std::size_t G = moc_to_cmfd_group_map_[g];

    // Loop over each FSR in CMFD cell i,j
    for (std::size_t l = 0; l < tot_cells; l++) {
      const auto& fsrs = fsrs_[l];
      const std::size_t linear_indx = G * tot_cells + l;
      const double& flx_ratio = update_ratios_(linear_indx);

      // Update scalar flux in each MOC FSR
      for (const auto f : fsrs) {
        for (std::size_t lj = 0; lj < moc.num_spherical_harmonics(); lj++) {
          const double new_flx = moc.flux(f, g, lj) * flx_ratio;
          moc.set_flux(f, g, new_flx, lj);
        }
      }
    }
  }

  // Update MOC angular flux at boundaries
  auto& moc_tracks = moc.tracks();
  for (auto& tracks : moc_tracks) {
    for (auto& track : tracks) {
      std::size_t entry_cell = track.entry_cmfd_cell();
      std::size_t exit_cell = track.exit_cmfd_cell();
      for (std::size_t g = 0; g < moc_to_cmfd_group_map_.size(); g++) {
        const std::size_t G = moc_to_cmfd_group_map_[g];
        const std::size_t g_indx = G * tot_cells;
        xt::view(track.entry_flux(), g, xt::all()) *=
            update_ratios_(g_indx + entry_cell);
        xt::view(track.exit_flux(), g, xt::all()) *=
            update_ratios_(g_indx + exit_cell);
      }
    }
  }

  if (flux_update_warning) {
    spdlog::warn(
        "At least one CMFD flux update ratio outside of (0.05, 20.0).");
  }
}

void CMFD::solve(MOCDriver& moc, double keff, std::size_t moc_iteration) {
  Timer cmfd_timer;
  cmfd_timer.reset();
  cmfd_timer.start();
  solved_ = false;
  moc_iteration_ = moc_iteration;
  if (moc.sim_mode() == SimulationMode::Keff) {
    mode_ = SimulationMode::Keff;
  } else if (moc.sim_mode() == SimulationMode::FixedSource) {
    mode_ = SimulationMode::FixedSource;
  }

  // Skip user-defined # of MOC iterations
  if (moc_iteration > skip_moc_iterations_) {
    cmfd_solves_++;

    this->normalize_currents();
    this->compute_homogenized_xs_and_flux(moc);
    this->create_loss_matrix(moc);
    this->create_source_matrix();
    if (mode_ == SimulationMode::Keff) {
      this->power_iteration(keff_);
    } else if (mode_ == SimulationMode::FixedSource) {
      this->fixed_source_solve();
    }
    solved_ = true;
    this->update_moc_fluxes(moc);

    if (neutron_balance_check_) {
      for (std::size_t i = 0; i < nx_; i++) {
        for (std::size_t j = 0; j < ny_; j++) {
          for (std::size_t g = 0; g < ng_; g++) {
            this->check_neutron_balance(i, j, g, keff);
          }
        }
      }
    }
  }

  cmfd_timer.stop();
  solve_time_ = cmfd_timer.elapsed_time();
}

}  // namespace scarabee

// REFERENCES
// [1] E. W. Larsen, "Infinite-medium solutions of the transport equation,
// SN discretization schemes, and the diffusion approximation," Transport Theory
// and Statistical Physics, vol. 32, no. 5–7, pp. 623–643, 2003,
//
// [2] A. Zhu et al., "An optimally diffusive Coarse Mesh Finite Difference
// method to accelerate neutron transport calculations," Ann. Nucl. Energy, vol.
// 95, pp. 116–124, 2016, doi: 10.1016/j.anucene.2016.05.004
