#include <diffusion/diffusion_geometry.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>

#include <xtensor/core/xstrides.hpp>

#include <algorithm>
#include <sstream>

namespace scarabee {

DiffusionGeometry::DiffusionGeometry(const std::vector<TileFill>& tiles,
                                     const std::vector<double>& dx,
                                     const std::vector<std::size_t>& xdivs,
                                     double albedo_xn, double albedo_xp)
    : tiles_(),
      xn_(),
      xp_(),
      yn_(),
      yp_(),
      zn_(),
      zp_(),
      tile_dx_(dx),
      x_divs_per_tile_(xdivs),
      tile_dy_(),
      y_divs_per_tile_(),
      tile_dz_(),
      z_divs_per_tile_(),
      x_bounds_(),
      y_bounds_(),
      z_bounds_(),
      nmats_(0),
      mat_indx_to_flat_geom_indx_(),
      nx_(0),
      ny_(0),
      nz_(0),
      geom_shape_() {
  // Make sure numbers are coherent !
  if ((tile_dx_.size() != x_divs_per_tile_.size()) ||
      (tile_dx_.size() != tiles.size())) {
    auto mssg =
        "The number of provided tiles, widths, and divisions does not agree.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (tile_dx_.size() == 0) {
    auto mssg = "No tiles were provided.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure all widths > 0
  for (std::size_t i = 0; i < tile_dx_.size(); i++) {
    if (tile_dx_[i] <= 0.) {
      std::stringstream mssg;
      mssg << "dx at index " << i << " is <= 0.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
  }

  // Make sure all divs are > 0
  nx_ = 0;
  for (std::size_t i = 0; i < x_divs_per_tile_.size(); i++) {
    nx_ += x_divs_per_tile_[i];
    if (x_divs_per_tile_[i] == 0) {
      std::stringstream mssg;
      mssg << "xdivs at index " << i << " is zero.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
  }

  // We can now assign the geometry shape
  geom_shape_ = {nx_};

  // Assign boundary conditions
  if (albedo_xn < 0. || albedo_xn > 1.) {
    std::stringstream mssg;
    mssg << "Negative x albedo is invalid. Must be in range [0, 1].";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }
  xn_.albedo = albedo_xn;
  xn_.xs = nullptr;

  if (albedo_xp < 0. || albedo_xp > 1.) {
    std::stringstream mssg;
    mssg << "Positive x albedo is invalid. Must be in range [0, 1].";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }
  xp_.albedo = albedo_xp;
  xp_.xs = nullptr;

  // Reshape tiles array
  tiles_.resize({tile_dx_.size()});

  // Assign all tiles
  for (std::size_t i = 0; i < tiles.size(); i++) {
    if (std::holds_alternative<double>(tiles[i])) {
      std::stringstream mssg;
      mssg << "A 1D diffusion problem cannot have albedo tiles.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    } else if (std::holds_alternative<std::shared_ptr<DiffusionData>>(
                   tiles[i])) {
      tiles_(i).xs = std::get<std::shared_ptr<DiffusionData>>(tiles[i]);
    } else {
      tiles_(i).xs = std::make_shared<DiffusionData>(
          std::get<std::shared_ptr<DiffusionCrossSection>>(tiles[i]));
    }
  }

  // Since a 1D problem cannot have albedo tiles, then we know all divs are
  // materials. Therefore, the material index m is equivalen to the flat
  // geometry index. This fact greatly simplifies the remaining bits.
  nmats_ = nx_;
  mat_indx_to_flat_geom_indx_.resize(nmats_);
  for (std::size_t m = 0; m < nmats_; m++) mat_indx_to_flat_geom_indx_[m] = m;

  // We must make sure mat_indx_to_flat_geom is sorted, otherwise our method
  // to get mat index will not work !
  if (std::is_sorted(mat_indx_to_flat_geom_indx_.begin(),
                     mat_indx_to_flat_geom_indx_.end()) == false) {
    auto mssg = "Material index to flat geometry index is not sorted.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  fill_x_bounds();
}

DiffusionGeometry::DiffusionGeometry(const std::vector<TileFill>& tiles,
                                     const std::vector<double>& dx,
                                     const std::vector<std::size_t>& xdivs,
                                     const std::vector<double>& dy,
                                     const std::vector<std::size_t>& ydivs,
                                     double albedo_xn, double albedo_xp,
                                     double albedo_yn, double albedo_yp)
    : tiles_(),
      xn_(),
      xp_(),
      yn_(),
      yp_(),
      zn_(),
      zp_(),
      tile_dx_(dx),
      x_divs_per_tile_(xdivs),
      tile_dy_(dy),
      y_divs_per_tile_(ydivs),
      tile_dz_(),
      z_divs_per_tile_(),
      x_bounds_(),
      y_bounds_(),
      z_bounds_(),
      nmats_(0),
      mat_indx_to_flat_geom_indx_(),
      nx_(0),
      ny_(0),
      nz_(0),
      geom_shape_() {
  // Make sure numbers are coherent !
  if ((tile_dx_.size() != x_divs_per_tile_.size()) ||
      (tile_dy_.size() != y_divs_per_tile_.size())) {
    auto mssg = "The number of provided widths and divisions do not agree.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (tiles.size() != tile_dx_.size() * tile_dy_.size()) {
    auto mssg =
        "The number of provided tiles does not agree with the number of "
        "divisions.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (tile_dx_.size() == 0) {
    auto mssg = "No tiles were provided along x.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  if (tile_dy_.size() == 0) {
    auto mssg = "No tiles were provided along y.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure all widths > 0
  for (std::size_t i = 0; i < tile_dx_.size(); i++) {
    if (tile_dx_[i] <= 0.) {
      std::stringstream mssg;
      mssg << "dx at index " << i << " is <= 0.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
  }
  for (std::size_t i = 0; i < tile_dy_.size(); i++) {
    if (tile_dy_[i] <= 0.) {
      std::stringstream mssg;
      mssg << "dy at index " << i << " is <= 0.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
  }

  // Make sure all divs are > 0
  nx_ = 0;
  for (std::size_t i = 0; i < x_divs_per_tile_.size(); i++) {
    nx_ += x_divs_per_tile_[i];
    if (x_divs_per_tile_[i] == 0) {
      std::stringstream mssg;
      mssg << "xdivs at index " << i << " is zero.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
  }

  ny_ = 0;
  for (std::size_t i = 0; i < y_divs_per_tile_.size(); i++) {
    ny_ += y_divs_per_tile_[i];
    if (y_divs_per_tile_[i] == 0) {
      std::stringstream mssg;
      mssg << "ydivs at index " << i << " is zero.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
  }

  // We can now assign the geometry shape
  geom_shape_ = {nx_, ny_};

  // Assign boundary conditions
  if (albedo_xn < 0. || albedo_xn > 1.) {
    std::stringstream mssg;
    mssg << "Negative x albedo is invalid. Must be in range [0, 1].";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }
  xn_.albedo = albedo_xn;
  xn_.xs = nullptr;

  if (albedo_xp < 0. || albedo_xp > 1.) {
    std::stringstream mssg;
    mssg << "Positive x albedo is invalid. Must be in range [0, 1].";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }
  xp_.albedo = albedo_xp;
  xp_.xs = nullptr;

  if (albedo_yn < 0. || albedo_yn > 1.) {
    std::stringstream mssg;
    mssg << "Negative y albedo is invalid. Must be in range [0, 1].";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }
  yn_.albedo = albedo_yn;
  yn_.xs = nullptr;

  if (albedo_yp < 0. || albedo_yp > 1.) {
    std::stringstream mssg;
    mssg << "Positive y albedo is invalid. Must be in range [0, 1].";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }
  yp_.albedo = albedo_yp;
  yp_.xs = nullptr;

  // Reshape tiles array
  tiles_.resize({tile_dx_.size(), tile_dy_.size()});

  // Assign all tiles
  std::size_t tile_indx = 0;
  std::size_t num_expected_mats = 0;
  for (std::size_t j = tile_dy_.size(); j > 0; j--) {
    for (std::size_t i = 0; i < tile_dx_.size(); i++) {
      const auto& tile = tiles[tile_indx];

      if (std::holds_alternative<double>(tile)) {
        tiles_(i, j - 1).albedo = std::get<double>(tile);
      } else if (std::holds_alternative<std::shared_ptr<DiffusionData>>(tile)) {
        tiles_(i, j - 1).xs = std::get<std::shared_ptr<DiffusionData>>(tile);
        num_expected_mats += x_divs_per_tile_[i] * y_divs_per_tile_[j - 1];
      } else {
        tiles_(i, j - 1).xs = std::make_shared<DiffusionData>(
            std::get<std::shared_ptr<DiffusionCrossSection>>(tile));
        num_expected_mats += x_divs_per_tile_[i] * y_divs_per_tile_[j - 1];
      }

      tile_indx++;
    }
  }

  // Assign all material indices and get their geometry index
  mat_indx_to_flat_geom_indx_.reserve(num_expected_mats);
  nmats_ = 0;
  for (std::size_t j = 0; j < ny_; j++) {
    for (std::size_t i = 0; i < nx_; i++) {
      // Create the geometry index
      const xt::svector<std::size_t> geo_inds{i, j};

      // Get the tile index
      const auto tile_inds = geom_to_tile_indx(geo_inds);

      // Grab the tile
      const auto& tile = tiles_.element(tile_inds.begin(), tile_inds.end());

      // Check if we have a material or not
      if (tile.albedo) continue;

      // If we get here, it is a material. We need to compute the flat geometry
      // index.
      std::array<xt::svector<std::size_t>, 1> geo_indx_vec{geo_inds};
      const std::size_t geom_flat_indx = xt::ravel_indices(
          geo_indx_vec, geom_shape_, xt::layout_type::column_major)[0];

      // Save to the mat_indx_to_flat_geom_indx vector
      mat_indx_to_flat_geom_indx_.push_back(geom_flat_indx);
      nmats_++;
    }
  }

  // We must make sure mat_indx_to_flat_geom is sorted, otherwise our method
  // to get mat index will not work !
  if (std::is_sorted(mat_indx_to_flat_geom_indx_.begin(),
                     mat_indx_to_flat_geom_indx_.end()) == false) {
    auto mssg = "Material index to flat geometry index is not sorted.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  fill_x_bounds();
  fill_y_bounds();
}

DiffusionGeometry::DiffusionGeometry(
    const std::vector<TileFill>& tiles, const std::vector<double>& dx,
    const std::vector<std::size_t>& xdivs, const std::vector<double>& dy,
    const std::vector<std::size_t>& ydivs, const std::vector<double>& dz,
    const std::vector<std::size_t>& zdivs, double albedo_xn, double albedo_xp,
    double albedo_yn, double albedo_yp, double albedo_zn, double albedo_zp)
    : tiles_(),
      xn_(),
      xp_(),
      yn_(),
      yp_(),
      zn_(),
      zp_(),
      tile_dx_(dx),
      x_divs_per_tile_(xdivs),
      tile_dy_(dy),
      y_divs_per_tile_(ydivs),
      tile_dz_(dz),
      z_divs_per_tile_(zdivs),
      x_bounds_(),
      y_bounds_(),
      z_bounds_(),
      nmats_(0),
      mat_indx_to_flat_geom_indx_(),
      nx_(0),
      ny_(0),
      nz_(0),
      geom_shape_() {
  // Make sure numbers are coherent !
  if ((tile_dx_.size() != x_divs_per_tile_.size()) ||
      (tile_dy_.size() != y_divs_per_tile_.size()) ||
      (tile_dz_.size() != z_divs_per_tile_.size())) {
    auto mssg = "The number of provided widths and divisions do not agree.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (tiles.size() != tile_dx_.size() * tile_dy_.size() * tile_dz_.size()) {
    auto mssg =
        "The number of provided tiles does not agree with the number of "
        "divisions.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (tile_dx_.size() == 0) {
    auto mssg = "No tiles were provided along x.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  if (tile_dy_.size() == 0) {
    auto mssg = "No tiles were provided along y.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  if (tile_dz_.size() == 0) {
    auto mssg = "No tiles were provided along z.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure all widths > 0
  for (std::size_t i = 0; i < tile_dx_.size(); i++) {
    if (tile_dx_[i] <= 0.) {
      std::stringstream mssg;
      mssg << "dx at index " << i << " is <= 0.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
  }
  for (std::size_t i = 0; i < tile_dy_.size(); i++) {
    if (tile_dy_[i] <= 0.) {
      std::stringstream mssg;
      mssg << "dy at index " << i << " is <= 0.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
  }
  for (std::size_t i = 0; i < tile_dz_.size(); i++) {
    if (tile_dz_[i] <= 0.) {
      std::stringstream mssg;
      mssg << "dz at index " << i << " is <= 0.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
  }

  // Make sure all divs are > 0
  nx_ = 0;
  for (std::size_t i = 0; i < x_divs_per_tile_.size(); i++) {
    nx_ += x_divs_per_tile_[i];
    if (x_divs_per_tile_[i] == 0) {
      std::stringstream mssg;
      mssg << "xdivs at index " << i << " is zero.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
  }

  ny_ = 0;
  for (std::size_t i = 0; i < y_divs_per_tile_.size(); i++) {
    ny_ += y_divs_per_tile_[i];
    if (y_divs_per_tile_[i] == 0) {
      std::stringstream mssg;
      mssg << "ydivs at index " << i << " is zero.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
  }

  nz_ = 0;
  for (std::size_t i = 0; i < z_divs_per_tile_.size(); i++) {
    nz_ += z_divs_per_tile_[i];
    if (z_divs_per_tile_[i] == 0) {
      std::stringstream mssg;
      mssg << "zdivs at index " << i << " is zero.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }
  }

  // We can now assign the geometry shape
  geom_shape_ = {nx_, ny_, nz_};

  // Assign boundary conditions
  if (albedo_xn < 0. || albedo_xn > 1.) {
    std::stringstream mssg;
    mssg << "Negative x albedo is invalid. Must be in range [0, 1].";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }
  xn_.albedo = albedo_xn;
  xn_.xs = nullptr;

  if (albedo_xp < 0. || albedo_xp > 1.) {
    std::stringstream mssg;
    mssg << "Positive x albedo is invalid. Must be in range [0, 1].";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }
  xp_.albedo = albedo_xp;
  xp_.xs = nullptr;

  if (albedo_yn < 0. || albedo_yn > 1.) {
    std::stringstream mssg;
    mssg << "Negative y albedo is invalid. Must be in range [0, 1].";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }
  yn_.albedo = albedo_yn;
  yn_.xs = nullptr;

  if (albedo_yp < 0. || albedo_yp > 1.) {
    std::stringstream mssg;
    mssg << "Positive y albedo is invalid. Must be in range [0, 1].";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }
  yp_.albedo = albedo_yp;
  yp_.xs = nullptr;

  if (albedo_zn < 0. || albedo_zn > 1.) {
    std::stringstream mssg;
    mssg << "Negative z albedo is invalid. Must be in range [0, 1].";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }
  zn_.albedo = albedo_zn;
  zn_.xs = nullptr;

  if (albedo_zp < 0. || albedo_zp > 1.) {
    std::stringstream mssg;
    mssg << "Positive z albedo is invalid. Must be in range [0, 1].";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }
  zp_.albedo = albedo_zp;
  zp_.xs = nullptr;

  // Reshape tiles array
  tiles_.resize({tile_dx_.size(), tile_dy_.size(), tile_dy_.size()});

  // Assign all tiles
  std::size_t tile_indx = 0;
  std::size_t num_expected_mats = 0;
  for (std::size_t k = tile_dz_.size(); k > 0; k--) {
    for (std::size_t j = tile_dy_.size(); j > 0; j--) {
      for (std::size_t i = 0; i < tile_dx_.size(); i++) {
        const auto& tile = tiles[tile_indx];

        if (std::holds_alternative<double>(tile)) {
          tiles_(i, j - 1, k - 1).albedo = std::get<double>(tile);
        } else if (std::holds_alternative<std::shared_ptr<DiffusionData>>(
                       tile)) {
          tiles_(i, j - 1, k - 1).xs =
              std::get<std::shared_ptr<DiffusionData>>(tile);
          num_expected_mats += x_divs_per_tile_[i] * y_divs_per_tile_[j - 1] *
                               z_divs_per_tile_[k - 1];
        } else {
          tiles_(i, j - 1, k - 1).xs = std::make_shared<DiffusionData>(
              std::get<std::shared_ptr<DiffusionCrossSection>>(tile));
          num_expected_mats += x_divs_per_tile_[i] * y_divs_per_tile_[j - 1] *
                               z_divs_per_tile_[k - 1];
        }

        tile_indx++;
      }
    }
  }

  // Assign all material indices and get their geometry index
  mat_indx_to_flat_geom_indx_.reserve(num_expected_mats);
  nmats_ = 0;
  for (std::size_t k = 0; k < nz_; k++) {
    for (std::size_t j = 0; j < ny_; j++) {
      for (std::size_t i = 0; i < nx_; i++) {
        // Create the geometry index
        const xt::svector<std::size_t> geo_inds{i, j, k};

        // Get the tile index
        const auto tile_inds = geom_to_tile_indx(geo_inds);

        // Grab the tile
        const auto& tile = tiles_.element(tile_inds.begin(), tile_inds.end());

        // Check if we have a material or not
        if (tile.albedo) continue;

        // If we get here, it is a material. We need to compute the flat
        // geometry index.
        std::array<xt::svector<std::size_t>, 1> geo_indx_vec{geo_inds};
        const std::size_t geom_flat_indx = xt::ravel_indices(
            geo_indx_vec, geom_shape_, xt::layout_type::column_major)[0];

        // Save to the mat_indx_to_flat_geom_indx vector
        mat_indx_to_flat_geom_indx_.push_back(geom_flat_indx);
        nmats_++;
      }
    }
  }

  // We must make sure mat_indx_to_flat_geom is sorted, otherwise our method
  // to get mat index will not work !
  if (std::is_sorted(mat_indx_to_flat_geom_indx_.begin(),
                     mat_indx_to_flat_geom_indx_.end()) == false) {
    auto mssg = "Material index to flat geometry index is not sorted.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  fill_x_bounds();
  fill_y_bounds();
  fill_z_bounds();
}

std::size_t DiffusionGeometry::ngroups() const {
  return this->mat(0)->ngroups();
}

std::pair<DiffusionGeometry::Tile, std::optional<std::size_t>>
DiffusionGeometry::neighbor(std::size_t m, Neighbor n) const {
  if (m >= nmats()) {
    auto mssg = "Material index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (ndims() == 1) {
    return neighbor_1d(m, n);
  } else if (ndims() == 2) {
    return neighbor_2d(m, n);
  } else {
    return neighbor_3d(m, n);
  }
}

std::pair<DiffusionGeometry::Tile, std::optional<std::size_t>>
DiffusionGeometry::neighbor_1d(std::size_t m, Neighbor n) const {
  if (n != Neighbor::XN && n != Neighbor::XP) {
    auto mssg = "Invalid neighbor requested for 1D geometry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // First, we get the array material index
  auto geo_indx = this->geom_indx(m);

  // First, check for and edge case
  if (geo_indx[0] == 0 && n == Neighbor::XN)
    return {xn_, std::nullopt};
  else if (geo_indx[0] == nx() - 1 && n == Neighbor::XP)
    return {xp_, std::nullopt};

  // Next, change mat array to desired neighbor
  if (n == Neighbor::XN)
    geo_indx[0]--;
  else
    geo_indx[0]++;

  // Now, we get the respective tile index and tile
  xt::svector<std::size_t> tile_indx = this->geom_to_tile_indx(geo_indx);
  const auto& tile = tiles_.element(tile_indx.begin(), tile_indx.end());

  if (tile.xs == nullptr) return {tile, std::nullopt};

  // Our tile is a material. We need to get the material index.
  std::size_t mn = this->geom_to_mat_indx(geo_indx).value();

  return {tile, mn};
}

std::pair<DiffusionGeometry::Tile, std::optional<std::size_t>>
DiffusionGeometry::neighbor_2d(std::size_t m, Neighbor n) const {
  if (n == Neighbor::ZN || n == Neighbor::ZP) {
    auto mssg = "Invalid neighbor requested for 2D geometry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // First, we get the array material index
  auto geo_indx = this->geom_indx(m);

  // First, check for and edge case
  if (geo_indx[0] == 0 && n == Neighbor::XN)
    return {xn_, std::nullopt};
  else if (geo_indx[0] == nx() - 1 && n == Neighbor::XP)
    return {xp_, std::nullopt};
  else if (geo_indx[1] == 0 && n == Neighbor::YN)
    return {yn_, std::nullopt};
  else if (geo_indx[1] == ny() - 1 && n == Neighbor::YP)
    return {yp_, std::nullopt};

  // Next, change mat array to desired neighbor
  if (n == Neighbor::XN)
    geo_indx[0]--;
  else if (n == Neighbor::XP)
    geo_indx[0]++;
  else if (n == Neighbor::YN)
    geo_indx[1]--;
  else
    geo_indx[1]++;

  // Now, we get the respective tile index and tile
  xt::svector<std::size_t> tile_indx = this->geom_to_tile_indx(geo_indx);
  const auto& tile = tiles_.element(tile_indx.begin(), tile_indx.end());

  if (tile.xs == nullptr) return {tile, std::nullopt};

  // Our tile is a material. We need to get the material index.
  std::size_t mn = this->geom_to_mat_indx(geo_indx).value();

  return {tile, mn};
}

std::pair<DiffusionGeometry::Tile, std::optional<std::size_t>>
DiffusionGeometry::neighbor_3d(std::size_t m, Neighbor n) const {
  // First, we get the array material index
  auto geo_indx = this->geom_indx(m);

  // First, check for and edge case
  if (geo_indx[0] == 0 && n == Neighbor::XN)
    return {xn_, std::nullopt};
  else if (geo_indx[0] == nx() - 1 && n == Neighbor::XP)
    return {xp_, std::nullopt};
  else if (geo_indx[1] == 0 && n == Neighbor::YN)
    return {yn_, std::nullopt};
  else if (geo_indx[1] == ny() - 1 && n == Neighbor::YP)
    return {yp_, std::nullopt};
  else if (geo_indx[2] == 0 && n == Neighbor::ZN)
    return {zn_, std::nullopt};
  else if (geo_indx[2] == nz() - 1 && n == Neighbor::ZP)
    return {zp_, std::nullopt};

  // Next, change mat array to desired neighbor
  if (n == Neighbor::XN)
    geo_indx[0]--;
  else if (n == Neighbor::XP)
    geo_indx[0]++;
  else if (n == Neighbor::YN)
    geo_indx[1]--;
  else if (n == Neighbor::YP)
    geo_indx[1]++;
  else if (n == Neighbor::ZN)
    geo_indx[2]--;
  else if (n == Neighbor::ZP)
    geo_indx[2]++;

  // Now, we get the respective tile index and tile
  xt::svector<std::size_t> tile_indx = this->geom_to_tile_indx(geo_indx);
  const auto& tile = tiles_.element(tile_indx.begin(), tile_indx.end());

  if (tile.xs == nullptr) return {tile, std::nullopt};

  // Our tile is a material. We need to get the material index.
  std::size_t mn = this->geom_to_mat_indx(geo_indx).value();

  return {tile, mn};
}

const std::shared_ptr<DiffusionData>& DiffusionGeometry::mat(
    std::size_t m) const {
  if (m >= nmats()) {
    auto mssg = "Material index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  const auto geo_indx = geom_indx(m);
  const auto tile_indx = geom_to_tile_indx(geo_indx);
  return tiles_.element(tile_indx.begin(), tile_indx.end()).xs;
}

const std::shared_ptr<DiffusionData>& DiffusionGeometry::mat(
    const xt::svector<std::size_t>& geo_indx) const {
  const auto tile_indx = geom_to_tile_indx(geo_indx);
  return tiles_.element(tile_indx.begin(), tile_indx.end()).xs;
}

xt::svector<std::size_t> DiffusionGeometry::geom_indx(std::size_t m) const {
  if (m >= nmats()) {
    auto mssg = "Material index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  std::size_t flat_geom_indx = mat_indx_to_flat_geom_indx_[m];
  std::array<std::size_t, 1> flat_geom_vec{flat_geom_indx};

  // Now get the complete geometry index
  auto geom_inds = xt::unravel_indices(flat_geom_vec, geom_shape_,
                                       xt::layout_type::column_major)[0];
  return xt::svector<std::size_t>(geom_inds.begin(), geom_inds.end());
}

xt::svector<std::size_t> DiffusionGeometry::geom_to_tile_indx(
    const xt::svector<std::size_t>& geo_indx) const {
  const std::size_t i = geom_x_indx_to_tile_x_indx(geo_indx[0]);
  if (ndims() == 1) return {i};

  const std::size_t j = geom_y_indx_to_tile_y_indx(geo_indx[1]);
  if (ndims() == 2) return {i, j};

  const std::size_t k = geom_z_indx_to_tile_z_indx(geo_indx[2]);
  return {i, j, k};
}

std::optional<std::size_t> DiffusionGeometry::geom_to_mat_indx(
    const xt::svector<std::size_t>& geo_indx) const {
  // Get the flat geometry index that we can search for
  std::array<xt::svector<std::size_t>, 1> geo_indx_vec{geo_indx};
  const std::size_t geom_flat_indx = xt::ravel_indices(
      geo_indx_vec, geom_shape_, xt::layout_type::column_major)[0];

  const auto it =
      std::lower_bound(mat_indx_to_flat_geom_indx_.begin(),
                       mat_indx_to_flat_geom_indx_.end(), geom_flat_indx);

  if (it == mat_indx_to_flat_geom_indx_.end() || *it != geom_flat_indx)
    return std::nullopt;

  return std::distance(mat_indx_to_flat_geom_indx_.begin(), it);
}

double DiffusionGeometry::adf_xp(std::size_t m, std::size_t g) const {
  // Get geometry index for m
  const auto geo_indx_m = geom_indx(m);
  if (geo_indx_m[0] == nx() - 1) {
    return mat(m)->adf_xp(g);
  }

  auto geo_indx_n = geo_indx_m;
  geo_indx_n[0]++;
  const auto n = geom_to_mat_indx(geo_indx_n);

  if (n.has_value() == false) {
    // There is no neighbor on that side in the geometry.
    // Use the current tile ADF.
    return mat(m)->adf_xp(g);
  }

  // The the x-index of the tile for the current m and the neighbor mat.
  const std::size_t tile_indx_m = geom_x_indx_to_tile_x_indx(geo_indx_m[0]);
  const std::size_t tile_indx_n = geom_x_indx_to_tile_x_indx(geo_indx_n[0]);

  // If the current m and neighbor n are in the same tile, we don't have a
  // discontinuity, so the ADF is 1.
  if (tile_indx_m == tile_indx_n) return 1.;

  // Check if we need to correct the reflector DF [1]
  if (mat(m)->reflector() && mat(*n)->reflector() == false) {
    return mat(m)->adf_xp(g) * mat(*n)->adf_xn(g);
  } else if (mat(m)->reflector() && mat(*n)->reflector()) {
    // At the interface of 2 reflector nodes, we only use and ADF of 1
    return 1.;
  }

  // We are at the boarder of a tile, so we return the mat ADF.
  return mat(m)->adf_xp(g);
}

double DiffusionGeometry::adf_xn(std::size_t m, std::size_t g) const {
  // Get geometry index for m
  const auto geo_indx_m = geom_indx(m);
  if (geo_indx_m[0] == 0) {
    return mat(m)->adf_xn(g);
  }

  auto geo_indx_n = geo_indx_m;
  geo_indx_n[0]--;
  const auto n = geom_to_mat_indx(geo_indx_n);

  if (n.has_value() == false) {
    // There is no neighbor on that side in the geometry.
    // Use the current tile ADF.
    return mat(m)->adf_xn(g);
  }

  // The the x-index of the tile for the current m and the neighbor mat.
  const std::size_t tile_indx_m = geom_x_indx_to_tile_x_indx(geo_indx_m[0]);
  const std::size_t tile_indx_n = geom_x_indx_to_tile_x_indx(geo_indx_n[0]);

  // If the current m and neighbor n are in the same tile, we don't have a
  // discontinuity, so the ADF is 1.
  if (tile_indx_m == tile_indx_n) return 1.;

  // Check if we need to correct the reflector DF [1]
  if (mat(m)->reflector() && mat(*n)->reflector() == false) {
    return mat(m)->adf_xn(g) * mat(*n)->adf_xp(g);
  } else if (mat(m)->reflector() && mat(*n)->reflector()) {
    // At the interface of 2 reflector nodes, we only use and ADF of 1
    return 1.;
  }

  // We are at the boarder of a tile, so we return the mat ADF.
  return mat(m)->adf_xn(g);
}

double DiffusionGeometry::adf_yp(std::size_t m, std::size_t g) const {
  // Get geometry index for m
  const auto geo_indx_m = geom_indx(m);
  if (geo_indx_m[1] == ny() - 1) {
    return mat(m)->adf_yp(g);
  }

  auto geo_indx_n = geo_indx_m;
  geo_indx_n[1]++;
  const auto n = geom_to_mat_indx(geo_indx_n);

  if (n.has_value() == false) {
    // There is no neighbor on that side in the geometry.
    // Use the current tile ADF.
    return mat(m)->adf_yp(g);
  }

  // The the y-index of the tile for the current m and the neighbor mat.
  const std::size_t tile_indx_m = geom_y_indx_to_tile_y_indx(geo_indx_m[1]);
  const std::size_t tile_indx_n = geom_y_indx_to_tile_y_indx(geo_indx_n[1]);

  // If the current m and neighbor n are in the same tile, we don't have a
  // discontinuity, so the ADF is 1.
  if (tile_indx_m == tile_indx_n) return 1.;

  // Check if we need to correct the reflector DF [1]
  if (mat(m)->reflector() && mat(*n)->reflector() == false) {
    return mat(m)->adf_yp(g) * mat(*n)->adf_yn(g);
  } else if (mat(m)->reflector() && mat(*n)->reflector()) {
    // At the interface of 2 reflector nodes, we only use and ADF of 1
    return 1.;
  }

  // We are at the boarder of a tile, so we return the mat ADF.
  return mat(m)->adf_yp(g);
}

double DiffusionGeometry::adf_yn(std::size_t m, std::size_t g) const {
  // Get geometry index for m
  const auto geo_indx_m = geom_indx(m);
  if (geo_indx_m[1] == 0) {
    return mat(m)->adf_yn(g);
  }

  auto geo_indx_n = geo_indx_m;
  geo_indx_n[1]--;
  const auto n = geom_to_mat_indx(geo_indx_n);

  if (n.has_value() == false) {
    // There is no neighbor on that side in the geometry.
    // Use the current tile ADF.
    return mat(m)->adf_yn(g);
  }

  // The the y-index of the tile for the current m and the neighbor mat.
  const std::size_t tile_indx_m = geom_y_indx_to_tile_y_indx(geo_indx_m[1]);
  const std::size_t tile_indx_n = geom_y_indx_to_tile_y_indx(geo_indx_n[1]);

  // If the current m and neighbor n are in the same tile, we don't have a
  // discontinuity, so the ADF is 1.
  if (tile_indx_m == tile_indx_n) return 1.;

  // Check if we need to correct the reflector DF [1]
  if (mat(m)->reflector() && mat(*n)->reflector() == false) {
    return mat(m)->adf_yn(g) * mat(*n)->adf_yp(g);
  } else if (mat(m)->reflector() && mat(*n)->reflector()) {
    // At the interface of 2 reflector nodes, we only use and ADF of 1
    return 1.;
  }

  // We are at the boarder of a tile, so we return the mat ADF.
  return mat(m)->adf_yn(g);
}

double DiffusionGeometry::adf_zp(std::size_t m, std::size_t g) const {
  // Get geometry index for m
  const auto geo_indx_m = geom_indx(m);
  if (geo_indx_m[2] == nz() - 1) {
    return mat(m)->adf_zp(g);
  }

  auto geo_indx_n = geo_indx_m;
  geo_indx_n[2]++;
  const auto n = geom_to_mat_indx(geo_indx_n);

  if (n.has_value() == false) {
    // There is no neighbor on that side in the geometry.
    // Use the current tile ADF.
    return mat(m)->adf_zp(g);
  }

  // The the z-index of the tile for the current m and the neighbor mat.
  const std::size_t tile_indx_m = geom_z_indx_to_tile_z_indx(geo_indx_m[2]);
  const std::size_t tile_indx_n = geom_z_indx_to_tile_z_indx(geo_indx_n[2]);

  // If the current m and neighbor n are in the same tile, we don't have a
  // discontinuity, so the ADF is 1.
  if (tile_indx_m == tile_indx_n) return 1.;

  // Check if we need to correct the reflector DF [1]
  if (mat(m)->reflector() && mat(*n)->reflector() == false) {
    return mat(m)->adf_zp(g) * mat(*n)->adf_zn(g);
  } else if (mat(m)->reflector() && mat(*n)->reflector()) {
    // At the interface of 2 reflector nodes, we only use and ADF of 1
    return 1.;
  }

  // We are at the boarder of a tile, so we return the mat ADF.
  return mat(m)->adf_zp(g);
}

double DiffusionGeometry::adf_zn(std::size_t m, std::size_t g) const {
  // Get geometry index for m
  const auto geo_indx_m = geom_indx(m);
  if (geo_indx_m[2] == 0) {
    return mat(m)->adf_zn(g);
  }

  auto geo_indx_n = geo_indx_m;
  geo_indx_n[2]--;
  const auto n = geom_to_mat_indx(geo_indx_n);

  if (n.has_value() == false) {
    // There is no neighbor on that side in the geometry.
    // Use the current tile ADF.
    return mat(m)->adf_zn(g);
  }

  // The the z-index of the tile for the current m and the neighbor mat.
  const std::size_t tile_indx_m = geom_z_indx_to_tile_z_indx(geo_indx_m[2]);
  const std::size_t tile_indx_n = geom_z_indx_to_tile_z_indx(geo_indx_n[2]);

  // If the current m and neighbor n are in the same tile, we don't have a
  // discontinuity, so the ADF is 1.
  if (tile_indx_m == tile_indx_n) return 1.;

  // Check if we need to correct the reflector DF [1]
  if (mat(m)->reflector() && mat(*n)->reflector() == false) {
    return mat(m)->adf_zn(g) * mat(*n)->adf_zp(g);
  } else if (mat(m)->reflector() && mat(*n)->reflector()) {
    // At the interface of 2 reflector nodes, we only use and ADF of 1
    return 1.;
  }

  // We are at the boarder of a tile, so we return the mat ADF.
  return mat(m)->adf_zn(g);
}

double DiffusionGeometry::cdf_I(std::size_t m, std::size_t g) const {
  // Get geometry index for m
  const auto geo_indx_m = geom_indx(m);

  if (geo_indx_m[0] == nx() - 1 || geo_indx_m[1] == ny() - 1) {
    return mat(m)->cdf_I(g);
  }

  auto geo_indx_n = geo_indx_m;
  geo_indx_n[0]++;
  geo_indx_n[1]++;
  const auto n = geom_to_mat_indx(geo_indx_n);

  if (n.has_value() == false) {
    // There is no neighbor on that side in the geometry.
    // Use the current tile CDF.
    return mat(m)->cdf_I(g);
  }

  // The the x-index of the tile for the current m and the neighbor mat.
  const std::size_t tile_x_indx_m = geom_x_indx_to_tile_x_indx(geo_indx_m[0]);
  const std::size_t tile_x_indx_n = geom_x_indx_to_tile_x_indx(geo_indx_n[0]);

  // The the y-index of the tile for the current m and the neighbor mat.
  const std::size_t tile_y_indx_m = geom_y_indx_to_tile_y_indx(geo_indx_m[1]);
  const std::size_t tile_y_indx_n = geom_y_indx_to_tile_y_indx(geo_indx_n[1]);

  // If the current m and neighbor n are in the same tile, we don't have a
  // discontinuity, so the CDF is 1.
  if (tile_x_indx_m == tile_x_indx_n && tile_y_indx_m == tile_y_indx_n) {
    return 1.;
  } else if (tile_x_indx_m == tile_x_indx_n) {
    return mat(m)->adf_yp(g);
  } else if (tile_y_indx_m == tile_y_indx_n) {
    return mat(m)->adf_xp(g);
  }

  // We are at the boarder of a tile, so we return the mat CDF.
  return mat(m)->cdf_I(g);
}

double DiffusionGeometry::cdf_II(std::size_t m, std::size_t g) const {
  // Get geometry index for m
  const auto geo_indx_m = geom_indx(m);

  if (geo_indx_m[0] == 0 || geo_indx_m[1] == ny() - 1) {
    return mat(m)->cdf_II(g);
  }

  auto geo_indx_n = geo_indx_m;
  geo_indx_n[0]--;
  geo_indx_n[1]++;
  const auto n = geom_to_mat_indx(geo_indx_n);

  if (n.has_value() == false) {
    // There is no neighbor on that side in the geometry.
    // Use the current tile CDF.
    return mat(m)->cdf_II(g);
  }

  // The the x-index of the tile for the current m and the neighbor mat.
  const std::size_t tile_x_indx_m = geom_x_indx_to_tile_x_indx(geo_indx_m[0]);
  const std::size_t tile_x_indx_n = geom_x_indx_to_tile_x_indx(geo_indx_n[0]);

  // The the y-index of the tile for the current m and the neighbor mat.
  const std::size_t tile_y_indx_m = geom_y_indx_to_tile_y_indx(geo_indx_m[1]);
  const std::size_t tile_y_indx_n = geom_y_indx_to_tile_y_indx(geo_indx_n[1]);

  // If the current m and neighbor n are in the same tile, we don't have a
  // discontinuity, so the CDF is 1.
  if (tile_x_indx_m == tile_x_indx_n && tile_y_indx_m == tile_y_indx_n) {
    return 1.;
  } else if (tile_x_indx_m == tile_x_indx_n) {
    return mat(m)->adf_yp(g);
  } else if (tile_y_indx_m == tile_y_indx_n) {
    return mat(m)->adf_xn(g);
  }

  // We are at the boarder of a tile, so we return the mat CDF.
  return mat(m)->cdf_II(g);
}

double DiffusionGeometry::cdf_III(std::size_t m, std::size_t g) const {
  // Get geometry index for m
  const auto geo_indx_m = geom_indx(m);

  if (geo_indx_m[0] == 0 || geo_indx_m[1] == 0) {
    return mat(m)->cdf_III(g);
  }

  auto geo_indx_n = geo_indx_m;
  geo_indx_n[0]--;
  geo_indx_n[1]--;
  const auto n = geom_to_mat_indx(geo_indx_n);

  if (n.has_value() == false) {
    // There is no neighbor on that side in the geometry.
    // Use the current tile CDF.
    return mat(m)->cdf_III(g);
  }

  // The the x-index of the tile for the current m and the neighbor mat.
  const std::size_t tile_x_indx_m = geom_x_indx_to_tile_x_indx(geo_indx_m[0]);
  const std::size_t tile_x_indx_n = geom_x_indx_to_tile_x_indx(geo_indx_n[0]);

  // The the y-index of the tile for the current m and the neighbor mat.
  const std::size_t tile_y_indx_m = geom_y_indx_to_tile_y_indx(geo_indx_m[1]);
  const std::size_t tile_y_indx_n = geom_y_indx_to_tile_y_indx(geo_indx_n[1]);

  // If the current m and neighbor n are in the same tile, we don't have a
  // discontinuity, so the CDF is 1.
  if (tile_x_indx_m == tile_x_indx_n && tile_y_indx_m == tile_y_indx_n) {
    return 1.;
  } else if (tile_x_indx_m == tile_x_indx_n) {
    return mat(m)->adf_yn(g);
  } else if (tile_y_indx_m == tile_y_indx_n) {
    return mat(m)->adf_xn(g);
  }

  // We are at the boarder of a tile, so we return the mat CDF.
  return mat(m)->cdf_III(g);
}

double DiffusionGeometry::cdf_IV(std::size_t m, std::size_t g) const {
  // Get geometry index for m
  const auto geo_indx_m = geom_indx(m);

  if (geo_indx_m[0] == nx() - 1 || geo_indx_m[1] == 0) {
    return mat(m)->cdf_IV(g);
  }

  auto geo_indx_n = geo_indx_m;
  geo_indx_n[0]++;
  geo_indx_n[1]--;
  const auto n = geom_to_mat_indx(geo_indx_n);

  if (n.has_value() == false) {
    // There is no neighbor on that side in the geometry.
    // Use the current tile CDF.
    return mat(m)->cdf_IV(g);
  }

  // The the x-index of the tile for the current m and the neighbor mat.
  const std::size_t tile_x_indx_m = geom_x_indx_to_tile_x_indx(geo_indx_m[0]);
  const std::size_t tile_x_indx_n = geom_x_indx_to_tile_x_indx(geo_indx_n[0]);

  // The the y-index of the tile for the current m and the neighbor mat.
  const std::size_t tile_y_indx_m = geom_y_indx_to_tile_y_indx(geo_indx_m[1]);
  const std::size_t tile_y_indx_n = geom_y_indx_to_tile_y_indx(geo_indx_n[1]);

  // If the current m and neighbor n are in the same tile, we don't have a
  // discontinuity, so the CDF is 1.
  if (tile_x_indx_m == tile_x_indx_n && tile_y_indx_m == tile_y_indx_n) {
    return 1.;
  } else if (tile_x_indx_m == tile_x_indx_n) {
    return mat(m)->adf_yn(g);
  } else if (tile_y_indx_m == tile_y_indx_n) {
    return mat(m)->adf_xp(g);
  }

  // We are at the boarder of a tile, so we return the mat CDF.
  return mat(m)->cdf_IV(g);
}

double DiffusionGeometry::volume(std::size_t m) const {
  if (m >= nmats()) {
    auto mssg = "Material index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  auto indxs = this->geom_indx(m);

  if (ndims() == 1) {
    return dx(indxs[0]);
  } else if (ndims() == 2) {
    return dx(indxs[0]) * dy(indxs[1]);
  } else {
    return dx(indxs[0]) * dy(indxs[1]) * dz(indxs[2]);
  }
}

std::optional<std::size_t> DiffusionGeometry::x_to_i(double x) const {
  if (x < x_bounds_.front() || x > x_bounds_.back()) return std::nullopt;

  for (std::size_t i = 0; i < x_bounds_.size() - 1; i++) {
    if (x_bounds_[i] <= x && x <= x_bounds_[i + 1]) return i;
  }

  // SHOULD NEVER GET HERE
  return x_bounds_.size() - 1;
}

std::optional<std::size_t> DiffusionGeometry::y_to_j(double y) const {
  if (y < y_bounds_.front() || y > y_bounds_.back()) return std::nullopt;

  for (std::size_t j = 0; j < y_bounds_.size() - 1; j++) {
    if (y_bounds_[j] <= y && y <= y_bounds_[j + 1]) return j;
  }

  // SHOULD NEVER GET HERE
  return y_bounds_.size() - 1;
}

std::optional<std::size_t> DiffusionGeometry::z_to_k(double z) const {
  if (z < z_bounds_.front() || z > z_bounds_.back()) return std::nullopt;

  for (std::size_t k = 0; k < z_bounds_.size() - 1; k++) {
    if (z_bounds_[k] <= z && z <= z_bounds_[k + 1]) return k;
  }

  // SHOULD NEVER GET HERE
  return z_bounds_.size() - 1;
}

std::size_t DiffusionGeometry::geom_x_indx_to_tile_x_indx(std::size_t i) const {
  if (i >= nx()) {
    auto mssg = "Index along x is out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  std::size_t i_tile = 0;
  std::size_t l = 0;
  while (l < i) {
    l += x_divs_per_tile_[i_tile];

    if (l > i) break;

    i_tile++;
  }

  return i_tile;
}

std::size_t DiffusionGeometry::geom_y_indx_to_tile_y_indx(std::size_t i) const {
  if (i >= ny()) {
    auto mssg = "Index along y is out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  std::size_t i_tile = 0;
  std::size_t l = 0;
  while (l < i) {
    l += y_divs_per_tile_[i_tile];

    if (l > i) break;

    i_tile++;
  }

  return i_tile;
}

std::size_t DiffusionGeometry::geom_z_indx_to_tile_z_indx(std::size_t i) const {
  if (i >= nz()) {
    auto mssg = "Index along z is out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  std::size_t i_tile = 0;
  std::size_t l = 0;
  while (l < i) {
    l += z_divs_per_tile_[i_tile];

    if (l > i) break;

    i_tile++;
  }

  return i_tile;
}

void DiffusionGeometry::fill_x_bounds() {
  // Count the total number of x regions and allocate array
  std::size_t nx =
      std::accumulate(x_divs_per_tile_.begin(), x_divs_per_tile_.end(), 0);
  x_bounds_.reserve(nx + 1);

  // Add initial 0
  x_bounds_.push_back(0.);

  for (std::size_t t = 0; t < tile_dx_.size(); t++) {
    const std::size_t nxt = x_divs_per_tile_[t];
    const double dx = tile_dx_[t] / static_cast<double>(nxt);

    for (std::size_t n = 0; n < nxt; n++) {
      x_bounds_.push_back(x_bounds_.back() + dx);
    }
  }

  if (x_bounds_.size() != nx + 1) {
    throw ScarabeeException("HELP BAD X BOUNDS !");
  }
}

void DiffusionGeometry::fill_y_bounds() {
  // Count the total number of y regions and allocate array
  std::size_t ny =
      std::accumulate(y_divs_per_tile_.begin(), y_divs_per_tile_.end(), 0);
  y_bounds_.reserve(ny + 1);

  // Add initial 0
  y_bounds_.push_back(0.);

  for (std::size_t t = 0; t < tile_dy_.size(); t++) {
    const std::size_t nyt = y_divs_per_tile_[t];
    const double dy = tile_dy_[t] / static_cast<double>(nyt);

    for (std::size_t n = 0; n < nyt; n++) {
      y_bounds_.push_back(y_bounds_.back() + dy);
    }
  }

  if (y_bounds_.size() != ny + 1) {
    throw ScarabeeException("HELP BAD Y BOUNDS !");
  }
}

void DiffusionGeometry::fill_z_bounds() {
  // Count the total number of z regions and allocate array
  std::size_t nz =
      std::accumulate(z_divs_per_tile_.begin(), z_divs_per_tile_.end(), 0);
  z_bounds_.reserve(nz + 1);

  // Add initial 0
  z_bounds_.push_back(0.);

  for (std::size_t t = 0; t < tile_dz_.size(); t++) {
    const std::size_t nzt = z_divs_per_tile_[t];
    const double dz = tile_dz_[t] / static_cast<double>(nzt);

    for (std::size_t n = 0; n < nzt; n++) {
      z_bounds_.push_back(z_bounds_.back() + dz);
    }
  }

  if (z_bounds_.size() != nz + 1) {
    throw ScarabeeException("HELP BAD Z BOUNDS !");
  }
}

}  // namespace scarabee

// [1] K. S. Smith, “Nodal diffusion methods and lattice physics data in LWR
//     analyses: Understanding numerous subtle details,” Prog Nucl Energ,
//     vol. 101, pp. 360–369, 2017, doi: 10.1016/j.pnucene.2017.06.013.
