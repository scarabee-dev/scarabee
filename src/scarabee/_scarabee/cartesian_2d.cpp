#include <moc/cartesian_2d.hpp>
#include <utils/scarabee_exception.hpp>
#include <utils/logging.hpp>
#include <utils/constants.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <vector>

namespace scarabee {

Cartesian2D::Cartesian2D(const std::vector<std::shared_ptr<Surface>>& x_bounds,
                         const std::vector<std::shared_ptr<Surface>>& y_bounds)
    : x_bounds_(x_bounds),
      y_bounds_(y_bounds),
      tiles_(),
      fsr_offset_map_(),
      nx_(),
      ny_() {
  // First, make sure we have at least 2 bounds in each direction
  if (x_bounds_.size() < 2) {
    auto mssg = "Must provide at least 2 x-bounds.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  } else if (y_bounds_.size() < 2) {
    auto mssg = "Must provide at least 2 y-bounds.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure bounds are sorted
  if (std::is_sorted(x_bounds_.begin(), x_bounds_.end(),
                     [](const auto& a, const auto& b) {
                       return a->x0() < b->x0();
                     }) == false) {
    auto mssg = "Provided x-bounds are not sorted.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (std::is_sorted(y_bounds_.begin(), y_bounds_.end(),
                     [](const auto& a, const auto& b) {
                       return a->y0() < b->y0();
                     }) == false) {
    auto mssg = "Provided y-bounds are not sorted.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Get sizes and allocate tiles array
  nx_ = x_bounds_.size() - 1;
  ny_ = y_bounds_.size() - 1;

  // All tiles start as uninitialized
  tiles_.resize({nx_, ny_});
  tiles_.fill(Tile{nullptr, nullptr});

  fsr_offset_map_.resize({nx_, ny_});
}

Cartesian2D::Cartesian2D(const std::vector<double>& dx,
                         const std::vector<double>& dy)
    : x_bounds_(), y_bounds_(), tiles_(), nx_(), ny_() {
  // Make sure we have at least 1 bin in each direction
  if (dx.size() == 0) {
    auto mssg = "Must provide at least 1 x width.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (dy.size() == 0) {
    auto mssg = "Must provide at least 1 y width.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure all widths > 0
  double dx_tot = 0.;
  for (std::size_t i = 0; i < dx.size(); i++) {
    if (dx[i] <= 0.) {
      std::stringstream mssg;
      mssg << "dx at index " << i << " is <= 0.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    dx_tot += dx[i];
  }

  double dy_tot = 0.;
  for (std::size_t j = 0; j < dy.size(); j++) {
    if (dy[j] <= 0.) {
      std::stringstream mssg;
      mssg << "dy at index " << j << " is <= 0.";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    dy_tot += dy[j];
  }

  // Create surfaces
  x_bounds_.reserve(dx.size() + 1);
  x_bounds_.push_back(std::make_shared<XPlane>(-0.5 * dx_tot));
  for (const auto& d : dx) {
    const double new_x0 = x_bounds_.back()->x0() + d;
    x_bounds_.push_back(std::make_shared<XPlane>(new_x0));
  }

  y_bounds_.reserve(dy.size() + 1);
  y_bounds_.push_back(std::make_shared<YPlane>(-0.5 * dy_tot));
  for (const auto& d : dy) {
    const double new_y0 = y_bounds_.back()->y0() + d;
    y_bounds_.push_back(std::make_shared<YPlane>(new_y0));
  }

  nx_ = dx.size();
  ny_ = dy.size();

  // All tiles start as uninitialized
  tiles_.resize({nx_, ny_});
  tiles_.fill(Tile{nullptr, nullptr});

  fsr_offset_map_.resize({nx_, ny_});
}

Cartesian2D::Cartesian2D(const Tuple& t)
    : x_bounds_(std::get<0>(t)),
      y_bounds_(std::get<1>(t)),
      tiles_(),
      fsr_offset_map_(),
      nx_(std::get<4>(t)),
      ny_(std::get<5>(t)) {
  // Get flat arrays
  std::vector<Tile::Tuple> flat_tiles = std::get<2>(t);
  std::vector<std::map<std::size_t, std::size_t>> flat_fsr_offset_map =
      std::get<3>(t);

  // Reconstruct 2D arrays
  const std::size_t NT = nx_ * ny_;
  tiles_.resize({nx_, ny_});
  fsr_offset_map_.resize({nx_, ny_});
  for (std::size_t i = 0; i < NT; i++) {
    tiles_.flat(i) = Tile::from_tuple(flat_tiles[i]);
    fsr_offset_map_.flat(i) = flat_fsr_offset_map[i];
  }
}

void Cartesian2D::set_tile(const TileIndex& ti,
                           const std::shared_ptr<Cartesian2D>& c2d) {
  check_tile_index(ti);
  const auto dxdy = tile_dx_dy(ti);

  if (std::abs(dxdy.first - c2d->dx()) > VEC_FP_TOL) {
    auto mssg = "x width does not agree with tile x width";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (std::abs(dxdy.second - c2d->dy()) > VEC_FP_TOL) {
    auto mssg = "y width does not agree with tile y width";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  tiles_(ti.i, ti.j).c2d = c2d;
}

void Cartesian2D::set_tile(const TileIndex& ti,
                           const std::shared_ptr<Cell>& cell) {
  check_tile_index(ti);
  const auto dxdy = tile_dx_dy(ti);

  std::pair<double, double> cdxdy = {cell->dx(), cell->dy()};

  if (std::abs(dxdy.first - cdxdy.first) > VEC_FP_TOL) {
    auto mssg = "x width does not agree with tile x width";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (std::abs(dxdy.second - cdxdy.second) > VEC_FP_TOL) {
    auto mssg = "y width does not agree with tile y width";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  tiles_(ti.i, ti.j).cell = cell;
}

void Cartesian2D::set_tiles(const std::vector<TileFill>& fills) {
  // Make sure dimensions match
  if (fills.size() != tiles_.size()) {
    auto mssg = "Number of provided tile fills does not match number of tiles.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  std::size_t indx = 0;
  for (std::size_t j = ny(); j > 0; j--) {
    for (std::size_t i = 0; i < nx(); i++) {
      // Check which fill type we have
      const TileFill& fill = fills[indx];
      if (std::holds_alternative<std::shared_ptr<Cartesian2D>>(fill)) {
        this->set_tile({i, j - 1},
                       std::get<std::shared_ptr<Cartesian2D>>(fill));
      } else {
        this->set_tile({i, j - 1}, std::get<std::shared_ptr<Cell>>(fill));
      }

      indx++;
    }
  }

  make_offset_map();
}

std::pair<double, double> Cartesian2D::tile_dx_dy(const TileIndex& ti) const {
  // Get the lower and upper x and y bounds
  const double xl = x_bounds_[ti.i]->x0();
  const double xh = x_bounds_[ti.i + 1]->x0();
  const double yl = y_bounds_[ti.j]->y0();
  const double yh = y_bounds_[ti.j + 1]->y0();

  return {xh - xl, yh - yl};
}

bool Cartesian2D::tiles_valid() const {
  for (const auto& t : tiles_) {
    if (t.valid() == false) return false;
  }

  return true;
}

std::shared_ptr<CrossSection> Cartesian2D::get_xs(const Vector& r,
                                                  const Direction& u) const {
  try {
    const auto& fsr = this->get_fsr(r, u);

    if (fsr.fsr == nullptr) {
      std::stringstream mssg;
      mssg << "Could not find an FSR at r = " << r << ", u = " << u << ".";
      spdlog::error(mssg.str());
      throw ScarabeeException(mssg.str());
    }

    return fsr.fsr->xs();
  } catch (ScarabeeException& err) {
    err.add_to_exception("Could not find flat source region.");
    throw err;
  }
}

UniqueFSR Cartesian2D::get_fsr(const Vector& r, const Direction& u) const {
  auto ti = this->get_tile_index(r, u);

  if (ti.has_value() == false) {
    // We apparently are not in a tile
    return {nullptr, 0};
  }

  const auto& t = this->tile(*ti);
  const Vector tc = get_tile_center(*ti);
  Vector r_tile = r - tc;

  if (t.valid() == false) {
    std::stringstream mssg;
    mssg << "Tile for Position r = " << r << ", and Direction u = " << u
         << " is empty.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  UniqueFSR out;

  if (t.c2d) {
    out = t.c2d->get_fsr(r_tile, u);
  } else {
    out = t.cell->get_fsr(r_tile, u);
  }

  // Get unique ID
  if (out.fsr) {
    out.instance += fsr_offset_map_(ti->i, ti->j).find(out.fsr->id())->second;
  }

  return out;
}

std::vector<UniqueFSR> Cartesian2D::get_all_fsr_in_cell(
    const Vector& r, const Direction& u) const {
  auto ti = this->get_tile_index(r, u);

  if (ti.has_value() == false) {
    // We apparently are not in a tile. Return empty vector.
    return {};
  }

  const auto& t = this->tile(*ti);
  const Vector tc = get_tile_center(*ti);
  Vector r_tile = r - tc;

  if (t.valid() == false) {
    std::stringstream mssg;
    mssg << "Tile for Position r = " << r << ", and Direction u = " << u
         << " is empty.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  std::vector<UniqueFSR> out;

  if (t.c2d) {
    out = t.c2d->get_all_fsr_in_cell(r_tile, u);
  } else {
    out = t.cell->get_all_fsr_in_cell(r_tile, u);
  }

  // Get unique ID
  for (auto& outi : out) {
    // Shouldn't need to check pointer, as the vector should only containvalid
    // valid FSRs
    outi.instance += fsr_offset_map_(ti->i, ti->j).find(outi.fsr->id())->second;
  }

  return out;
}

std::size_t Cartesian2D::get_num_fsr_instances(std::size_t id) const {
  std::size_t n = 0;
  for (const auto& t : tiles_) {
    n += t.get_num_fsr_instances(id);
  }
  return n;
}

std::size_t Cartesian2D::num_fsrs() const {
  std::size_t n = 0;

  for (const auto& tile : tiles_) n += tile.num_fsrs();

  return n;
}

std::size_t Cartesian2D::Tile::num_fsrs() const {
  if (this->valid() == false) {
    return 0;
  }

  if (c2d) {
    return c2d->num_fsrs();
  } else {
    return cell->num_fsrs();
  }
}

std::size_t Cartesian2D::Tile::ngroups() const {
  if (this->valid() == false) {
    return 0;
  }

  if (c2d) {
    return c2d->ngroups();
  } else {
    return cell->ngroups();
  }
}

std::size_t Cartesian2D::Tile::get_num_fsr_instances(std::size_t id) const {
  if (c2d) {
    return c2d->get_num_fsr_instances(id);
  } else if (cell) {
    return cell->get_num_fsr_instances(id);
  }
  return 0;
}

std::vector<std::size_t> Cartesian2D::get_all_fsr_ids() const {
  std::set<std::size_t> fsr_ids;

  for (std::size_t t = 0; t < tiles_.size(); t++) {
    std::vector<std::size_t> tile_ids;
    if (tiles_.flat(t).c2d) {
      tile_ids = tiles_.flat(t).c2d->get_all_fsr_ids();
    } else if (tiles_.flat(t).cell) {
      tile_ids = tiles_.flat(t).cell->get_all_fsr_ids();
    }

    for (const auto fsr : tile_ids) fsr_ids.insert(fsr);
  }

  std::vector<std::size_t> out_ids(fsr_ids.begin(), fsr_ids.end());
  std::sort(out_ids.begin(), out_ids.end());

  return out_ids;
}

void Cartesian2D::fill_fsrs(
    std::map<std::size_t, const FlatSourceRegion*>& fsrs) const {
  for (const auto& t : tiles_) {
    if (t.c2d) {
      t.c2d->fill_fsrs(fsrs);
    } else if (t.cell) {
      t.cell->fill_fsrs(fsrs);
    }
  }
}

Cartesian2D::Tuple Cartesian2D::to_tuple() const {
  const std::size_t NT = nx_ * ny_;
  if (tiles_.size() != NT || fsr_offset_map_.size() != NT) {
    const auto mssg =
        "The tiles_ and fsr_offset_map_ have inconsistent sizes. Cannot create "
        "tuple.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  std::vector<Tile::Tuple> flat_tiles;
  std::vector<std::map<std::size_t, std::size_t>> flat_fsr_offset_map;
  flat_tiles.reserve(NT);
  flat_fsr_offset_map.reserve(NT);
  for (std::size_t i = 0; i < NT; i++) {
    flat_tiles.push_back(tiles_.flat(i).to_tuple());
    flat_fsr_offset_map.push_back(fsr_offset_map_.flat(i));
  }

  return {x_bounds_, y_bounds_, flat_tiles, flat_fsr_offset_map, nx_, ny_};
}

void Cartesian2D::make_offset_map() {
  auto fsr_ids = this->get_all_fsr_ids();

  // Set all offsets to be zero
  for (auto& map : fsr_offset_map_) {
    for (std::size_t fsr_id : fsr_ids) {
      map[fsr_id] = 0;
    }
  }

  // Now go through and build all offsets
  for (std::size_t i = 1; i < tiles_.size(); i++) {
    const auto& t = tiles_.flat(i - 1);

    for (std::size_t fsr_id : fsr_ids) {
      fsr_offset_map_.flat(i)[fsr_id] = fsr_offset_map_.flat(i - 1)[fsr_id];
      fsr_offset_map_.flat(i)[fsr_id] += t.get_num_fsr_instances(fsr_id);
    }
  }
}

}  // namespace scarabee
