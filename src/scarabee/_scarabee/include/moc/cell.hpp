#ifndef CELL_H
#define CELL_H

#include <data/cross_section.hpp>
#include <moc/direction.hpp>
#include <moc/segment.hpp>
#include <moc/vector.hpp>
#include <moc/flat_source_region.hpp>
#include <moc/surface.hpp>
#include <utils/constants.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>

namespace scarabee {

class Cell {
 public:
  using Tuple = std::tuple<std::vector<FlatSourceRegion::Tuple>,
                           std::shared_ptr<Surface>, std::shared_ptr<Surface>,
                           std::shared_ptr<Surface>, std::shared_ptr<Surface>>;

  // Must provide a virtual destructor so that Pybind11 can use RTTI to down
  // cast the specific type of Cell (i.e. PinCell) which is picklable.
  virtual ~Cell() = default;

  bool inside(const Vector& r, const Direction& u) const {
    if (x_min_->side(r, u) == Surface::Side::Negative) return false;
    if (y_min_->side(r, u) == Surface::Side::Negative) return false;
    if (x_max_->side(r, u) == Surface::Side::Positive) return false;
    if (y_max_->side(r, u) == Surface::Side::Positive) return false;
    return true;
  }

  double distance(const Vector& r, const Direction& u) const {
    const double x_min_dist = x_min_->distance(r, u);
    const double x_max_dist = x_max_->distance(r, u);
    const double y_min_dist = y_min_->distance(r, u);
    const double y_max_dist = y_max_->distance(r, u);
    return std::min(std::min(x_min_dist, x_max_dist),
                    std::min(y_min_dist, y_max_dist));
  }

  UniqueFSR get_fsr(const Vector& r, const Direction& u) const {
    std::stringstream mssg;
    if (this->inside(r, u) == false) {
      return {nullptr, 0};
    }

    for (const auto& fsr : fsrs_) {
      if (fsr.inside(r, u)) return {&fsr, 0};
    }

    mssg << "Could not find FSR at r = " << r << ", u = " << u << ".";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());

    // NEVER GETS HERE
    return {&fsrs_.front(), 0};
  }

  std::vector<UniqueFSR> get_all_fsr_in_cell(const Vector& r,
                                             const Direction& u) const;

  std::size_t num_fsrs() const { return fsrs_.size(); }

  std::size_t ngroups() const { return fsrs_.front().xs()->ngroups(); }

  std::vector<std::size_t> get_all_fsr_ids() const;

  std::size_t get_num_fsr_instances(std::size_t id) const;

  void fill_fsrs(std::map<std::size_t, const FlatSourceRegion*>& fsrs) const;

  double dx() const { return x_max_->x0() - x_min_->x0(); }
  double dy() const { return y_max_->y0() - y_min_->y0(); }

  Tuple to_tuple() const {
    std::vector<FlatSourceRegion::Tuple> fsr_list;
    for (const auto& fsr : fsrs_) fsr_list.push_back(fsr.to_tuple());
    return {fsr_list, x_min_, y_min_, x_max_, y_max_};
  }

 protected:
  std::vector<FlatSourceRegion> fsrs_;
  std::shared_ptr<Surface> x_min_, y_min_, x_max_, y_max_;

  Cell(const Tuple& t) {
    std::vector<FlatSourceRegion::Tuple> fsr_list = std::get<0>(t);
    fsrs_.reserve(fsr_list.size());
    for (std::size_t i = 0; i < fsr_list.size(); i++)
      fsrs_.push_back(FlatSourceRegion(fsr_list[i]));
    x_min_ = std::get<1>(t);
    y_min_ = std::get<2>(t);
    x_max_ = std::get<3>(t);
    y_max_ = std::get<4>(t);
  }

  Cell(double dx, double dy);
  void check_surfaces() const;

  friend class cereal::access;
  Cell() {}
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(fsrs_), CEREAL_NVP(x_min_), CEREAL_NVP(y_min_),
        CEREAL_NVP(x_max_), CEREAL_NVP(y_max_));
  }
};

}  // namespace scarabee

#endif
