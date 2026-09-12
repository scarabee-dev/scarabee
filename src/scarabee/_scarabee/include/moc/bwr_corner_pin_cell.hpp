#ifndef BWR_CORNER_PIN_CELL_H
#define BWR_CORNER_PIN_CELL_H

#include <moc/cell.hpp>
#include <moc/surface.hpp>
#include <moc/pin_cell_type.hpp>
#include <data/cross_section.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/base_class.hpp>

#include <memory>
#include <vector>

namespace scarabee {

class BWRCornerPinCell : public Cell {
 public:
  BWRCornerPinCell(const std::vector<double>& pin_rads,
                   const std::vector<std::shared_ptr<CrossSection>>& pin_mats,
                   double inner_gap, std::shared_ptr<CrossSection> inner_mod,
                   double box_width, double rc,
                   std::shared_ptr<CrossSection> box_mat,
                   std::shared_ptr<CrossSection> outer_mod, double dx,
                   double dy, BWRCornerType corner_type);
  BWRCornerPinCell(py::tuple t)
      : Cell(t[0].cast<py::tuple>()),
        pin_radii_(t[1].cast<std::vector<double>>()),
        pin_mats_(t[2].cast<std::vector<std::shared_ptr<CrossSection>>>()),
        surfs_(t[3].cast<std::vector<std::shared_ptr<Surface>>>()),
        inner_mod_(t[4].cast<std::shared_ptr<CrossSection>>()),
        inner_gap_(t[5].cast<double>()),
        box_width_(t[6].cast<double>()),
        box_mat_(t[7].cast<std::shared_ptr<CrossSection>>()),
        outer_mod_(t[8].cast<std::shared_ptr<CrossSection>>()),
        xm_(t[9].cast<std::shared_ptr<Surface>>()),
        pd_(t[10].cast<std::shared_ptr<Surface>>()),
        ym_(t[11].cast<std::shared_ptr<Surface>>()),
        nd_(t[12].cast<std::shared_ptr<Surface>>()),
        rc_(t[13].cast<double>()),
        corner_type_(t[14].cast<BWRCornerType>()) {}

  py::tuple to_tuple() const {
    return py::make_tuple(Cell::to_tuple(), pin_radii_, pin_mats_, surfs_,
                          inner_mod_, inner_gap_, box_width_, box_mat_,
                          outer_mod_, xm_, pd_, ym_, nd_, rc_, corner_type_);
  }

 private:
  std::vector<double> pin_radii_;
  std::vector<std::shared_ptr<CrossSection>> pin_mats_;
  std::vector<std::shared_ptr<Surface>> surfs_;
  std::shared_ptr<CrossSection> inner_mod_;
  double inner_gap_;
  double box_width_;
  std::shared_ptr<CrossSection> box_mat_;
  std::shared_ptr<CrossSection> outer_mod_;
  std::shared_ptr<Surface> xm_, pd_, ym_, nd_;  // Surfaces for pin octants
  double rc_;
  BWRCornerType corner_type_;

  void build_I();
  void build_II();
  void build_III();
  void build_IV();
  void build_pin(const double Rpx, const double Rpy);

  void estimate_unknown_areas_cull_fsrs();

  friend class cereal::access;
  BWRCornerPinCell() {}

  template <class Archive>
  void serialize(Archive& arc) {
    arc(cereal::base_class<Cell>(this), CEREAL_NVP(pin_radii_),
        CEREAL_NVP(pin_mats_), CEREAL_NVP(surfs_), CEREAL_NVP(inner_mod_),
        CEREAL_NVP(inner_gap_), CEREAL_NVP(box_width_), CEREAL_NVP(box_mat_),
        CEREAL_NVP(outer_mod_), CEREAL_NVP(xm_), CEREAL_NVP(pd_),
        CEREAL_NVP(ym_), CEREAL_NVP(nd_), CEREAL_NVP(rc_),
        CEREAL_NVP(corner_type_));
  }
};

}  // namespace scarabee

#endif
