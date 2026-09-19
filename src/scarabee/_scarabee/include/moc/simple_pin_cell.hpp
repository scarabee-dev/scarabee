#ifndef SIMPLE_PIN_CELL_H
#define SIMPLE_PIN_CELL_H

#include <moc/cell.hpp>
#include <moc/surface.hpp>
#include <moc/pin_cell_type.hpp>
#include <data/cross_section.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/base_class.hpp>

#include <memory>
#include <tuple>

struct SimplePinCellPickler;

namespace scarabee {

class SimplePinCell : public Cell {
 public:
  using Tuple = std::tuple<Cell::Tuple, std::vector<double>,
                           std::vector<std::shared_ptr<CrossSection>>,
                           std::vector<std::shared_ptr<Surface>>, std::uint8_t>;

  SimplePinCell(const std::vector<double>& rads,
                const std::vector<std::shared_ptr<CrossSection>>& mats,
                double dx, double dy, PinCellType pin_type = PinCellType::Full);
  SimplePinCell(const Tuple& t)
      : Cell(std::get<0>(t)),
        mat_radii_(std::get<1>(t)),
        mats_(std::get<2>(t)),
        radii_(std::get<3>(t)),
        pin_type_(static_cast<PinCellType>(std::get<4>(t))) {}

  Tuple to_tuple() const {
    return {Cell::to_tuple(), mat_radii_, mats_, radii_,
            static_cast<std::uint8_t>(pin_type_)};
  }

 private:
  std::vector<double> mat_radii_;
  std::vector<std::shared_ptr<CrossSection>> mats_;
  std::vector<std::shared_ptr<Surface>> radii_;
  PinCellType pin_type_;

  friend class cereal::access;
  friend struct ::SimplePinCellPickler;
  SimplePinCell() {}
  template <class Archive>
  void serialize(Archive& arc) {
    arc(cereal::base_class<Cell>(this), CEREAL_NVP(mat_radii_),
        CEREAL_NVP(mats_), CEREAL_NVP(radii_), CEREAL_NVP(pin_type_));
  }

  void build_full();
  void build_xn();
  void build_xp();
  void build_yn();
  void build_yp();
  void build_i();
  void build_ii();
  void build_iii();
  void build_iv();
};

}  // namespace scarabee

#endif
