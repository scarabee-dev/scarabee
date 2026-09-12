#ifndef EMPTY_CELL_H
#define EMPTY_CELL_H

#include <moc/cell.hpp>
#include <moc/surface.hpp>
#include <data/cross_section.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/base_class.hpp>

#include <memory>

namespace scarabee {

class EmptyCell : public Cell {
 public:
  EmptyCell(const std::shared_ptr<CrossSection>& mat, double dx, double dy);
  EmptyCell(py::tuple t)
      : Cell(t[0].cast<py::tuple>()),
        mat_(t[1].cast<std::shared_ptr<CrossSection>>()) {}

  py::tuple to_tuple() const { return py::make_tuple(Cell::to_tuple(), mat_); }

 private:
  std::shared_ptr<CrossSection> mat_;

  friend class cereal::access;
  EmptyCell() {}
  template <class Archive>
  void serialize(Archive& arc) {
    arc(cereal::base_class<Cell>(this), CEREAL_NVP(mat_));
  }
};

}  // namespace scarabee

#endif
