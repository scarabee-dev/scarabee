#ifndef EMPTY_CELL_H
#define EMPTY_CELL_H

#include <moc/cell.hpp>
#include <moc/surface.hpp>
#include <data/cross_section.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/base_class.hpp>

#include <memory>
#include <tuple>

namespace scarabee {

class EmptyCell : public Cell {
 public:
  using Tuple = std::tuple<Cell::Tuple, std::shared_ptr<CrossSection>>;

  EmptyCell(const std::shared_ptr<CrossSection>& mat, double dx, double dy);
  EmptyCell(const Tuple& t) : Cell(std::get<0>(t)), mat_(std::get<1>(t)) {}

  Tuple to_tuple() const { return {Cell::to_tuple(), mat_}; }

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
