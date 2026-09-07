#ifndef SEGMENT_H
#define SEGMENT_H

#include <moc/flat_source_region.hpp>
#include <data/cross_section.hpp>
#include <moc/cmfd.hpp>

#include <xtensor/containers/xtensor.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>

#include <memory>
#include <tuple>

namespace scarabee {

class Segment {
 public:
  using Tuple =
      std::tuple<std::shared_ptr<CrossSection>, double, double, std::size_t,
                 CMFDSurfaceCrossing::Tuple, CMFDSurfaceCrossing::Tuple>;

  Segment(const FlatSourceRegion* fsr, double length, std::size_t indx)
      : xs_(fsr->xs()),
        volume_(fsr->volume()),
        length_(length),
        fsr_indx_(indx),
        entry_cmfd_surface_(),
        exit_cmfd_surface_() {}

  Segment(const Tuple& t)
      : xs_(std::get<0>(t)),
        volume_(std::get<1>(t)),
        length_(std::get<2>(t)),
        fsr_indx_(std::get<3>(t)),
        entry_cmfd_surface_(CMFDSurfaceCrossing::from_tuple(std::get<4>(t))),
        exit_cmfd_surface_(CMFDSurfaceCrossing::from_tuple(std::get<5>(t))) {}

  // Here for use with cereal and std::vector
  Segment() {}

  double length() const { return length_; }

  void set_length(double l) { length_ = l; }

  double volume() const { return volume_; }

  const std::shared_ptr<CrossSection>& xs() const { return xs_; }

  std::size_t fsr_indx() const { return fsr_indx_; }

  CMFDSurfaceCrossing& entry_cmfd_surface() { return entry_cmfd_surface_; }
  const CMFDSurfaceCrossing& entry_cmfd_surface() const {
    return entry_cmfd_surface_;
  }

  CMFDSurfaceCrossing& exit_cmfd_surface() { return exit_cmfd_surface_; }
  const CMFDSurfaceCrossing& exit_cmfd_surface() const {
    return exit_cmfd_surface_;
  }

  Tuple to_tuple() const {
    return {xs_,
            volume_,
            length_,
            fsr_indx_,
            entry_cmfd_surface_.to_tuple(),
            exit_cmfd_surface_.to_tuple()};
  }

 private:
  std::shared_ptr<CrossSection> xs_;
  double volume_;
  double length_;
  std::size_t fsr_indx_;
  CMFDSurfaceCrossing entry_cmfd_surface_;
  CMFDSurfaceCrossing exit_cmfd_surface_;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(xs_), CEREAL_NVP(volume_), CEREAL_NVP(length_),
        CEREAL_NVP(fsr_indx_), CEREAL_NVP(entry_cmfd_surface_),
        CEREAL_NVP(exit_cmfd_surface_));
  }
};

}  // namespace scarabee

#endif
