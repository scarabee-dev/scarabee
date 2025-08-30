#ifndef TRACK_H
#define TRACK_H

#include <moc/segment.hpp>
#include <moc/vector.hpp>
#include <moc/direction.hpp>
#include <moc/boundary_condition.hpp>
#include <utils/constants.hpp>
#include <utils/serialization.hpp>

#include <xtensor/containers/xtensor.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>

#include <vector>

namespace scarabee {

class Track {
 public:
  Track(const Vector& entry, const Vector& exit, const Direction& dir,
        double phi, double wgt, double width,
        const std::vector<Segment>& segments, std::size_t forward_phi_index,
        std::size_t backward_phi_index);

  // Here for use with cereal and std::vector
  Track() : entry_(0., 0.), exit_(0., 0.) {}

  double wgt() const { return wgt_; }
  double width() const { return width_; }
  double phi() const { return phi_; }
  const std::vector<Segment>& segments() const { return segments_; }
  std::size_t phi_index_forward() const { return forward_phi_index_; }
  std::size_t phi_index_backward() const { return backward_phi_index_; }

  const Vector& entry_pos() const { return entry_; }
  const Vector& exit_pos() const { return exit_; }
  const Direction& dir() const { return dir_; }

  std::size_t& entry_cmfd_cell() { return cmfd_entry_cell_; }
  std::size_t& exit_cmfd_cell() { return cmfd_exit_cell_; }

  BoundaryCondition& entry_bc() { return entry_bc_; }
  const BoundaryCondition& entry_bc() const { return entry_bc_; }

  BoundaryCondition& exit_bc() { return exit_bc_; }
  const BoundaryCondition& exit_bc() const { return exit_bc_; }

  xt::xtensor<double, 2>& entry_flux() { return entry_flux_; }
  const xt::xtensor<double, 2>& entry_flux() const { return entry_flux_; }

  xt::xtensor<double, 2>& exit_flux() { return exit_flux_; }
  const xt::xtensor<double, 2>& exit_flux() const { return exit_flux_; }

  xt::xtensor<double, 2>& entry_track_flux() { return *entry_track_flux_; }
  void set_entry_track_flux(xt::xtensor<double, 2>* etf) {
    entry_track_flux_ = etf;
  }

  xt::xtensor<double, 2>& exit_track_flux() { return *exit_track_flux_; }
  void set_exit_track_flux(xt::xtensor<double, 2>* etf) {
    exit_track_flux_ = etf;
  }

  // Indexing is only done in forward direction
  std::size_t size() const { return segments_.size(); }

  Segment& operator[](std::size_t i) { return segments_[i]; }
  const Segment& operator[](std::size_t i) const { return segments_[i]; }

  Segment& at(std::size_t i);
  const Segment& at(std::size_t i) const;

  //--------------------------------------------------------------------------
  // Iterators
  using iterator = std::vector<Segment>::iterator;
  using const_iterator = std::vector<Segment>::const_iterator;
  using reverse_iterator = std::vector<Segment>::reverse_iterator;
  using const_reverse_iterator = std::vector<Segment>::const_reverse_iterator;

  iterator begin() { return segments_.begin(); }
  const_iterator begin() const { return segments_.begin(); }
  const_iterator cbegin() const { return segments_.cbegin(); }

  iterator end() { return segments_.end(); }
  const_iterator end() const { return segments_.end(); }
  const_iterator cend() const { return segments_.cend(); }

  reverse_iterator rbegin() { return segments_.rbegin(); }
  const_reverse_iterator rbegin() const { return segments_.rbegin(); }
  const_reverse_iterator crbegin() const { return segments_.crbegin(); }

  reverse_iterator rend() { return segments_.rend(); }
  const_reverse_iterator rend() const { return segments_.rend(); }
  const_reverse_iterator crend() const { return segments_.crend(); }

 private:
  xt::xtensor<double, 2> entry_flux_;  // Indexed on group and polar angle
  xt::xtensor<double, 2> exit_flux_;
  std::vector<Segment> segments_;
  Vector entry_;
  Vector exit_;
  Direction dir_;
  xt::xtensor<double, 2>* entry_track_flux_;
  xt::xtensor<double, 2>* exit_track_flux_;
  double wgt_;    // Track weight
  double width_;  // Track width
  double phi_;    // Azimuthal angle of Track
  BoundaryCondition entry_bc_;
  BoundaryCondition exit_bc_;
  std::size_t forward_phi_index_;  // azimuthal angle index in forward direction
  std::size_t
      backward_phi_index_;  // azimuthal angle index in backaward direction

  std::size_t cmfd_entry_cell_;
  std::size_t cmfd_exit_cell_;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(entry_flux_), CEREAL_NVP(exit_flux_), CEREAL_NVP(segments_),
        CEREAL_NVP(entry_), CEREAL_NVP(exit_), CEREAL_NVP(dir_),
        CEREAL_NVP(wgt_), CEREAL_NVP(width_), CEREAL_NVP(phi_),
        CEREAL_NVP(entry_bc_), CEREAL_NVP(exit_bc_),
        CEREAL_NVP(forward_phi_index_), CEREAL_NVP(backward_phi_index_),
        CEREAL_NVP(cmfd_entry_cell_), CEREAL_NVP(cmfd_exit_cell_));
  }
};

}  // namespace scarabee

#endif
