#ifndef FLAT_SOURCE_REGION_H
#define FLAT_SOURCE_REGION_H

#include <moc/surface.hpp>
#include <moc/vector.hpp>
#include <moc/direction.hpp>
#include <data/cross_section.hpp>
#include <utils/constants.hpp>
#include <utils/logging.hpp>
#include <utils/serialization.hpp>
#include <utils/scarabee_exception.hpp>

#include <htl/static_vector.hpp>

#include <xtensor/containers/xtensor.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

#include <memory>
#include <tuple>

namespace scarabee {

struct RegionToken {
  std::shared_ptr<Surface> surface;
  Surface::Side side;

  bool inside(const Vector& r, const Direction& u) const {
    const auto current_side = surface->side(r, u);
    return current_side == side;
  }

  using Tuple = std::tuple<std::shared_ptr<Surface>, bool>;
  static RegionToken from_tuple(const Tuple& t) {
    RegionToken rt;
    rt.surface = std::get<0>(t);
    rt.side = static_cast<Surface::Side>(std::get<1>(t));
    return rt;
  }

  Tuple to_tuple() const { return {surface, static_cast<bool>(side)}; }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(surface), CEREAL_NVP(side));
  }
};

class FlatSourceRegion {
 public:
  using Tuple = std::tuple<std::vector<RegionToken::Tuple>,
                           std::shared_ptr<CrossSection>, double, std::size_t>;

  FlatSourceRegion() : tokens_(), xs_(), volume_(), id_(id_counter++) {}

  FlatSourceRegion(const Tuple& t) {
    // Fill tokens
    std::vector<RegionToken::Tuple> tokens_list = std::get<0>(t);

    if (tokens_list.size() > tokens_.capacity()) {
      auto mssg = "Cannot unpickle FlatSourceRegion. Too many RegionTokens.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    for (std::size_t i = 0; i < tokens_list.size(); i++)
      tokens_.push_back(RegionToken::from_tuple(tokens_list[i]));

    // Get the xs, volume, and id
    xs_ = std::get<1>(t);
    volume_ = std::get<2>(t);
    id_ = std::get<3>(t);

    // If the id is greater than the know id, we increment to avoid any id
    // collisions.
    if (id_ >= id_counter) id_counter = id_ + 1;
  }

  bool inside(const Vector& r, const Direction& u) const {
    for (const auto& t : tokens_) {
      if (t.inside(r, u) == false) return false;
    }
    return true;
  }

  double distance(const Vector& r, const Direction& u) const {
    double min_dist = INF;
    for (const auto& t : tokens_) {
      double token_dist = t.surface->distance(r, u);
      if (token_dist < min_dist) min_dist = token_dist;
    }
    return min_dist;
  }

  std::size_t id() const { return id_; }

  std::shared_ptr<CrossSection>& xs() { return xs_; }
  const std::shared_ptr<CrossSection>& xs() const { return xs_; }

  htl::static_vector<RegionToken, MAX_SURFS>& tokens() { return tokens_; }
  const htl::static_vector<RegionToken, MAX_SURFS>& tokens() const {
    return tokens_;
  }

  double& volume() { return volume_; }
  const double& volume() const { return volume_; }

  Tuple to_tuple() const {
    std::vector<RegionToken::Tuple> tokens_list;
    for (const auto& token : tokens_) {
      tokens_list.push_back(token.to_tuple());
    }
    return {tokens_list, xs_, volume_, id_};
  }

 private:
  htl::static_vector<RegionToken, MAX_SURFS> tokens_;
  std::shared_ptr<CrossSection> xs_;
  double volume_;
  std::size_t id_;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(tokens_), CEREAL_NVP(xs_), CEREAL_NVP(volume_),
        CEREAL_NVP(id_));
  }

  static std::size_t id_counter;
};

struct UniqueFSR {
  const FlatSourceRegion* fsr{nullptr};
  std::size_t instance{0};

  bool operator<(const UniqueFSR& rhs) const {
    if (this->fsr < rhs.fsr)
      return true;
    else if (this->fsr > rhs.fsr)
      return false;

    // fsr is equal here
    return this->instance < rhs.instance;
  }

  bool operator==(const UniqueFSR& rhs) const {
    return (this->fsr == rhs.fsr) && (this->instance && rhs.instance);
  }
};

}  // namespace scarabee

#endif
