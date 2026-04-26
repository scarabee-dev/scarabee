#ifndef SCARABEE_NODAL_CMFD_SURFACE_H
#define SCARABEE_NODAL_CMFD_SURFACE_H

#include <diffusion/diffusion_geometry.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <compare>
#include <optional>

namespace scarabee {

namespace detail {
template <class T>
inline void hash_combine(std::uint64_t& s, const T& v) {
  std::hash<T> h;
  s ^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
}
}  // namespace detail

using Side = DiffusionGeometry::Neighbor;  // Index to side of node

struct NodalCMFDSurface {
  NodalCMFDSurface(std::size_t n1, Side s, std::size_t n2)
      : node1(n1), side(s), node2(n2) {
    // Make sure node1 is always on the - side, and node2 is always on the +
    // side ! We do this by swapping if s is a negative side.
    switch (s) {
      case Side::XN:
        std::swap(n1, n2);
        s = Side::XP;
        break;
      case Side::YN:
        std::swap(n1, n2);
        s = Side::YP;
        break;
      case Side::ZN:
        std::swap(n1, n2);
        s = Side::ZP;
        break;
      default:
        break;
    }

    if (n1 == n2) {
      // Apparently, n1 = n2, which does not define a side !
      const auto mssg = "Node1 and Node2 cannot be equal.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    node1 = n1;
    node2 = n2;
    side = s;
  }
  NodalCMFDSurface(std::size_t n1, Side s)
      : node1(n1), side(s), node2(std::nullopt) {}

  std::strong_ordering operator<=>(const NodalCMFDSurface& other) const {
    if (this->node1 < other.node1)
      return std::strong_ordering::less;
    else if (this->node1 > other.node1)
      return std::strong_ordering::greater;

    if (this->node2 && other.node2 == false)
      return std::strong_ordering::less;
    else if (this->node2 == false && other.node2)
      return std::strong_ordering::greater;
    else if (this->node2 && other.node2) {
      if (this->node2.value() < other.node2.value())
        return std::strong_ordering::less;
      else if (this->node2.value() > other.node2.value())
        return std::strong_ordering::greater;
    }
    return std::strong_ordering::equal;
  }

  bool operator<(const NodalCMFDSurface& other) const = default;
  bool operator<=(const NodalCMFDSurface& other) const = default;
  bool operator==(const NodalCMFDSurface& other) const = default;
  bool operator!=(const NodalCMFDSurface& other) const = default;
  bool operator>=(const NodalCMFDSurface& other) const = default;
  bool operator>(const NodalCMFDSurface& other) const = default;

  std::size_t node1;
  Side side;
  std::optional<std::size_t> node2;
};

}  // namespace scarabee

namespace std {
template <>
class hash<scarabee::NodalCMFDSurface> {
 public:
  std::uint64_t operator()(const scarabee::NodalCMFDSurface& s) const {
    std::uint64_t res = 0;
    ::scarabee::detail::hash_combine(res, s.node1);
    ::scarabee::detail::hash_combine(res, s.side);
    ::scarabee::detail::hash_combine(res, s.node2);
    return res;
  }
};
}  // namespace std

#endif
