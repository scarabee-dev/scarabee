#ifndef POLAR_QUADRATURE_H
#define POLAR_QUADRATURE_H

#include <moc/quadrature/legendre.hpp>
#include <moc/quadrature/yamamoto_tabuchi.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/variant.hpp>

#include <span>
#include <variant>

namespace scarabee {

using PolarQuadratureType =
    std::variant<Legendre<2>, Legendre<4>, Legendre<6>, Legendre<8>,
                 Legendre<10>, Legendre<12>, Legendre<16>, Legendre<32>,
                 Legendre<64>, YamamotoTabuchi<2>, YamamotoTabuchi<4>,
                 YamamotoTabuchi<6>>;

class PolarQuadrature {
 public:
  template <typename P>
  PolarQuadrature(P pq) : pq_(pq), invs_sin_(), wsin_(), sin_(), wgt_() {
    this->set_spans();
  }

  PolarQuadrature(PolarQuadratureType pq)
      : pq_(pq), invs_sin_(), wsin_(), sin_(), wgt_() {
    this->set_spans();
  }

  const std::span<const double>& invs_sin() const { return invs_sin_; }
  const std::span<const double>& wsin() const { return wsin_; }
  const std::span<const double>& sin() const { return sin_; }
  const std::span<const double>& wgt() const { return wgt_; }
  const std::span<const double>& polar_angle() const { return polar_angle_; }

 private:
  PolarQuadratureType pq_;
  std::span<const double> invs_sin_;
  std::span<const double> wsin_;
  std::span<const double> sin_;
  std::span<const double> wgt_;
  std::span<const double> polar_angle_;

  void set_spans() {
    invs_sin_ = std::visit([](const auto& pq) { return pq.invs_sin(); }, pq_);
    wsin_ = std::visit([](const auto& pq) { return pq.wsin(); }, pq_);
    sin_ = std::visit([](const auto& pq) { return pq.sin(); }, pq_);
    wgt_ = std::visit([](const auto& pq) { return pq.wgt(); }, pq_);
    polar_angle_ =
        std::visit([](const auto& pq) { return pq.polar_angle(); }, pq_);
  }

  friend class cereal::access;
  PolarQuadrature() {}

  template <class Archive>
  void save(Archive& arc) const {
    arc(CEREAL_NVP(pq_));
  }

  template <class Archive>
  void load(Archive& arc) {
    arc(CEREAL_NVP(pq_));
    this->set_spans();
  }
};

}  // namespace scarabee

#endif
