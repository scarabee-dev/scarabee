#ifndef SCARABEE_DEPLETION_MATRIX_H
#define SCARABEE_DEPLETION_MATRIX_H

#include <data/depletion_chain.hpp>
#include <data/nd_library.hpp>
#include <data/material.hpp>
#include <data/micro_cross_sections.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>
#include <utils/serialization.hpp>

#include <Eigen/SparseCore>

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

#include <array>
#include <complex>
#include <memory>
#include <string>
#include <vector>

struct DepletionMatrixPickler;

namespace scarabee {

class DepletionMatrix {
 public:
  DepletionMatrix(const std::vector<std::string>& nuclides);

  const std::vector<std::string>& nuclides() const { return nuclides_; }
  bool has_nuclide(const std::string& nuclide) const;
  std::size_t get_nuclide_index(const std::string& nuclide) const;

  std::size_t size() const { return nuclides_.size(); }

  double value(std::size_t row, std::size_t col) const {
    return matrix_.coeff(row, col);
  }

  double& ref(std::size_t row, std::size_t col) {
    return matrix_.coeffRef(row, col);
  }

  void zero() { matrix_.setZero(); }

  bool is_compressed() const { return matrix_.isCompressed(); }
  void compress() { matrix_.makeCompressed(); }

  void exponential_product(std::span<double> N, bool cram48 = false) const;

  DepletionMatrix& operator+=(const DepletionMatrix& A) {
    if (same_nuclides(this->nuclides(), A.nuclides()) == false) {
      const auto mssg = "Depletion matrices do not have the same nuclides.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    this->matrix_ += A.matrix_;
    return *this;
  }

  DepletionMatrix& operator-=(const DepletionMatrix& A) {
    if (same_nuclides(this->nuclides(), A.nuclides()) == false) {
      const auto mssg = "Depletion matrices do not have the same nuclides.";
      spdlog::error(mssg);
      throw ScarabeeException(mssg);
    }

    this->matrix_ -= A.matrix_;
    return *this;
  }

  DepletionMatrix& operator*=(double c) {
    matrix_ *= c;
    return *this;
  }

  DepletionMatrix& operator/=(double c) {
    matrix_ /= c;
    return *this;
  }

  DepletionMatrix operator+(const DepletionMatrix& A) const {
    DepletionMatrix out(*this);
    out += A;
    return out;
  }

  DepletionMatrix operator-(const DepletionMatrix& A) const {
    DepletionMatrix out(*this);
    out -= A;
    return out;
  }

  DepletionMatrix operator*(double c) const {
    DepletionMatrix out(*this);
    out *= c;
    return out;
  }

  DepletionMatrix operator/(double c) const {
    DepletionMatrix out(*this);
    out /= c;
    return out;
  }

 private:
  std::vector<std::string> nuclides_;
  Eigen::SparseMatrix<double> matrix_;

  friend class cereal::access;
  friend struct ::DepletionMatrixPickler;

  DepletionMatrix() = default;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(CEREAL_NVP(nuclides_), CEREAL_NVP(matrix_));
  }

  bool same_nuclides(const std::vector<std::string>& n1,
                     const std::vector<std::string>& n2) const {
    if (n1.size() != n2.size()) return false;

    for (std::size_t i = 0; i < n1.size(); i++) {
      if (n1[i] != n2[i]) return false;
    }

    return true;
  }

  void detail_exp_product(std::span<double> N,
                          std::span<const std::complex<double>> theta,
                          std::span<const std::complex<double>> alpha,
                          double alpha0) const;

  // Static data for performing matrix exponentials
  static const std::array<std::complex<double>, 8> cram16_alpha_;
  static const std::array<std::complex<double>, 8> cram16_theta_;
  static const double cram16_alpha0_;

  static const std::array<std::complex<double>, 24> cram48_alpha_;
  static const std::array<std::complex<double>, 24> cram48_theta_;
  static const double cram48_alpha0_;
};

std::shared_ptr<DepletionMatrix> build_depletion_matrix(
    std::shared_ptr<DepletionChain> chain, std::shared_ptr<Material> mat,
    std::span<const double> flux, std::shared_ptr<NDLibrary> ndl);

std::shared_ptr<DepletionMatrix> build_depletion_matrix(
    std::shared_ptr<DepletionChain> chain,
    std::vector<DepletionReactionRates>& nuc_rrs);

}  // namespace scarabee

#endif
