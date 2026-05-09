#ifndef SCARABEE_NODE_H
#define SCARABEE_NODE_H

#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>

#include <array>

namespace scarabee {

class Node {
 public:
  Node() : data_() {
    data_.fill(0.);

    // Initialize ADFs to 1
    this->adf_xp() = 1.;
    this->adf_xn() = 1.;
    this->adf_yp() = 1.;
    this->adf_yn() = 1.;
    this->adf_zp() = 1.;
    this->adf_zn() = 1.;
  }

  // Average flux
  double phi0() const { return data_[0]; }
  double& phi0() { return data_[0]; }

  // Average Surface Currents
  double J_xp() const { return data_[1]; }
  double& J_xp() { return data_[1]; }

  double J_xn() const { return data_[2]; }
  double& J_xn() { return data_[2]; }

  double J_yp() const { return data_[3]; }
  double& J_yp() { return data_[3]; }

  double J_yn() const { return data_[4]; }
  double& J_yn() { return data_[4]; }

  double J_zp() const { return data_[5]; }
  double& J_zp() { return data_[5]; }

  double J_zn() const { return data_[6]; }
  double& J_zn() { return data_[6]; }

  // Expansion terms for the transverse leakage when solving for x
  double Lx_rho_y1() const { return data_[7]; }
  double& Lx_rho_y1() { return data_[7]; }

  double Lx_rho_y2() const { return data_[8]; }
  double& Lx_rho_y2() { return data_[8]; }

  double Lx_rho_z1() const { return data_[9]; }
  double& Lx_rho_z1() { return data_[9]; }

  double Lx_rho_z2() const { return data_[10]; }
  double& Lx_rho_z2() { return data_[10]; }

  // Expansion terms for the transverse leakage when solving for y
  double Ly_rho_x1() const { return data_[11]; }
  double& Ly_rho_x1() { return data_[11]; }

  double Ly_rho_x2() const { return data_[12]; }
  double& Ly_rho_x2() { return data_[12]; }

  double Ly_rho_z1() const { return data_[13]; }
  double& Ly_rho_z1() { return data_[13]; }

  double Ly_rho_z2() const { return data_[14]; }
  double& Ly_rho_z2() { return data_[14]; }

  // Expansion terms for the transverse leakage when solving for z
  double Lz_rho_x1() const { return data_[15]; }
  double& Lz_rho_x1() { return data_[15]; }

  double Lz_rho_x2() const { return data_[16]; }
  double& Lz_rho_x2() { return data_[16]; }

  double Lz_rho_y1() const { return data_[17]; }
  double& Lz_rho_y1() { return data_[17]; }

  double Lz_rho_y2() const { return data_[18]; }
  double& Lz_rho_y2() { return data_[18]; }

  // Discontinuity factors
  double adf_xp() const { return data_[19]; }
  double& adf_xp() { return data_[19]; }

  double adf_xn() const { return data_[20]; }
  double& adf_xn() { return data_[20]; }

  double adf_yp() const { return data_[21]; }
  double& adf_yp() { return data_[21]; }

  double adf_yn() const { return data_[22]; }
  double& adf_yn() { return data_[22]; }

  double adf_zp() const { return data_[23]; }
  double& adf_zp() { return data_[23]; }

  double adf_zn() const { return data_[24]; }
  double& adf_zn() { return data_[24]; }

  // Surface fluxes
  double phi_xp() const { return data_[25]; }
  double& phi_xp() { return data_[25]; }

  double phi_xn() const { return data_[26]; }
  double& phi_xn() { return data_[26]; }

  double phi_yp() const { return data_[27]; }
  double& phi_yp() { return data_[27]; }

  double phi_yn() const { return data_[28]; }
  double& phi_yn() { return data_[28]; }

  double phi_zp() const { return data_[29]; }
  double& phi_zp() { return data_[29]; }

  double phi_zn() const { return data_[30]; }
  double& phi_zn() { return data_[30]; }

 private:
  // - The average flux phi
  // - Must know 6 Net Currents (each node face)
  // - For each direction, must know transverse leakage coeffs px1, px2, etc
  //   Must know 12 transverse leakage coefficients
  // - Must know 6 ADFs, one for each side
  // - Must know 6 surface fluxes
  std::array<double, 31> data_;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(data_));
  }
};
}  // namespace scarabee

#endif
