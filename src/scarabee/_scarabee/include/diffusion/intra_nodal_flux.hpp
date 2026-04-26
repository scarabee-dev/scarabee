#ifndef SCARABEE_INTRA_NODAL_FLUX_H
#define SCARABEE_INTRA_NODAL_FLUX_H

#include <cereal/cereal.hpp>

#include <cmath>

namespace scarabee {

// The method used for intranodal flux reconstruction is based on ANOVA-HDMR
// decomposition, as outlined by Bokov et al. [1].
struct IntraNodalFlux {
  double phi_0 = 0.;  // f0
  double eps = 0.;
  double ax0 = 0., ax1 = 0., ax2 = 0., bx1 = 0., bx2 = 0.;  // fx
  double ay0 = 0., ay1 = 0., ay2 = 0., by1 = 0., by2 = 0.;  // fy
  double az0 = 0., az1 = 0., az2 = 0., bz1 = 0., bz2 = 0.;  // fz
  double cxy11 = 0., cxy12 = 0., cxy21 = 0., cxy22 = 0.;    // fxy
  double invs_dx = 0., invs_dy = 0., invs_dz = 0.;
  double zeta_x = 0., zeta_y = 0.;
  double xm = 0., ym = 0., zm = 0.;  // Mid point of node

  double operator()(double x, double y, double z) const {
    x -= xm;
    y -= ym;
    z -= zm;

    return phi_0 + fx(x) + fy(y) + fz(z) + fxy(x, y);
  }

  double flux_xy_no_cross(double x, double y) const {
    x -= xm;
    y -= ym;

    return phi_0 + fx(x) + fy(y);
  }

  double fx(double x) const {
    return ax0 + ax1 * std::cosh(eps * x) + ax2 * std::sinh(eps * x) +
           bx1 * p1(2. * x * invs_dx) + bx2 * p2(2. * x * invs_dx);
  }

  double fy(double y) const {
    return ay0 + ay1 * std::cosh(eps * y) + ay2 * std::sinh(eps * y) +
           by1 * p1(2. * y * invs_dy) + by2 * p2(2. * y * invs_dy);
  }

  double fz(double z) const {
    return az0 + az1 * std::cosh(eps * z) + az2 * std::sinh(eps * z) +
           bz1 * p1(2. * z * invs_dz) + bz2 * p2(2. * z * invs_dz);
  }

  double fxy(double x, double y) const {
    x *= 2. * invs_dx;
    y *= 2. * invs_dy;
    const double p1x = p1(x);
    const double p2x = p2(x);
    const double p1y = p1(y);
    const double p2y = p2(y);
    return cxy11 * p1x * p1y + cxy12 * p1x * p2y + cxy21 * p2x * p1y +
           cxy22 * p2x * p2y;
  }

  double p1(double xi) const { return xi; }
  double p2(double xi) const { return 0.5 * (3. * xi * xi - 1.); }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(phi_0), CEREAL_NVP(eps), CEREAL_NVP(ax0), CEREAL_NVP(ax1),
        CEREAL_NVP(ax2), CEREAL_NVP(bx1), CEREAL_NVP(bx2), CEREAL_NVP(ay0),
        CEREAL_NVP(ay1), CEREAL_NVP(ay2), CEREAL_NVP(by1), CEREAL_NVP(by2),
        CEREAL_NVP(az0), CEREAL_NVP(az1), CEREAL_NVP(az2), CEREAL_NVP(bz1),
        CEREAL_NVP(bz2), CEREAL_NVP(cxy11), CEREAL_NVP(cxy12),
        CEREAL_NVP(cxy21), CEREAL_NVP(cxy22), CEREAL_NVP(invs_dx),
        CEREAL_NVP(invs_dy), CEREAL_NVP(invs_dz), CEREAL_NVP(zeta_x),
        CEREAL_NVP(zeta_y), CEREAL_NVP(xm), CEREAL_NVP(ym), CEREAL_NVP(zm));
  }
};

}  // namespace scarabee

// References
// ----------
// [1] P. M. Bokov, D. Botes, R. H. Prinsloo, and D. I. Tomašević, “A
//     Multigroup Homogeneous Flux Reconstruction Method Based on the
//     ANOVA-HDMR Decomposition,” Nucl. Sci. Eng., vol. 197, no. 2,
//     pp. 308–332, 2023, doi: 10.1080/00295639.2022.2108654.

#endif
