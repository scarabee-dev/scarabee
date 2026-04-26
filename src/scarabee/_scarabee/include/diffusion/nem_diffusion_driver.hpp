#ifndef SCARABEE_NEM_DIFFUSION_DRIVER_H
#define SCARABEE_NEM_DIFFUSION_DRIVER_H

#include <diffusion/diffusion_data.hpp>
#include <diffusion/diffusion_geometry.hpp>
#include <utils/serialization.hpp>

#include <Eigen/Dense>
#include <Eigen/LU>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/utility.hpp>

#include <xtensor/containers/xtensor.hpp>

#include <cmath>
#include <memory>
#include <optional>
#include <utility>

namespace scarabee {

class NEMDiffusionDriver {
 public:
  NEMDiffusionDriver(std::shared_ptr<DiffusionGeometry> geom);

  std::shared_ptr<DiffusionGeometry> geometry() const { return geom_; }

  std::size_t ngroups() const { return geom_->ngroups(); }

  void solve();
  bool solved() const { return solved_; }

  double keff_tolerance() const { return keff_tol_; }
  void set_keff_tolerance(double ktol);

  double flux_tolerance() const { return flux_tol_; }
  void set_flux_tolerance(double ftol);

  bool leakage_corrections() const { return leakage_corrections_; }
  void set_leakage_corrections(bool lc) { leakage_corrections_ = lc; }

  double keff() const { return keff_; }

  double flux(double x, double y, double z, std::size_t g) const;
  xt::xtensor<double, 4> flux(const xt::xtensor<double, 1>& x,
                              const xt::xtensor<double, 1>& y,
                              const xt::xtensor<double, 1>& z) const;
  xt::xtensor<double, 4> avg_flux() const;

  double power(double x, double y, double z) const;
  xt::xtensor<double, 3> power(const xt::xtensor<double, 1>& x,
                               const xt::xtensor<double, 1>& y,
                               const xt::xtensor<double, 1>& z) const;
  xt::xtensor<double, 3> avg_power() const;

  void save(const std::string& fname);
  static std::unique_ptr<NEMDiffusionDriver> load(const std::string& fname);

 private:
  //----------------------------------------------------------------------------
  // TYPES
  // Definition of coupling matrix types
  using Current = Eigen::Matrix<double, 6, 1>;
  enum CurrentIndx : ::std::size_t {
    XP = 0,
    XM = 1,
    YP = 2,
    YM = 3,
    ZP = 4,
    ZM = 5
  };

  using MomentsVector = Eigen::Matrix<double, 7, 1>;
  enum MomentIndx : ::std::size_t {
    AVG = 0,
    X1 = 1,
    Y1 = 2,
    Z1 = 3,
    X2 = 4,
    Y2 = 5,
    Z2 = 6
  };

  using Neighbor = DiffusionGeometry::Neighbor;

  using RMat = Eigen::Matrix<double, 6, 6>;
  using PMat = Eigen::Matrix<double, 6, 7>;

  //----------------------------------------------------------------------------
  // PRIVATE MEMBERS
  std::shared_ptr<DiffusionGeometry> geom_;
  std::size_t NG_;  // Number of groups
  std::size_t NM_;  // Number of regions

  // Quantities required for reconstructing the flux  (kept after solution)
  xt::xtensor<double, 3>
      flux_;  // First index is group, second is node, third is moments
  xt::xtensor<Current, 3>
      j_in_out_;  // First index is group, second is node, third is in/out

  // Quantites used for calculation (not needed for reconstruction)
  xt::xtensor<RMat, 2> Rmats_;  // First index is group, second is node
  xt::xtensor<PMat, 2> Pmats_;
  xt::xtensor<MomentsVector, 2> Q_;  // Source

  // Neighbors for each node
  // XP = 0, XN = 1, YP = 2, YN = 3, ZP = 4, ZN = 5
  using NeighborInfo =
      std::pair<DiffusionGeometry::Tile, std::optional<std::size_t>>;
  xt::xtensor<NeighborInfo, 2> neighbors_;

  xt::xtensor<xt::svector<std::size_t>, 1> geom_inds_;
  std::vector<std::shared_ptr<DiffusionCrossSection>>
      mats_;  // These cross sections get modified in solve !
  std::vector<std::shared_ptr<DiffusionData>> diff_datas_;
  xt::xtensor<double, 3> adf_;  // m, group, side

  double keff_ = 1.;
  double kshift_ = 1.;
  double flux_tol_ = 1.E-5;
  double keff_tol_ = 1.E-5;
  bool leakage_corrections_{false};
  bool solved_{false};

  //----------------------------------------------------------------------------
  // PRIVATE METHODS
  void fill_node_coupling_matrices(std::size_t g, std::size_t m,
                                   const DiffusionCrossSection& xs,
                                   const double del_x, const double del_y,
                                   const double del_z);
  void fill_coupling_matrices();
  void fill_mats_adf();
  void fill_source();
  void fill_neighbors_and_geom_inds();
  void update_Jin_from_Jout(std::size_t g, std::size_t m);
  MomentsVector calc_leakage_moments(std::size_t g, std::size_t m) const;
  double calc_keff(double keff, const xt::xtensor<double, 3>& old_flux,
                   const xt::xtensor<double, 3>& new_flux) const;
  double calc_flux_error(const xt::xtensor<double, 3>& old_flux,
                         const xt::xtensor<double, 3>& new_flux) const;
  void calc_node(const std::size_t g, const std::size_t m, const double invs_dx,
                 const double invs_dy, const double invs_dz,
                 const DiffusionCrossSection& xs);
  double calc_avg_node_DB2(const std::size_t g, const std::size_t m,
                           const double dx, const double dy,
                           const double dz) const;
  void update_node_xs_and_matrices();
  void inner_iteration();

  inline double calc_net_current(const Current& Jin, const Current& Jout,
                                 CurrentIndx indx) const {
    if (indx == CurrentIndx::XP || indx == CurrentIndx::YP ||
        indx == CurrentIndx::ZP) {
      return Jout(indx) - Jin(indx);
    } else {
      return Jin(indx) - Jout(indx);
    }
  }

  // The method used for intranodal flux reconstruction is based on ANOVA-HDMR
  // decomposition, as outlined by Bokov et al. [1].
  struct NodeFlux {
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

  xt::xtensor<NodeFlux, 2> recon_params;

  NodeFlux fit_node_recon_params(std::size_t g, std::size_t m) const;
  void fit_node_recon_params_corners(std::size_t g, std::size_t m);

  enum class Corner { PP, PM, MP, MM };
  double eval_heter_xy_corner_flux(std::size_t g, std::size_t m,
                                   Corner c) const;
  double avg_xy_corner_flux(std::size_t g, std::size_t m, Corner c) const;

  friend class cereal::access;
  NEMDiffusionDriver() {}
  template <class Archive>
  void serialize(Archive& arc) {
    arc(CEREAL_NVP(geom_), CEREAL_NVP(NG_), CEREAL_NVP(NM_), CEREAL_NVP(flux_),
        CEREAL_NVP(j_in_out_), CEREAL_NVP(Rmats_), CEREAL_NVP(Pmats_),
        CEREAL_NVP(Q_), CEREAL_NVP(neighbors_), CEREAL_NVP(geom_inds_),
        CEREAL_NVP(mats_), CEREAL_NVP(adf_), CEREAL_NVP(keff_),
        CEREAL_NVP(flux_tol_), CEREAL_NVP(keff_tol_), CEREAL_NVP(solved_),
        CEREAL_NVP(recon_params));
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
