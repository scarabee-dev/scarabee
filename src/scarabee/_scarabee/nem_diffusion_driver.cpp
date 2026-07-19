#include <diffusion/nem_diffusion_driver.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>
#include <utils/timer.hpp>
#include <utils/constants.hpp>

#include <cereal/archives/portable_binary.hpp>

#include <array>
#include <cmath>
#include <fstream>

namespace scarabee {

NEMDiffusionDriver::NEMDiffusionDriver(std::shared_ptr<DiffusionGeometry> geom)
    : geom_(geom), NG_(geom_->ngroups()), NM_(geom_->nmats()) {
  if (geom_ == nullptr) {
    auto mssg = "NEMDiffusionDriver provided with nullptr geometry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (geom_->ndims() != 3) {
    auto mssg = "NEMDiffusionDriver requires a 3D diffusion geometry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  spdlog::warn(
      "NEMDiffusionDriver is deprecated; please use NEM4DiffusionDriver "
      "instead.");

  bool has_leakage_corrections = false;
  for (std::size_t m = 0; m < NM_; m++) {
    const auto geom_indx = geom_->geom_indx(m);
    const auto& diff_data = geom_->mat(geom_indx);
    // Check if we have leakage corrections present at all
    if (diff_data->leakage_corrections()) has_leakage_corrections = true;
  }

  // By default, if there is a node that provides leakage corrections, they
  // will be used in the simulation. The use of leakage corrections can still
  // be disabled after initialization, if desired by the user.
  this->leakage_corrections_ = has_leakage_corrections;
}

void NEMDiffusionDriver::set_flux_tolerance(double ftol) {
  if (ftol <= 0.) {
    auto mssg = "Tolerance for flux must be in the interval (0., 0.1).";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (ftol >= 0.1) {
    auto mssg = "Tolerance for flux must be in the interval (0., 0.1).";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  flux_tol_ = ftol;
}

void NEMDiffusionDriver::set_keff_tolerance(double ktol) {
  if (ktol <= 0.) {
    auto mssg = "Tolerance for keff must be in the interval (0., 0.1).";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (ktol >= 0.1) {
    auto mssg = "Tolerance for keff must be in the interval (0., 0.1).";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  keff_tol_ = ktol;
}

double NEMDiffusionDriver::calc_keff(
    double keff, const xt::xtensor<double, 3>& old_flux,
    const xt::xtensor<double, 3>& new_flux) const {
  double num = 0.;
  double denom = 0.;

  for (std::size_t m = 0; m < NM_; m++) {
    const auto geom_indx = geom_inds_[m];
    const double Vr = geom_->dx(geom_indx[0]) * geom_->dy(geom_indx[1]) *
                      geom_->dz(geom_indx[2]);
    const auto& mat = mats_[m];
    for (std::size_t g = 0; g < NG_; g++) {
      const double VvEf = Vr * mat->vEf(g);
      const double nflx = new_flux(g, m, MomentIndx::AVG);
      const double oflx = old_flux(g, m, MomentIndx::AVG);

      num += VvEf * nflx;
      denom += VvEf * oflx;
    }
  }

  return keff * num / denom;
}

double NEMDiffusionDriver::calc_flux_error(
    const xt::xtensor<double, 3>& old_flux,
    const xt::xtensor<double, 3>& new_flux) const {
  double max_diff = 0.;
  for (std::size_t g = 0; g < NG_; g++) {
    for (std::size_t m = 0; m < NM_; m++) {
      const double rel_diff = std::abs(
          (old_flux(g, m, MomentIndx::AVG) - new_flux(g, m, MomentIndx::AVG)) /
          new_flux(g, m, MomentIndx::AVG));

      if (rel_diff > max_diff) max_diff = rel_diff;
    }
  }

  return max_diff;
}

void NEMDiffusionDriver::fill_node_coupling_matrices(
    std::size_t g, std::size_t m, const DiffusionCrossSection& xs,
    const double del_x, const double del_y, const double del_z) {
  const double D = xs.D(g);
  const double Er = xs.Er(g);
  const double dx = D / del_x;
  const double dy = D / del_y;
  const double dz = D / del_z;
  const double lx = 1. / (Er * del_x);
  const double ly = 1. / (Er * del_y);
  const double lz = 1. / (Er * del_z);

  // Create matrices
  const double ax = 1. + 32. * dx + 120. * dx * lx + 960. * dx * dx * lx +
                    840. * dx * dx * lx * lx;
  const double ay = 1. + 32. * dy + 120. * dy * ly + 960. * dy * dy * ly +
                    840. * dy * dy * ly * ly;
  const double az = 1. + 32. * dz + 120. * dz * lz + 960. * dz * dz * lz +
                    840. * dz * dz * lz * lz;
  const double ax1 =
      8. * dx + 60. * dx * lx + 720. * dx * dx * lx + 840. * dx * dx * lx * lx;
  const double ay1 =
      8. * dy + 60. * dy * ly + 720. * dy * dy * ly + 840. * dy * dy * ly * ly;
  const double az1 =
      8. * dz + 60. * dz * lz + 720. * dz * dz * lz + 840. * dz * dz * lz * lz;
  const double xy = 20. * dx * ly + 840. * dx * dx * lx * ly;
  const double xz = 20. * dx * lz + 840. * dx * dx * lx * lz;
  const double yx = 20. * dy * lx + 840. * dy * dy * ly * lx;
  const double yz = 20. * dy * lz + 840. * dy * dy * ly * lz;
  const double zx = 20. * dz * lx + 840. * dz * dz * lz * lx;
  const double zy = 20. * dz * ly + 840. * dz * dz * lz * ly;

  Eigen::Matrix<double, 6, 6> A{
      {ax, ax1, xy, xy, xz, xz}, {ax1, ax, xy, xy, xz, xz},
      {yx, yx, ay, ay1, yz, yz}, {yx, yx, ay1, ay, yz, yz},
      {zx, zx, zy, zy, az, az1}, {zx, zx, zy, zy, az1, az}};

  const double bx = 1. - 32. * dx + 120. * dx * lx - 960. * dx * dx * lx +
                    840. * dx * dx * lx * lx;
  const double by = 1. - 32. * dy + 120. * dy * ly - 960. * dy * dy * ly +
                    840. * dy * dy * ly * ly;
  const double bz = 1. - 32. * dz + 120. * dz * lz - 960. * dz * dz * lz +
                    840. * dz * dz * lz * lz;
  const double bx1 =
      -8. * dx + 60. * dx * lx - 720. * dx * dx * lx + 840. * dx * dx * lx * lx;
  const double by1 =
      -8. * dy + 60. * dy * ly - 720. * dy * dy * ly + 840. * dy * dy * ly * ly;
  const double bz1 =
      -8. * dz + 60. * dz * lz - 720. * dz * dz * lz + 840. * dz * dz * lz * lz;

  Eigen::Matrix<double, 6, 6> B{
      {bx, bx1, xy, xy, xz, xz}, {bx1, bx, xy, xy, xz, xz},
      {yx, yx, by, by1, yz, yz}, {yx, yx, by1, by, yz, yz},
      {zx, zx, zy, zy, bz, bz1}, {zx, zx, zy, zy, bz1, bz}};

  const double cx = 20. * dx * lx * del_x + 840. * dx * dx * lx * lx * del_x;
  const double cy = 20. * dy * ly * del_y + 840. * dy * dy * ly * ly * del_y;
  const double cz = 20. * dz * lz * del_z + 840. * dz * dz * lz * lz * del_z;
  const double cx1 = 60. * dx * lx * del_x;
  const double cy1 = 60. * dy * ly * del_y;
  const double cz1 = 60. * dz * lz * del_z;
  const double cx2 = 140. * dx * lx * del_x;
  const double cy2 = 140. * dy * ly * del_y;
  const double cz2 = 140. * dz * lz * del_z;

  Eigen::Matrix<double, 6, 7> C{
      {cx, cx1, 0., 0., cx2, 0., 0.}, {cx, -cx1, 0., 0., cx2, 0., 0.},
      {cy, 0., cy1, 0., 0., cy2, 0.}, {cy, 0., -cy1, 0., 0., cy2, 0.},
      {cz, 0., 0., cz1, 0., 0., cz2}, {cz, 0., 0., -cz1, 0., 0., cz2}};

  auto Ainvs = A.inverse();
  Rmats_(g, m) = Ainvs * B;
  Pmats_(g, m) = Ainvs * C;
}

void NEMDiffusionDriver::fill_coupling_matrices() {
  spdlog::info("Loading coupling matrices");

  for (std::size_t m = 0; m < NM_; m++) {
    const auto geom_indx = geom_inds_(m);
    const double del_x = geom_->dx(geom_indx[0]);
    const double del_y = geom_->dy(geom_indx[1]);
    const double del_z = geom_->dz(geom_indx[2]);
    const auto& xs = *mats_[m];

    for (std::size_t g = 0; g < NG_; g++) {
      this->fill_node_coupling_matrices(g, m, xs, del_x, del_y, del_z);
    }
  }
}

void NEMDiffusionDriver::fill_mats_adf() {
  // Save all material cross sections
  diff_datas_.clear();
  mats_.clear();
  mats_.reserve(NM_);
  diff_datas_.reserve(NM_);
  for (std::size_t m = 0; m < NM_; m++) {
    const auto geom_indx = geom_inds_[m];
    const auto& diff_data = geom_->mat(geom_indx);
    const auto& xs_ptr = diff_data->xs();
    // Save hard copy of cross section that can be modified !
    mats_.push_back(std::make_shared<DiffusionCrossSection>(*xs_ptr));
    // We don't modify the DiffusionData !
    diff_datas_.push_back(diff_data);
  }

  // Save all node ADFs
  adf_ = xt::zeros<double>({NM_, NG_, static_cast<std::size_t>(6)});
  for (std::size_t m = 0; m < NM_; m++) {
    for (std::size_t g = 0; g < NG_; g++) {
      adf_(m, g, DiffusionData::ADF::XN) = geom_->adf_xn(m, g);
      adf_(m, g, DiffusionData::ADF::XP) = geom_->adf_xp(m, g);
      adf_(m, g, DiffusionData::ADF::YN) = geom_->adf_yn(m, g);
      adf_(m, g, DiffusionData::ADF::YP) = geom_->adf_yp(m, g);
      adf_(m, g, DiffusionData::ADF::ZN) = geom_->adf_zn(m, g);
      adf_(m, g, DiffusionData::ADF::ZP) = geom_->adf_zp(m, g);
    }
  }
}

void NEMDiffusionDriver::fill_source() {
  for (std::size_t m = 0; m < NM_; m++) {
    const auto& mat = mats_[m];
    for (std::size_t g = 0; g < NG_; g++) {
      auto& Q = Q_(g, m);
      Q.fill(0.);

      const double chi_g = mat->chi(g);
      const double invs_keff = 1. / keff_;

      for (std::size_t gg = 0; gg < NG_; gg++) {
        const double vEf_gg = mat->vEf(gg);
        const double Es_gg_g = mat->Es(gg, g);

        // i is the moment index (Avg, X1, Y1, Z1, X2, Y2, Z2)
        for (int mi = 0; mi < 7; mi++) {
          const double flx = flux_(gg, m, static_cast<std::size_t>(mi));
          Q(mi) += invs_keff * chi_g * vEf_gg * flx;
          if (gg != g) Q(mi) += Es_gg_g * flx;
        }
      }
    }
  }
}

void NEMDiffusionDriver::fill_neighbors_and_geom_inds() {
  neighbors_.resize({NM_, 6});
  geom_inds_.resize({NM_});

  // Go through all mats
  for (std::size_t m = 0; m < NM_; m++) {
    geom_inds_(m) = geom_->geom_indx(m);

    neighbors_(m, Neighbor::XP) = geom_->neighbor(m, Neighbor::XP);
    neighbors_(m, Neighbor::XN) = geom_->neighbor(m, Neighbor::XN);
    neighbors_(m, Neighbor::YP) = geom_->neighbor(m, Neighbor::YP);
    neighbors_(m, Neighbor::YN) = geom_->neighbor(m, Neighbor::YN);
    neighbors_(m, Neighbor::ZP) = geom_->neighbor(m, Neighbor::ZP);
    neighbors_(m, Neighbor::ZN) = geom_->neighbor(m, Neighbor::ZN);
  }
}

void NEMDiffusionDriver::update_Jin_from_Jout(std::size_t g, std::size_t m) {
  // Get neighbor info
  const auto& n_xp = neighbors_(m, Neighbor::XP);
  const auto& n_xm = neighbors_(m, Neighbor::XN);
  const auto& n_yp = neighbors_(m, Neighbor::YP);
  const auto& n_ym = neighbors_(m, Neighbor::YN);
  const auto& n_zp = neighbors_(m, Neighbor::ZP);
  const auto& n_zm = neighbors_(m, Neighbor::ZN);

  // UPDATE INCOMING CURRENTS IN NEIGHBORING NODES / B.C.
  // x+ surface
  if (n_xp.second) {
    const double a =
        0.5 * (1. - (adf_(n_xp.second.value(), g, DiffusionData::ADF::XN) /
                     adf_(m, g, DiffusionData::ADF::XP)));

    j_in_out_(g, n_xp.second.value(), 0)(CurrentIndx::XM) =
        (1. / (1. - a)) *
        (j_in_out_(g, m, 1)(CurrentIndx::XP) +
         a * j_in_out_(g, n_xp.second.value(), 1)(CurrentIndx::XM));
  } else {
    const double albedo = n_xp.first.albedo.value();
    j_in_out_(g, m, 0)(CurrentIndx::XP) =
        albedo * j_in_out_(g, m, 1)(CurrentIndx::XP);
  }

  // x- surface
  if (n_xm.second) {
    const double a =
        0.5 * (1. - (adf_(n_xm.second.value(), g, DiffusionData::ADF::XP) /
                     adf_(m, g, DiffusionData::ADF::XN)));

    j_in_out_(g, n_xm.second.value(), 0)(CurrentIndx::XP) =
        (1. / (1. - a)) *
        (j_in_out_(g, m, 1)(CurrentIndx::XM) +
         a * j_in_out_(g, n_xm.second.value(), 1)(CurrentIndx::XP));
  } else {
    const double albedo = n_xm.first.albedo.value();
    j_in_out_(g, m, 0)(CurrentIndx::XM) =
        albedo * j_in_out_(g, m, 1)(CurrentIndx::XM);
  }

  // y+ surface
  if (n_yp.second) {
    const double a =
        0.5 * (1. - (adf_(n_yp.second.value(), g, DiffusionData::ADF::YN) /
                     adf_(m, g, DiffusionData::ADF::YP)));

    j_in_out_(g, n_yp.second.value(), 0)(CurrentIndx::YM) =
        (1. / (1. - a)) *
        (j_in_out_(g, m, 1)(CurrentIndx::YP) +
         a * j_in_out_(g, n_yp.second.value(), 1)(CurrentIndx::YM));
  } else {
    const double albedo = n_yp.first.albedo.value();
    j_in_out_(g, m, 0)(CurrentIndx::YP) =
        albedo * j_in_out_(g, m, 1)(CurrentIndx::YP);
  }

  // y- surface
  if (n_ym.second) {
    const double a =
        0.5 * (1. - (adf_(n_ym.second.value(), g, DiffusionData::ADF::YP) /
                     adf_(m, g, DiffusionData::ADF::YN)));

    j_in_out_(g, n_ym.second.value(), 0)(CurrentIndx::YP) =
        (1. / (1. - a)) *
        (j_in_out_(g, m, 1)(CurrentIndx::YM) +
         a * j_in_out_(g, n_ym.second.value(), 1)(CurrentIndx::YP));
  } else {
    const double albedo = n_ym.first.albedo.value();
    j_in_out_(g, m, 0)(CurrentIndx::YM) =
        albedo * j_in_out_(g, m, 1)(CurrentIndx::YM);
  }

  // z+ surface
  if (n_zp.second) {
    const double a =
        0.5 * (1. - (adf_(n_zp.second.value(), g, DiffusionData::ADF::ZN) /
                     adf_(m, g, DiffusionData::ADF::ZP)));

    j_in_out_(g, n_zp.second.value(), 0)(CurrentIndx::ZM) =
        (1. / (1. - a)) *
        (j_in_out_(g, m, 1)(CurrentIndx::ZP) +
         a * j_in_out_(g, n_zp.second.value(), 1)(CurrentIndx::ZM));
  } else {
    const double albedo = n_zp.first.albedo.value();
    j_in_out_(g, m, 0)(CurrentIndx::ZP) =
        albedo * j_in_out_(g, m, 1)(CurrentIndx::ZP);
  }

  // z- surface
  if (n_zm.second) {
    const double a =
        0.5 * (1. - (adf_(n_zm.second.value(), g, DiffusionData::ADF::ZP) /
                     adf_(m, g, DiffusionData::ADF::ZN)));

    j_in_out_(g, n_zm.second.value(), 0)(CurrentIndx::ZP) =
        (1. / (1. - a)) *
        (j_in_out_(g, m, 1)(CurrentIndx::ZM) +
         a * j_in_out_(g, n_zm.second.value(), 1)(CurrentIndx::ZP));
  } else {
    const double albedo = n_zm.first.albedo.value();
    j_in_out_(g, m, 0)(CurrentIndx::ZM) =
        albedo * j_in_out_(g, m, 1)(CurrentIndx::ZM);
  }
}

NEMDiffusionDriver::MomentsVector NEMDiffusionDriver::calc_leakage_moments(
    std::size_t g, std::size_t m) const {
  const auto& Jin = j_in_out_(g, m, 0);
  const auto& Jout = j_in_out_(g, m, 1);

  // Get the currents along each axis at the positive and negative bounds
  const double Jxp = calc_net_current(Jin, Jout, CurrentIndx::XP);
  const double Jxm = calc_net_current(Jin, Jout, CurrentIndx::XM);
  const double Jyp = calc_net_current(Jin, Jout, CurrentIndx::YP);
  const double Jym = calc_net_current(Jin, Jout, CurrentIndx::YM);
  const double Jzp = calc_net_current(Jin, Jout, CurrentIndx::ZP);
  const double Jzm = calc_net_current(Jin, Jout, CurrentIndx::ZM);

  // Compute average transverse leakages in each direction
  const double Lx = Jxp - Jxm;
  const double Ly = Jyp - Jym;
  const double Lz = Jzp - Jzm;

  // Get neighbor info
  const auto& n_xp = neighbors_(m, Neighbor::XP);
  const auto& n_xm = neighbors_(m, Neighbor::XN);
  const auto& n_yp = neighbors_(m, Neighbor::YP);
  const auto& n_ym = neighbors_(m, Neighbor::YN);
  const auto& n_zp = neighbors_(m, Neighbor::ZP);
  const auto& n_zm = neighbors_(m, Neighbor::ZN);

  // Obtain geometry spacings for node
  const auto geom_indxs = geom_inds_(m);
  const double dx = geom_->dx(geom_indxs[0]);
  const double dy = geom_->dy(geom_indxs[1]);
  const double dz = geom_->dz(geom_indxs[2]);
  const double invs_dx = 1. / dx;
  const double invs_dy = 1. / dy;
  const double invs_dz = 1. / dz;

  constexpr double invs_12 = 1. / 12.;
  constexpr double invs_20 = 1. / 20.;

  // This returns the average transverse leakage moments for a given node
  auto comp_avg_trans_lks = [this](std::size_t g, std::size_t m) {
    const auto& Jin = j_in_out_(g, m, 0);
    const auto& Jout = j_in_out_(g, m, 1);

    const double Jxp = calc_net_current(Jin, Jout, CurrentIndx::XP);
    const double Jxm = calc_net_current(Jin, Jout, CurrentIndx::XM);
    const double Jyp = calc_net_current(Jin, Jout, CurrentIndx::YP);
    const double Jym = calc_net_current(Jin, Jout, CurrentIndx::YM);
    const double Jzp = calc_net_current(Jin, Jout, CurrentIndx::ZP);
    const double Jzm = calc_net_current(Jin, Jout, CurrentIndx::ZM);

    const double Lx = Jxp - Jxm;
    const double Ly = Jyp - Jym;
    const double Lz = Jzp - Jzm;

    return std::array<double, 3>{Lx, Ly, Lz};
  };

  // Compute net moments
  MomentsVector L;
  L.fill(0.);

  std::array<double, 3> tmp;
  // x-axis
  if (n_xp.second && n_xm.second) {
    const double dx_xp = geom_->dx(geom_indxs[0] + 1);
    const double dx_xm = geom_->dx(geom_indxs[0] - 1);
    const double eta_xp = dx_xp * invs_dx;
    const double eta_xm = dx_xm * invs_dx;
    const double p1xm = eta_xm + 1.;
    const double p2xm = 2. * eta_xm + 1.;
    const double p1xp = eta_xp + 1.;
    const double p2xp = 2. * eta_xp + 1.;
    const double invs_denom = 1. / (p1xp * p1xm * (eta_xp + eta_xm + 1.));

    tmp = comp_avg_trans_lks(g, n_xp.second.value());
    const double Ly_xp = tmp[1];
    const double Lz_xp = tmp[2];

    tmp = comp_avg_trans_lks(g, n_xm.second.value());
    const double Ly_xm = tmp[1];
    const double Lz_xm = tmp[2];

    const double rho1yx = (p1xm * p2xm * Ly_xp - p1xp * p2xp * Ly_xm +
                           (p1xp * p2xp - p1xm * p2xm) * Ly) *
                          invs_denom;
    const double rho2yx =
        (p1xm * Ly_xp + p1xp * Ly_xm - (eta_xp + eta_xm + 2.) * Ly) *
        invs_denom;
    const double rho1zx = (p1xm * p2xm * Lz_xp - p1xp * p2xp * Lz_xm +
                           (p1xp * p2xp - p1xm * p2xm) * Lz) *
                          invs_denom;
    const double rho2zx =
        (p1xm * Lz_xp + p1xp * Lz_xm - (eta_xp + eta_xm + 2.) * Lz) *
        invs_denom;

    const double Lyx1 = invs_12 * rho1yx;
    const double Lyx2 = invs_20 * rho2yx;
    const double Lzx1 = invs_12 * rho1zx;
    const double Lzx2 = invs_20 * rho2zx;

    L(MomentIndx::X1) = invs_dy * Lyx1 + invs_dz * Lzx1;
    L(MomentIndx::X2) = invs_dy * Lyx2 + invs_dz * Lzx2;
  }

  // y-axis
  if (n_yp.second && n_ym.second) {
    const double dy_yp = geom_->dy(geom_indxs[1] + 1);
    const double dy_ym = geom_->dy(geom_indxs[1] - 1);
    const double eta_yp = dy_yp * invs_dy;
    const double eta_ym = dy_ym * invs_dy;
    const double p1ym = eta_ym + 1.;
    const double p2ym = 2. * eta_ym + 1.;
    const double p1yp = eta_yp + 1.;
    const double p2yp = 2. * eta_yp + 1.;
    const double invs_denom = 1. / (p1yp * p1ym * (eta_yp + eta_ym + 1.));

    tmp = comp_avg_trans_lks(g, n_yp.second.value());
    const double Lx_yp = tmp[0];
    const double Lz_yp = tmp[2];

    tmp = comp_avg_trans_lks(g, n_ym.second.value());
    const double Lx_ym = tmp[0];
    const double Lz_ym = tmp[2];

    const double rho1xy = (p1ym * p2ym * Lx_yp - p1yp * p2yp * Lx_ym +
                           (p1yp * p2yp - p1ym * p2ym) * Lx) *
                          invs_denom;
    const double rho2xy =
        (p1ym * Lx_yp + p1yp * Lx_ym - (eta_yp + eta_ym + 2.) * Lx) *
        invs_denom;
    const double rho1zy = (p1ym * p2ym * Lz_yp - p1yp * p2yp * Lz_ym +
                           (p1yp * p2yp - p1ym * p2ym) * Lz) *
                          invs_denom;
    const double rho2zy =
        (p1ym * Lz_yp + p1yp * Lz_ym - (eta_yp + eta_ym + 2.) * Lz) *
        invs_denom;

    const double Lxy1 = invs_12 * rho1xy;
    const double Lxy2 = invs_20 * rho2xy;
    const double Lzy1 = invs_12 * rho1zy;
    const double Lzy2 = invs_20 * rho2zy;

    L(MomentIndx::Y1) = invs_dx * Lxy1 + invs_dz * Lzy1;
    L(MomentIndx::Y2) = invs_dx * Lxy2 + invs_dz * Lzy2;
  }

  // z-axis
  if (n_zp.second && n_zm.second) {
    const double dz_zp = geom_->dz(geom_indxs[2] + 1);
    const double dz_zm = geom_->dz(geom_indxs[2] - 1);
    const double eta_zp = dz_zp * invs_dz;
    const double eta_zm = dz_zm * invs_dz;
    const double p1zm = eta_zm + 1.;
    const double p2zm = 2. * eta_zm + 1.;
    const double p1zp = eta_zp + 1.;
    const double p2zp = 2. * eta_zp + 1.;
    const double invs_denom = 1. / (p1zp * p1zm * (eta_zp + eta_zm + 1.));

    tmp = comp_avg_trans_lks(g, n_zp.second.value());
    const double Lx_zp = tmp[0];
    const double Ly_zp = tmp[1];

    tmp = comp_avg_trans_lks(g, n_zm.second.value());
    const double Lx_zm = tmp[0];
    const double Ly_zm = tmp[1];

    const double rho1xz = (p1zm * p2zm * Lx_zp - p1zp * p2zp * Lx_zm +
                           (p1zp * p2zp - p1zm * p2zm) * Lx) *
                          invs_denom;
    const double rho2xz =
        (p1zm * Lx_zp + p1zp * Lx_zm - (eta_zp + eta_zm + 2.) * Lx) *
        invs_denom;
    const double rho1yz = (p1zm * p2zm * Ly_zp - p1zp * p2zp * Ly_zm +
                           (p1zp * p2zp - p1zm * p2zm) * Ly) *
                          invs_denom;
    const double rho2yz =
        (p1zm * Ly_zp + p1zp * Ly_zm - (eta_zp + eta_zm + 2.) * Ly) *
        invs_denom;

    const double Lxz1 = invs_12 * rho1xz;
    const double Lxz2 = invs_20 * rho2xz;
    const double Lyz1 = invs_12 * rho1yz;
    const double Lyz2 = invs_20 * rho2yz;

    L(MomentIndx::Z1) = invs_dx * Lxz1 + invs_dy * Lyz1;
    L(MomentIndx::Z2) = invs_dx * Lxz2 + invs_dy * Lyz2;
  }

  // Return the transverse leakage moments vector
  return L;
}

void NEMDiffusionDriver::calc_node(const std::size_t g, const std::size_t m,
                                   const double invs_dx, const double invs_dy,
                                   const double invs_dz,
                                   const DiffusionCrossSection& xs) {
  //----------------------------------------------------------------------------
  // OBTAIN NECESSARY ARRAYS AND DATA
  const auto& Q = Q_(g, m);
  const auto& R = Rmats_(g, m);
  const auto& P = Pmats_(g, m);
  auto& Jin = j_in_out_(g, m, 0);  // Not const as we update this here
  auto& Jout = j_in_out_(g, m, 1);
  const double D = xs.D(g);    // Diffusion coefficient
  const double Er = xs.Er(g);  // Removal cross section
  const double invs_Er = 1. / Er;

  //----------------------------------------------------------------------------
  // CALCULATE TRANSVERSE LEAKAGES AND THEIR MOMENTS

  // Get the currents along each axis at the positive and negative bounds
  double Jxp = calc_net_current(Jin, Jout, CurrentIndx::XP);
  double Jxm = calc_net_current(Jin, Jout, CurrentIndx::XM);
  double Jyp = calc_net_current(Jin, Jout, CurrentIndx::YP);
  double Jym = calc_net_current(Jin, Jout, CurrentIndx::YM);
  double Jzp = calc_net_current(Jin, Jout, CurrentIndx::ZP);
  double Jzm = calc_net_current(Jin, Jout, CurrentIndx::ZM);

  // Compute the transverse leakage moments vector
  MomentsVector L = calc_leakage_moments(g, m);

  //----------------------------------------------------------------------------
  // UPDATE OUTGOING PARTIAL CURRENTS
  Jout = R * Jin + P * (Q - L);

  //----------------------------------------------------------------------------
  // UPDATE FLUX AVERAGE AND MOMENTS
  // Get the currents along each axis at the positive and negative bounds
  Jxp = calc_net_current(Jin, Jout, CurrentIndx::XP);
  Jxm = calc_net_current(Jin, Jout, CurrentIndx::XM);
  Jyp = calc_net_current(Jin, Jout, CurrentIndx::YP);
  Jym = calc_net_current(Jin, Jout, CurrentIndx::YM);
  Jzp = calc_net_current(Jin, Jout, CurrentIndx::ZP);
  Jzm = calc_net_current(Jin, Jout, CurrentIndx::ZM);

  // Compute average transverse leakages in each direction
  const double Lx = Jxp - Jxm;
  const double Ly = Jyp - Jym;
  const double Lz = Jzp - Jzm;

  // Compute average T in each direction
  const double Tx = Jxp + Jxm;
  const double Ty = Jyp + Jym;
  const double Tz = Jzp + Jzm;

  // From the partial currents, compute the surface fluxes
  const double flx_xp = 2. * (Jout(CurrentIndx::XP) + Jin(CurrentIndx::XP));
  const double flx_xm = 2. * (Jout(CurrentIndx::XM) + Jin(CurrentIndx::XM));
  const double flx_yp = 2. * (Jout(CurrentIndx::YP) + Jin(CurrentIndx::YP));
  const double flx_ym = 2. * (Jout(CurrentIndx::YM) + Jin(CurrentIndx::YM));
  const double flx_zp = 2. * (Jout(CurrentIndx::ZP) + Jin(CurrentIndx::ZP));
  const double flx_zm = 2. * (Jout(CurrentIndx::ZM) + Jin(CurrentIndx::ZM));

  // Compute the new average flux
  const double flx_avg =
      (Q(MomentIndx::AVG) - invs_dx * Lx - invs_dy * Ly - invs_dz * Lz) / Er;
  flux_(g, m, MomentIndx::AVG) = flx_avg;

  // Calculate the first two polynomial coefficients along each direction
  const double ax1 = flx_xp - flx_xm;
  const double ax2 = flx_xp + flx_xm - 2. * flx_avg;
  const double ay1 = flx_yp - flx_ym;
  const double ay2 = flx_yp + flx_ym - 2. * flx_avg;
  const double az1 = flx_zp - flx_zm;
  const double az2 = flx_zp + flx_zm - 2. * flx_avg;

  // Calculate the first flux moments
  flux_(g, m, MomentIndx::X1) =
      (Q(MomentIndx::X1) - L(MomentIndx::X1) - 0.5 * invs_dx * Tx -
       invs_dx * invs_dx * D * ax1) *
      invs_Er;

  flux_(g, m, MomentIndx::Y1) =
      (Q(MomentIndx::Y1) - L(MomentIndx::Y1) - 0.5 * invs_dy * Ty -
       invs_dy * invs_dy * D * ay1) *
      invs_Er;

  flux_(g, m, MomentIndx::Z1) =
      (Q(MomentIndx::Z1) - L(MomentIndx::Z1) - 0.5 * invs_dz * Tz -
       invs_dz * invs_dz * D * az1) *
      invs_Er;

  // Calculate the second flux moments
  flux_(g, m, MomentIndx::X2) =
      (Q(MomentIndx::X2) - L(MomentIndx::X2) - 0.5 * invs_dx * Lx -
       3. * invs_dx * invs_dx * D * ax2) *
      invs_Er;

  flux_(g, m, MomentIndx::Y2) =
      (Q(MomentIndx::Y2) - L(MomentIndx::Y2) - 0.5 * invs_dy * Ly -
       3. * invs_dy * invs_dy * D * ay2) *
      invs_Er;

  flux_(g, m, MomentIndx::Z2) =
      (Q(MomentIndx::Z2) - L(MomentIndx::Z2) - 0.5 * invs_dz * Lz -
       3. * invs_dz * invs_dz * D * az2) *
      invs_Er;

  //----------------------------------------------------------------------------
  // UPDATE INCOMING CURRENTS IN NEIGHBORING NODES / B.C.
  update_Jin_from_Jout(g, m);
}

double NEMDiffusionDriver::calc_avg_node_DB2(const std::size_t g,
                                             const std::size_t m,
                                             const double dx, const double dy,
                                             const double dz) const {
  // For the given node, grab the partial currents
  const auto& Jin = j_in_out_(g, m, 0);
  const auto& Jout = j_in_out_(g, m, 1);
  const double flx_avg = flux_(g, m, MomentIndx::AVG);

  double DB2 = 0.;
  DB2 += dy * dz *
         (-calc_net_current(Jin, Jout, CurrentIndx::XM) +
          calc_net_current(Jin, Jout, CurrentIndx::XP));
  DB2 += dx * dz *
         (-calc_net_current(Jin, Jout, CurrentIndx::YM) +
          calc_net_current(Jin, Jout, CurrentIndx::YP));
  DB2 += dx * dy *
         (-calc_net_current(Jin, Jout, CurrentIndx::ZM) +
          calc_net_current(Jin, Jout, CurrentIndx::ZP));
  DB2 /= flx_avg * dx * dy * dz;

  return DB2;
}

void NEMDiffusionDriver::update_node_xs_and_matrices() {
  // Go through all nodes and groups, updating the cross sections
  for (std::size_t m = 0; m < NM_; m++) {
    const auto& dd = *diff_datas_[m];
    if (dd.leakage_corrections().has_value() == false || dd.reflector())
      continue;
    const auto& lc = dd.leakage_corrections().value();
    const auto& sa_xs = *dd.xs();  // Single-Assembly (un-buckled) xs

    const auto geom_indx = geom_inds_(m);
    const double del_x = geom_->dx(geom_indx[0]);
    const double del_y = geom_->dy(geom_indx[1]);
    const double del_z = geom_->dz(geom_indx[2]);
    auto& xs = *mats_[m];

    for (std::size_t g_in = 0; g_in < NG_; g_in++) {
      // Compute node/group leakage to loss ratio
      const double DB2 = this->calc_avg_node_DB2(g_in, m, del_x, del_y, del_z);
      const double LRr = DB2 / xs.Er(g_in);

      // Can now determine fractional change in each group cross section
      const double fD = lc.D(g_in) * LRr;
      const double fEa = lc.Ea(g_in) * LRr;
      const double fEf = lc.Ef(g_in) * LRr;
      const double fvEf = lc.vEf(g_in) * LRr;

      // Update the cross sections
      xs.D_ref(g_in) = (fD + 1.) * sa_xs.D(g_in);
      xs.Ea_ref(g_in) = (fEa + 1.) * sa_xs.Ea(g_in);
      xs.Ef_ref(g_in) = (fEf + 1.) * sa_xs.Ef(g_in);
      xs.vEf_ref(g_in) = (fvEf + 1.) * sa_xs.vEf(g_in);

      for (std::size_t g_out = g_in + 1; g_out < NG_; g_out++) {
        const double fEs = lc.Es(g_in, g_out) * LRr;
        xs.Es_ref(g_in, g_out) = (fEs + 1.) * sa_xs.Es(g_in, g_out);
      }

      this->fill_node_coupling_matrices(g_in, m, xs, del_x, del_y, del_z);
    }
  }
}

void NEMDiffusionDriver::inner_iteration() {
  // Iterate through all nodes
  for (std::size_t m = 0; m < NM_; m++) {
    const auto geom_indx = geom_inds_(m);
    const double dx = geom_->dx(geom_indx[0]);
    const double dy = geom_->dy(geom_indx[1]);
    const double dz = geom_->dz(geom_indx[2]);
    const auto& xs = *mats_[m];
    const double invs_dx = 1. / dx;
    const double invs_dy = 1. / dy;
    const double invs_dz = 1. / dz;

    for (std::size_t g = 0; g < NG_; g++) {
      calc_node(g, m, invs_dx, invs_dy, invs_dz, xs);
    }
  }
}

void NEMDiffusionDriver::solve() {
  Timer sim_timer;
  sim_timer.start();

  spdlog::info("Solving for keff.");
  spdlog::info("keff tolerance: {:.5E}", keff_tol_);
  spdlog::info("Flux tolerance: {:.5E}", flux_tol_);

  // Allocate all arrays
  flux_.resize({NG_, NM_, 7});
  xt::xtensor<double, 3> old_flux = flux_;
  j_in_out_.resize({NG_, NM_, 2});
  Rmats_.resize({NG_, NM_});
  Pmats_.resize({NG_, NM_});
  Q_.resize({NG_, NM_});

  // Load the flux and current values with an initial guess
  flux_.fill(1.);
  old_flux.fill(1.);
  for (std::size_t g = 0; g < NG_; g++) {
    for (std::size_t m = 0; m < NM_; m++) {
      j_in_out_(g, m, 0).fill(1.);
      j_in_out_(g, m, 1).fill(1.);
    }
  }

  // Fill the coupling matrices
  fill_neighbors_and_geom_inds();
  fill_mats_adf();
  fill_coupling_matrices();

  // Begin power iteration
  double keff_diff = 100.;
  double flux_diff = 100.;
  std::size_t iteration = 0;
  Timer iteration_timer;
  while (keff_diff > keff_tol_ || flux_diff > flux_tol_) {
    iteration_timer.reset();
    iteration_timer.start();
    iteration++;

    // Copy current flux into old flux
    old_flux = flux_;

    // Calculating the source
    fill_source();

    // Perform 2 inner iterations per outer generation
    inner_iteration();
    inner_iteration();

    // Compute new keff
    double prev_keff = keff_;
    keff_ = calc_keff(prev_keff, old_flux, flux_);
    keff_diff = std::abs(keff_ - prev_keff) / keff_;

    // Find the max flux error
    // flux_diff = xt::amax(xt::abs(flux_ - old_flux) / flux_)();
    flux_diff = calc_flux_error(old_flux, flux_);

    // Write information
    spdlog::info("-------------------------------------");
    spdlog::info("Iteration {:>4d}          keff: {:.5f}", iteration, keff_);
    spdlog::info("     keff difference:     {:.5E}", keff_diff);
    spdlog::info("     max flux difference: {:.5E}", flux_diff);
    spdlog::info("     iteration time: {:.5E} s",
                 iteration_timer.elapsed_time());

    if (leakage_corrections() && iteration % 50 == 0) {
      spdlog::info("-------------------------------------");
      spdlog::info("");
      spdlog::info("Updating cross sections and coupling matrices");
      spdlog::info("");
      this->update_node_xs_and_matrices();
    }
  }

  solved_ = true;

  sim_timer.stop();
  spdlog::info("");
  spdlog::info("Simulation Time: {:.5E} s", sim_timer.elapsed_time());

  // Deallocate arrays that are not needed for reconstruction
  Rmats_.resize({0, 0});
  Pmats_.resize({0, 0});
  Q_.resize({0, 0});

  // Before reconstruction, we want to compute the average power in each node,
  // and then normalize by that.
  const auto node_powers = this->avg_power();
  double power_sum = 0.;
  double nonzero_node_cntr = 0.;
  for (std::size_t n = 0; n < node_powers.size(); n++) {
    if (node_powers.flat(n) > 0.) {
      power_sum += node_powers.flat(n);
      nonzero_node_cntr += 1.;
    }
  }
  const double avg_node_pwr = power_sum / nonzero_node_cntr;
  flux_ /= avg_node_pwr;
  j_in_out_ /= avg_node_pwr;

  // Calculate flux reconstruction parameters for each node
  Timer fitting_timer;
  fitting_timer.start();
  spdlog::info("Fitting flux reconstruction parameters");
  recon_params.resize({NG_, NM_});
#pragma omp parallel for
  for (int im = 0; im < static_cast<int>(NM_); im++) {
    std::size_t m = static_cast<std::size_t>(im);
    for (std::size_t g = 0; g < NG_; g++) {
      recon_params(g, m) = fit_node_recon_params(g, m);
    }
  }
#pragma omp parallel for
  for (int im = 0; im < static_cast<int>(NM_); im++) {
    std::size_t m = static_cast<std::size_t>(im);
    for (std::size_t g = 0; g < NG_; g++) {
      fit_node_recon_params_corners(g, m);
    }
  }
  fitting_timer.stop();
  spdlog::info("Fitting Time: {:.5E} s", fitting_timer.elapsed_time());
}

double NEMDiffusionDriver::flux(double x, double y, double z,
                                std::size_t g) const {
  // If problem isn't solved yet, we error
  if (solved_ == false) {
    auto mssg = "Cannot compute flux. Problem has not been solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Check group index
  if (g >= ngroups()) {
    std::stringstream mssg;
    mssg << "Group index g = " << g << " is out of range.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // Get geometry index
  const auto oi = geom_->x_to_i(x);
  const auto oj = geom_->y_to_j(y);
  const auto ok = geom_->z_to_k(z);
  if (oi.has_value() == false || oj.has_value() == false ||
      ok.has_value() == false)
    return 0.;
  const std::size_t i = oi.value();
  const std::size_t j = oj.value();
  const std::size_t k = ok.value();
  const xt::svector<std::size_t> geom_inds{i, j, k};

  // Get material index
  const auto om = geom_->geom_to_mat_indx(geom_inds);
  if (om.has_value() == false) return 0.;
  const std::size_t m = om.value();

  return recon_params(g, m)(x, y, z);
}

xt::xtensor<double, 4> NEMDiffusionDriver::flux(
    const xt::xtensor<double, 1>& x, const xt::xtensor<double, 1>& y,
    const xt::xtensor<double, 1>& z) const {
  // If problem isn't solved yet, we error
  if (solved_ == false) {
    auto mssg = "Cannot compute flux. Problem has not been solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure x, y, and z have at least 1 coordinate
  if (x.size() == 0) {
    auto mssg = "Array of x coordinates must have at least one entry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  if (y.size() == 0) {
    auto mssg = "Array of y coordinates must have at least one entry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  if (z.size() == 0) {
    auto mssg = "Array of z coordinates must have at least one entry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  xt::xtensor<double, 4> flux_out;
  flux_out.resize({ngroups(), x.size(), y.size(), z.size()});
  flux_out.fill(0.);

  for (std::size_t g = 0; g < ngroups(); g++) {
#pragma omp parallel for
    for (int ii = 0; ii < static_cast<int>(x.size()); ii++) {
      std::size_t i = static_cast<std::size_t>(ii);
      for (std::size_t j = 0; j < y.size(); j++) {
        for (std::size_t k = 0; k < z.size(); k++) {
          // Get geometry index
          const auto oi = geom_->x_to_i(x[i]);
          const auto oj = geom_->y_to_j(y[j]);
          const auto ok = geom_->z_to_k(z[k]);
          if (oi.has_value() == false || oj.has_value() == false ||
              ok.has_value() == false) {
            continue;
          }
          const std::size_t gi = oi.value();
          const std::size_t gj = oj.value();
          const std::size_t gk = ok.value();
          const xt::svector<std::size_t> geom_inds{gi, gj, gk};

          // Get material index
          const auto om = geom_->geom_to_mat_indx(geom_inds);
          if (om.has_value() == false) continue;
          const std::size_t m = om.value();

          flux_out(g, i, j, k) = recon_params(g, m)(x[i], y[j], z[k]);
        }
      }
    }
  }

  return flux_out;
}

xt::xtensor<double, 4> NEMDiffusionDriver::avg_flux() const {
  // If problem isn't solved yet, we error
  if (solved_ == false) {
    auto mssg = "Cannot compute flux. Problem has not been solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  const std::size_t nx = geom_->nx();
  const std::size_t ny = geom_->ny();
  const std::size_t nz = geom_->nz();

  xt::xtensor<double, 4> flux_out;
  flux_out.resize({ngroups(), nx, ny, nz});

  for (std::size_t g = 0; g < ngroups(); g++) {
    for (std::size_t i = 0; i < nx; i++) {
      for (std::size_t j = 0; j < ny; j++) {
        for (std::size_t k = 0; k < nz; k++) {
          const auto om = geom_->geom_to_mat_indx({i, j, k});

          if (om.has_value() == false)
            flux_out(g, i, j, k) = 0.;
          else
            flux_out(g, i, j, k) = flux_(g, om.value(), MomentIndx::AVG);
        }
      }
    }
  }

  return flux_out;
}

double NEMDiffusionDriver::power(double x, double y, double z) const {
  // If problem isn't solved yet, we error
  if (solved_ == false) {
    auto mssg = "Cannot compute power. Problem has not been solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Get geometry index
  const auto oi = geom_->x_to_i(x);
  const auto oj = geom_->y_to_j(y);
  const auto ok = geom_->z_to_k(z);
  if (oi.has_value() == false || oj.has_value() == false ||
      ok.has_value() == false)
    return 0.;
  const std::size_t i = oi.value();
  const std::size_t j = oj.value();
  const std::size_t k = ok.value();
  const xt::svector<std::size_t> geom_inds{i, j, k};

  // Get material index
  const auto om = geom_->geom_to_mat_indx(geom_inds);
  if (om.has_value() == false) return 0.;
  const std::size_t m = om.value();

  const auto& xs = *mats_[m];

  double pwr = 0.;

  for (std::size_t g = 0; g < NG_; g++) {
    pwr += recon_params(g, m)(x, y, z) * xs.Ef(g);
  }

  return pwr;
}

xt::xtensor<double, 3> NEMDiffusionDriver::power(
    const xt::xtensor<double, 1>& x, const xt::xtensor<double, 1>& y,
    const xt::xtensor<double, 1>& z) const {
  // If problem isn't solved yet, we error
  if (solved_ == false) {
    auto mssg = "Cannot compute power. Problem has not been solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Make sure x, y, and z have at least 1 coordinate
  if (x.size() == 0) {
    auto mssg = "Array of x coordinates must have at least one entry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  if (y.size() == 0) {
    auto mssg = "Array of y coordinates must have at least one entry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }
  if (z.size() == 0) {
    auto mssg = "Array of z coordinates must have at least one entry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  xt::xtensor<double, 3> pwr_out;
  pwr_out.resize({x.size(), y.size(), z.size()});
  pwr_out.fill(0.);

#pragma omp parallel for
  for (int ii = 0; ii < static_cast<int>(x.size()); ii++) {
    std::size_t i = static_cast<std::size_t>(ii);
    for (std::size_t j = 0; j < y.size(); j++) {
      for (std::size_t k = 0; k < z.size(); k++) {
        // Get geometry index
        const auto oi = geom_->x_to_i(x[i]);
        const auto oj = geom_->y_to_j(y[j]);
        const auto ok = geom_->z_to_k(z[k]);
        if (oi.has_value() == false || oj.has_value() == false ||
            ok.has_value() == false) {
          continue;
        }
        const std::size_t gi = oi.value();
        const std::size_t gj = oj.value();
        const std::size_t gk = ok.value();
        const xt::svector<std::size_t> geom_inds{gi, gj, gk};

        // Get material index
        const auto om = geom_->geom_to_mat_indx(geom_inds);
        if (om.has_value() == false) {
          continue;
        }
        const std::size_t m = om.value();

        const auto& xs = *mats_[m];

        for (std::size_t g = 0; g < NG_; g++) {
          pwr_out(i, j, k) += recon_params(g, m)(x[i], y[j], z[k]) * xs.Ef(g);
        }
      }
    }
  }

  return pwr_out;
}

xt::xtensor<double, 3> NEMDiffusionDriver::avg_power() const {
  // If problem isn't solved yet, we error
  if (solved_ == false) {
    auto mssg = "Cannot compute power. Problem has not been solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  const std::size_t nx = geom_->nx();
  const std::size_t ny = geom_->ny();
  const std::size_t nz = geom_->nz();

  xt::xtensor<double, 3> pwr_out;
  pwr_out.resize({nx, ny, nz});
  pwr_out.fill(0.);

  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        const auto om = geom_->geom_to_mat_indx({i, j, k});

        if (om.has_value() == false) {
          continue;
        }
        const std::size_t m = om.value();
        const auto& xs = *mats_[m];

        for (std::size_t g = 0; g < NG_; g++) {
          pwr_out(i, j, k) += flux_(g, m, MomentIndx::AVG) * xs.Ef(g);
        }
      }
    }
  }

  return pwr_out;
}

NEMDiffusionDriver::NodeFlux NEMDiffusionDriver::fit_node_recon_params(
    std::size_t g, std::size_t m) const {
  // Get node parameters
  const auto geom_indx = geom_->geom_indx(m);
  const double dx = geom_->dx(geom_indx[0]);
  const double dy = geom_->dy(geom_indx[1]);
  const double dz = geom_->dz(geom_indx[2]);
  const auto& xs = *mats_[m];
  const double D = xs.D(g);
  const double Er = xs.Er(g);
  const double eps = std::sqrt(Er / D);

  const double x_low = geom_->x_bounds()[geom_indx[0]];
  const double x_hi = geom_->x_bounds()[geom_indx[0] + 1];
  const double y_low = geom_->y_bounds()[geom_indx[1]];
  const double y_hi = geom_->y_bounds()[geom_indx[1] + 1];
  const double z_low = geom_->z_bounds()[geom_indx[2]];
  const double z_hi = geom_->z_bounds()[geom_indx[2] + 1];

  const auto& Jin = j_in_out_(g, m, 0);
  const auto& Jout = j_in_out_(g, m, 1);

  const double flx_xp = 2. * (Jout(CurrentIndx::XP) + Jin(CurrentIndx::XP));
  const double flx_xm = 2. * (Jout(CurrentIndx::XM) + Jin(CurrentIndx::XM));
  const double flx_yp = 2. * (Jout(CurrentIndx::YP) + Jin(CurrentIndx::YP));
  const double flx_ym = 2. * (Jout(CurrentIndx::YM) + Jin(CurrentIndx::YM));
  const double flx_zp = 2. * (Jout(CurrentIndx::ZP) + Jin(CurrentIndx::ZP));
  const double flx_zm = 2. * (Jout(CurrentIndx::ZM) + Jin(CurrentIndx::ZM));

  const double flx_avg = flux_(g, m, MomentIndx::AVG);
  const double Jxp = calc_net_current(Jin, Jout, CurrentIndx::XP);
  const double Jxm = calc_net_current(Jin, Jout, CurrentIndx::XM);
  const double Jyp = calc_net_current(Jin, Jout, CurrentIndx::YP);
  const double Jym = calc_net_current(Jin, Jout, CurrentIndx::YM);
  const double Jzp = calc_net_current(Jin, Jout, CurrentIndx::ZP);
  const double Jzm = calc_net_current(Jin, Jout, CurrentIndx::ZM);

  auto sinhc = [](double x) { return std::sinh(x) / x; };

  NodeFlux nf = recon_params(g, m);
  nf.phi_0 = flx_avg;
  nf.eps = eps;
  nf.xm = 0.5 * (x_low + x_hi);
  nf.ym = 0.5 * (y_low + y_hi);
  nf.zm = 0.5 * (z_low + z_hi);
  nf.invs_dx = 1. / dx;
  nf.invs_dy = 1. / dy;
  nf.invs_dz = 1. / dz;

  // Initial base matrix for finding fx, fy, and fz coefficients
  Eigen::Matrix<double, 4, 4> M{
      {0., 0., 1., 1.}, {0., 0., -1., 1.}, {0., 0., 1., 3.}, {0., 0., 1., -3.}};
  Eigen::Matrix<double, 4, 1> b;
  Eigen::Matrix<double, 4, 1> fu_coeffs;

  // Determine fx coefficients
  const double zeta_x = 0.5 * eps * dx;
  M(0, 0) = std::cosh(zeta_x) - sinhc(zeta_x);
  M(0, 1) = std::sinh(zeta_x);
  M(1, 0) = M(0, 0);
  M(1, 1) = -M(0, 1);
  M(2, 0) = zeta_x * std::sinh(zeta_x);
  M(2, 1) = zeta_x * std::cosh(zeta_x);
  M(3, 0) = -M(2, 0);
  M(3, 1) = M(2, 1);
  b(0) = flx_xp - flx_avg;
  b(1) = flx_xm - flx_avg;
  b(2) = -0.5 * Jxp * dx / D;
  b(3) = -0.5 * Jxm * dx / D;
  fu_coeffs = M.inverse() * b;
  nf.ax1 = fu_coeffs(0);
  nf.ax2 = fu_coeffs(1);
  nf.bx1 = fu_coeffs(2);
  nf.bx2 = fu_coeffs(3);
  nf.ax0 = -nf.ax1 * sinhc(zeta_x);
  nf.zeta_x = zeta_x;

  // Determine fy coefficients
  const double zeta_y = 0.5 * eps * dy;
  M(0, 0) = std::cosh(zeta_y) - sinhc(zeta_y);
  M(0, 1) = std::sinh(zeta_y);
  M(1, 0) = M(0, 0);
  M(1, 1) = -M(0, 1);
  M(2, 0) = zeta_y * std::sinh(zeta_y);
  M(2, 1) = zeta_y * std::cosh(zeta_y);
  M(3, 0) = -M(2, 0);
  M(3, 1) = M(2, 1);
  b(0) = flx_yp - flx_avg;
  b(1) = flx_ym - flx_avg;
  b(2) = -0.5 * Jyp * dy / D;
  b(3) = -0.5 * Jym * dy / D;
  fu_coeffs = M.inverse() * b;
  nf.ay1 = fu_coeffs(0);
  nf.ay2 = fu_coeffs(1);
  nf.by1 = fu_coeffs(2);
  nf.by2 = fu_coeffs(3);
  nf.ay0 = -nf.ay1 * sinhc(zeta_y);
  nf.zeta_y = zeta_y;

  // Determine fz coefficients
  const double zeta_z = 0.5 * eps * dz;
  M(0, 0) = std::cosh(zeta_z) - sinhc(zeta_z);
  M(0, 1) = std::sinh(zeta_z);
  M(1, 0) = M(0, 0);
  M(1, 1) = -M(0, 1);
  M(2, 0) = zeta_z * std::sinh(zeta_z);
  M(2, 1) = zeta_z * std::cosh(zeta_z);
  M(3, 0) = -M(2, 0);
  M(3, 1) = M(2, 1);
  b(0) = flx_zp - flx_avg;
  b(1) = flx_zm - flx_avg;
  b(2) = -0.5 * Jzp * dz / D;
  b(3) = -0.5 * Jzm * dz / D;
  fu_coeffs = M.inverse() * b;
  nf.az1 = fu_coeffs(0);
  nf.az2 = fu_coeffs(1);
  nf.bz1 = fu_coeffs(2);
  nf.bz2 = fu_coeffs(3);
  nf.az0 = -nf.az1 * sinhc(zeta_z);

  return nf;
}

void NEMDiffusionDriver::fit_node_recon_params_corners(std::size_t g,
                                                       std::size_t m) {
  const auto geom_indx = geom_->geom_indx(m);

  const double x_low = geom_->x_bounds()[geom_indx[0]];
  const double x_hi = geom_->x_bounds()[geom_indx[0] + 1];
  const double y_low = geom_->y_bounds()[geom_indx[1]];
  const double y_hi = geom_->y_bounds()[geom_indx[1] + 1];

  NodeFlux& nf = recon_params(g, m);

  // If the corner point we are looking at is along an outer boundary,
  // we do not compute the average value of the flux, but instead use
  // the value estimate by the previous node reconstruction, without
  // any cross terms. This allows the use of and f(x,y) term in the flux
  // reconstruction on boundary nodes, without leading to the cusps that
  // would occur when trying to take the average.

  // Determine fxy coefficients
  double flx_pp, flx_pm, flx_mp, flx_mm;
  if (geom_indx[0] != geom_->nx() - 1 && geom_indx[1] != geom_->ny() - 1) {
    flx_pp = avg_xy_corner_flux(g, m, Corner::PP);
  } else {
    flx_pp = nf.flux_xy_no_cross(x_hi, y_hi);
  }

  if (geom_indx[0] != geom_->nx() - 1 && geom_indx[1] != 0) {
    flx_pm = avg_xy_corner_flux(g, m, Corner::PM);
  } else {
    flx_pm = nf.flux_xy_no_cross(x_hi, y_low);
  }

  if (geom_indx[0] != 0 && geom_indx[1] != geom_->ny() - 1) {
    flx_mp = avg_xy_corner_flux(g, m, Corner::MP);
  } else {
    flx_mp = nf.flux_xy_no_cross(x_low, y_hi);
  }

  if (geom_indx[0] != 0 && geom_indx[1] != 0) {
    flx_mm = avg_xy_corner_flux(g, m, Corner::MM);
  } else {
    flx_mm = nf.flux_xy_no_cross(x_low, y_low);
  }

  double pp = flx_pp - nf.flux_xy_no_cross(x_hi, y_hi);
  double pm = flx_pm - nf.flux_xy_no_cross(x_hi, y_low);
  double mp = flx_mp - nf.flux_xy_no_cross(x_low, y_hi);
  double mm = flx_mm - nf.flux_xy_no_cross(x_low, y_low);

  nf.cxy11 = 0.25 * (pp - pm + mm - mp);
  nf.cxy12 = 0.25 * (pp + pm - mm - mp);
  nf.cxy21 = 0.25 * (pp - pm - mm + mp);
  nf.cxy22 = 0.25 * (pp + pm + mm + mp);
}

double NEMDiffusionDriver::eval_heter_xy_corner_flux(std::size_t g,
                                                     std::size_t m,
                                                     Corner c) const {
  const NodeFlux& nf = recon_params(g, m);
  const double dx = 1. / nf.invs_dx;
  const double x_hi = nf.xm + 0.5 * dx;
  const double x_low = nf.xm - 0.5 * dx;
  const double dy = 1. / nf.invs_dy;
  const double y_hi = nf.ym + 0.5 * dy;
  const double y_low = nf.ym - 0.5 * dy;

  // Since the definition of the CDF is
  // CDF = flux_het / flux_hom
  // we compute the heterogeneous flux as flux_het = flux_hom * CDF

  switch (c) {
    case Corner::PP:
      return nf.flux_xy_no_cross(x_hi, y_hi) * geom_->cdf_I(m, g);
      break;

    case Corner::PM:
      return nf.flux_xy_no_cross(x_hi, y_low) * geom_->cdf_IV(m, g);
      break;

    case Corner::MP:
      return nf.flux_xy_no_cross(x_low, y_hi) * geom_->cdf_II(m, g);
      break;

    case Corner::MM:
      return nf.flux_xy_no_cross(x_low, y_low) * geom_->cdf_III(m, g);
      break;
  }

  // NEVER GETS HERE
  return 0.;
}

double NEMDiffusionDriver::avg_xy_corner_flux(std::size_t g, std::size_t m,
                                              Corner c) const {
  const auto geom_inds = geom_inds_(m);

  double num = 0.;
  double denom = 0.;

  // First, we add our contribution to the corner flux estimation
  num += eval_heter_xy_corner_flux(g, m, c);
  denom += 1.;

  if (c == Corner::PP) {
    const auto& n_xp = neighbors_(m, Neighbor::XP);
    const auto& n_yp = neighbors_(m, Neighbor::YP);

    if (n_xp.second) {
      num += eval_heter_xy_corner_flux(g, n_xp.second.value(), Corner::MP);
      denom += 1.;
    }
    if (n_yp.second) {
      num += eval_heter_xy_corner_flux(g, n_yp.second.value(), Corner::PM);
      denom += 1.;
    }

    const auto om = geom_->geom_to_mat_indx(
        {geom_inds[0] + 1, geom_inds[1] + 1, geom_inds[2]});
    if (om) {
      num += eval_heter_xy_corner_flux(g, om.value(), Corner::MM);
      denom += 1.;
    }
  } else if (c == Corner::PM) {
    const auto& n_xp = neighbors_(m, Neighbor::XP);
    const auto& n_ym = neighbors_(m, Neighbor::YN);

    if (n_xp.second) {
      num += eval_heter_xy_corner_flux(g, n_xp.second.value(), Corner::MM);
      denom += 1.;
    }
    if (n_ym.second) {
      num += eval_heter_xy_corner_flux(g, n_ym.second.value(), Corner::PP);
      denom += 1.;
    }

    const auto om = geom_->geom_to_mat_indx(
        {geom_inds[0] + 1, geom_inds[1] - 1, geom_inds[2]});
    if (om) {
      num += eval_heter_xy_corner_flux(g, om.value(), Corner::MP);
      denom += 1.;
    }
  } else if (c == Corner::MM) {
    const auto& n_xm = neighbors_(m, Neighbor::XN);
    const auto& n_ym = neighbors_(m, Neighbor::YN);

    if (n_xm.second) {
      num += eval_heter_xy_corner_flux(g, n_xm.second.value(), Corner::PM);
      denom += 1.;
    }
    if (n_ym.second) {
      num += eval_heter_xy_corner_flux(g, n_ym.second.value(), Corner::MP);
      denom += 1.;
    }

    const auto om = geom_->geom_to_mat_indx(
        {geom_inds[0] - 1, geom_inds[1] - 1, geom_inds[2]});
    if (om) {
      num += eval_heter_xy_corner_flux(g, om.value(), Corner::PP);
      denom += 1.;
    }
  } else {  // c = Corner::MP
    const auto& n_xm = neighbors_(m, Neighbor::XN);
    const auto& n_yp = neighbors_(m, Neighbor::YP);

    if (n_xm.second) {
      num += eval_heter_xy_corner_flux(g, n_xm.second.value(), Corner::PP);
      denom += 1.;
    }
    if (n_yp.second) {
      num += eval_heter_xy_corner_flux(g, n_yp.second.value(), Corner::MM);
      denom += 1.;
    }

    const auto om = geom_->geom_to_mat_indx(
        {geom_inds[0] - 1, geom_inds[1] + 1, geom_inds[2]});
    if (om) {
      num += eval_heter_xy_corner_flux(g, om.value(), Corner::PM);
      denom += 1.;
    }
  }

  const double avg_het_flx = num / denom;

  // The homogeneous flux is then flux_hom = flux_het / CDF
  double CDF = 1.;
  switch (c) {
    case Corner::PP:
      CDF = geom_->cdf_I(m, g);
      break;

    case Corner::PM:
      CDF = geom_->cdf_IV(m, g);
      break;

    case Corner::MP:
      CDF = geom_->cdf_II(m, g);
      break;

    case Corner::MM:
      CDF = geom_->cdf_III(m, g);
      break;
  }

  return avg_het_flx / CDF;
}

void NEMDiffusionDriver::save(const std::string& fname) {
  if (std::filesystem::exists(fname)) {
    std::filesystem::remove(fname);
  }

  std::ofstream file(fname, std::ios_base::binary);

  cereal::PortableBinaryOutputArchive arc(file);

  arc(*this);
}

std::unique_ptr<NEMDiffusionDriver> NEMDiffusionDriver::load(
    const std::string& fname) {
  if (std::filesystem::exists(fname) == false) {
    std::stringstream mssg;
    mssg << "The file \"" << fname << "\" does not exist.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  std::unique_ptr<NEMDiffusionDriver> out(new NEMDiffusionDriver());

  std::ifstream file(fname, std::ios_base::binary);

  cereal::PortableBinaryInputArchive arc(file);

  arc(*out);

  return out;
}

}  // namespace scarabee
