#include <diffusion/fd_diffusion_driver.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>
#include <utils/timer.hpp>

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <cereal/archives/portable_binary.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "utils/simulation_mode.hpp"

namespace scarabee {

void set_x_current_diff(DiffusionGeometry& geom,
                        Eigen::SparseMatrix<double, Eigen::RowMajor>& M,
                        const std::size_t g, const std::size_t m) {
  // Get material index
  const auto indxs = geom.geom_indx(m);

  // Get our left and right neighbors info
  DiffusionGeometry::Tile tile_mm1, tile_mp1;
  std::optional<std::size_t> op_mm1, op_mp1;

  std::tie(tile_mm1, op_mm1) =
      geom.neighbor(m, DiffusionGeometry::Neighbor::XN);
  std::tie(tile_mp1, op_mp1) =
      geom.neighbor(m, DiffusionGeometry::Neighbor::XP);

  // Get parameters for the tile in question
  const double D_m = geom.mat(m)->D(g);
  const double dx_m = geom.dx(indxs[0]);
  const double d_m = D_m / dx_m;

  // Get discontinuity factors for our node
  const double f_p = geom.adf_xp(m, g);
  const double f_m = geom.adf_xn(m, g);

  // Our fill method depends on if we have boundaries or not.
  if (op_mm1 && op_mp1) {
    // Both the left and right are materials. We can fill it like a normal tile
    const std::size_t mm1 = op_mm1.value();
    const std::size_t mp1 = op_mp1.value();

    // Get diffusion coefficients
    const double D_mm1 = geom.mat(mm1)->D(g);
    const double D_mp1 = geom.mat(mp1)->D(g);

    // Get widths
    const double dx_mm1 = geom.dx(indxs[0] - 1);
    const double dx_mp1 = geom.dx(indxs[0] + 1);

    // Get discontinuity factors
    const double f_mp = geom.adf_xp(mm1, g);
    const double f_pm = geom.adf_xn(mp1, g);
    const double r_p = f_p / f_pm;
    const double r_m = f_m / f_mp;

    // Compute reduced diffusion coefficients
    const double d_mm1 = D_mm1 / dx_mm1;
    const double d_mp1 = D_mp1 / dx_mp1;

    // Now we compute the coefficient for each flux term
    const double a =
        -(2. / dx_m) * (d_m * d_mp1 / (d_m + r_p * d_mp1));  // for flux m+1
    const double c =
        -(2. / dx_m) * (d_m * d_mm1 / (d_m + r_m * d_mm1));  // for flux m-1
    const double b = -(r_p * a + r_m * c);                   // for flux m

    // Set matrix components
    M.coeffRef(m + g * geom.nmats(), mp1 + g * geom.nmats()) += a;
    M.coeffRef(m + g * geom.nmats(), m + g * geom.nmats()) += b;
    M.coeffRef(m + g * geom.nmats(), mm1 + g * geom.nmats()) += c;
  } else if (op_mm1) {
    // Boundary condition on the right
    const std::size_t mm1 = op_mm1.value();
    const double D_mm1 = geom.mat(mm1)->D(g);
    const double dx_mm1 = geom.dx(indxs[0] - 1);
    const double d_mm1 = D_mm1 / dx_mm1;

    // Get discontinuity factors
    const double f_mp = geom.adf_xp(mm1, g);
    const double r_m = f_m / f_mp;

    const double c =
        -(2. / dx_m) * (d_m * d_mm1 / (d_m + r_m * d_mm1));  // for flux m-1
    const double alb = tile_mp1.albedo.value();
    const double R = (1. - alb) / (1. + alb);
    const double b =
        -r_m * c + (2. * d_m * R / (dx_m * (4. * d_m + R)));  // for flux m
    // const double b = -c + (2.*d_m/dx_m); // for zero flux at boundary

    M.coeffRef(m + g * geom.nmats(), m + g * geom.nmats()) += b;
    M.coeffRef(m + g * geom.nmats(), mm1 + g * geom.nmats()) += c;
  } else if (op_mp1) {
    // Boundary condition on the left
    const std::size_t mp1 = op_mp1.value();
    const double D_mp1 = geom.mat(mp1)->D(g);
    const double dx_mp1 = geom.dx(indxs[0] + 1);
    const double d_mp1 = D_mp1 / dx_mp1;

    // Get discontinuity factors
    const double f_pm = geom.adf_xn(mp1, g);
    const double r_p = f_p / f_pm;

    const double a =
        -(2. / dx_m) * (d_m * d_mp1 / (d_m + r_p * d_mp1));  // for flux m+1
    const double alb = tile_mm1.albedo.value();
    const double R = (1. - alb) / (1. + alb);
    const double b =
        -r_p * a + (2. * d_m * R / (dx_m * (4. * d_m + R)));  // for flux m
    // const double b = -a + (2.*d_m/dx_m); // for zero flux at boundary

    M.coeffRef(m + g * geom.nmats(), mp1 + g * geom.nmats()) += a;
    M.coeffRef(m + g * geom.nmats(), m + g * geom.nmats()) += b;
  } else {
    // Both sides are boundary conditions. Nothing we can do here.
    std::stringstream mssg;
    mssg << "Boundary condition on left and right of tile in x-direciton.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }
}

void set_y_current_diff(DiffusionGeometry& geom,
                        Eigen::SparseMatrix<double, Eigen::RowMajor>& M,
                        const std::size_t g, const std::size_t m) {
  // Get material index
  const auto indxs = geom.geom_indx(m);

  // Get our left and right neighbors info
  DiffusionGeometry::Tile tile_mm1, tile_mp1;
  std::optional<std::size_t> op_mm1, op_mp1;

  std::tie(tile_mm1, op_mm1) =
      geom.neighbor(m, DiffusionGeometry::Neighbor::YN);
  std::tie(tile_mp1, op_mp1) =
      geom.neighbor(m, DiffusionGeometry::Neighbor::YP);

  // Get parameters for the tile in question
  const double D_m = geom.mat(m)->D(g);
  const double dy_m = geom.dy(indxs[1]);
  const double d_m = D_m / dy_m;

  // Get discontinuity factors for our node
  const double f_p = geom.adf_yp(m, g);
  const double f_m = geom.adf_yn(m, g);

  // Our fill method depends on if we have boundaries or not.
  if (op_mm1 && op_mp1) {
    // Both the left and right are materials. We can fill it like a normal tile
    const std::size_t mm1 = op_mm1.value();
    const std::size_t mp1 = op_mp1.value();

    // Get diffusion coefficients
    const double D_mm1 = geom.mat(mm1)->D(g);
    const double D_mp1 = geom.mat(mp1)->D(g);

    // Get widths
    const double dy_mm1 = geom.dy(indxs[1] - 1);
    const double dy_mp1 = geom.dy(indxs[1] + 1);

    // Get discontinuity factors
    const double f_mp = geom.adf_yp(mm1, g);
    const double f_pm = geom.adf_yn(mp1, g);
    const double r_p = f_p / f_pm;
    const double r_m = f_m / f_mp;

    // Compute reduced diffusion coefficients
    const double d_mm1 = D_mm1 / dy_mm1;
    const double d_mp1 = D_mp1 / dy_mp1;

    // Now we compute the coefficient for each flux term
    const double a =
        -(2. / dy_m) * (d_m * d_mp1 / (d_m + r_p * d_mp1));  // for flux m+1
    const double c =
        -(2. / dy_m) * (d_m * d_mm1 / (d_m + r_m * d_mm1));  // for flux m-1
    const double b = -(r_p * a + r_m * c);                   // for flux m

    // Set matrix components
    M.coeffRef(m + g * geom.nmats(), mp1 + g * geom.nmats()) += a;
    M.coeffRef(m + g * geom.nmats(), m + g * geom.nmats()) += b;
    M.coeffRef(m + g * geom.nmats(), mm1 + g * geom.nmats()) += c;
  } else if (op_mm1) {
    // Boundary condition on the right
    const std::size_t mm1 = op_mm1.value();
    const double D_mm1 = geom.mat(mm1)->D(g);
    const double dy_mm1 = geom.dy(indxs[1] - 1);
    const double d_mm1 = D_mm1 / dy_mm1;

    // Get discontinuity factors
    const double f_mp = geom.adf_yp(mm1, g);
    const double r_m = f_m / f_mp;

    const double c =
        -(2. / dy_m) * (d_m * d_mm1 / (d_m + r_m * d_mm1));  // for flux m-1
    const double alb = tile_mp1.albedo.value();
    const double R = (1. - alb) / (1. + alb);
    const double b =
        -r_m * c + (2. * d_m * R / (dy_m * (4. * d_m + R)));  // for flux m
    // const double b = -c + (2.*d_m/dy_m); // for zero flux at boundary

    M.coeffRef(m + g * geom.nmats(), m + g * geom.nmats()) += b;
    M.coeffRef(m + g * geom.nmats(), mm1 + g * geom.nmats()) += c;
  } else if (op_mp1) {
    // Boundary condition on the left
    const std::size_t mp1 = op_mp1.value();
    const double D_mp1 = geom.mat(mp1)->D(g);
    const double dy_mp1 = geom.dy(indxs[1] + 1);
    const double d_mp1 = D_mp1 / dy_mp1;

    // Get discontinuity factors
    const double f_pm = geom.adf_yn(mp1, g);
    const double r_p = f_p / f_pm;

    const double a =
        -(2. / dy_m) * (d_m * d_mp1 / (d_m + r_p * d_mp1));  // for flux m+1
    const double alb = tile_mm1.albedo.value();
    const double R = (1. - alb) / (1. + alb);
    const double b =
        -r_p * a + (2. * d_m * R / (dy_m * (4. * d_m + R)));  // for flux m
    // const double b = -a + (2.*d_m/dy_m); // for zero flux at boundary

    M.coeffRef(m + g * geom.nmats(), mp1 + g * geom.nmats()) += a;
    M.coeffRef(m + g * geom.nmats(), m + g * geom.nmats()) += b;
  } else {
    // Both sides are boundary conditions. Nothing we can do here.
    std::stringstream mssg;
    mssg << "Boundary condition on left and right of tile in y-direciton.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }
}

void set_z_current_diff(DiffusionGeometry& geom,
                        Eigen::SparseMatrix<double, Eigen::RowMajor>& M,
                        const std::size_t g, const std::size_t m) {
  // Get material index
  const auto indxs = geom.geom_indx(m);

  // Get our left and right neighbors info
  DiffusionGeometry::Tile tile_mm1, tile_mp1;
  std::optional<std::size_t> op_mm1, op_mp1;

  std::tie(tile_mm1, op_mm1) =
      geom.neighbor(m, DiffusionGeometry::Neighbor::ZN);
  std::tie(tile_mp1, op_mp1) =
      geom.neighbor(m, DiffusionGeometry::Neighbor::ZP);

  // Get parameters for the tile in question
  const double D_m = geom.mat(m)->D(g);
  const double dz_m = geom.dz(indxs[2]);
  const double d_m = D_m / dz_m;

  // Get discontinuity factors for our node
  const double f_p = geom.adf_zp(m, g);
  const double f_m = geom.adf_zn(m, g);

  // Our fill method depends on if we have boundaries or not.
  if (op_mm1 && op_mp1) {
    // Both the left and right are materials. We can fill it like a normal tile
    const std::size_t mm1 = op_mm1.value();
    const std::size_t mp1 = op_mp1.value();

    // Get diffusion coefficients
    const double D_mm1 = geom.mat(mm1)->D(g);
    const double D_mp1 = geom.mat(mp1)->D(g);

    // Get widths
    const double dz_mm1 = geom.dz(indxs[2] - 1);
    const double dz_mp1 = geom.dz(indxs[2] + 1);

    // Get discontinuity factors
    const double f_mp = geom.adf_zp(mm1, g);
    const double f_pm = geom.adf_zn(mp1, g);
    const double r_p = f_p / f_pm;
    const double r_m = f_m / f_mp;

    // Compute reduced diffusion coefficients
    const double d_mm1 = D_mm1 / dz_mm1;
    const double d_mp1 = D_mp1 / dz_mp1;

    // Now we compute the coefficient for each flux term
    const double a =
        -(2. / dz_m) * (d_m * d_mp1 / (d_m + r_p * d_mp1));  // for flux m+1
    const double c =
        -(2. / dz_m) * (d_m * d_mm1 / (d_m + r_m * d_mm1));  // for flux m-1
    const double b = -(r_p * a + r_m * c);                   // for flux m

    // Set matrix components
    M.coeffRef(m + g * geom.nmats(), mp1 + g * geom.nmats()) += a;
    M.coeffRef(m + g * geom.nmats(), m + g * geom.nmats()) += b;
    M.coeffRef(m + g * geom.nmats(), mm1 + g * geom.nmats()) += c;
  } else if (op_mm1) {
    // Boundary condition on the right
    const std::size_t mm1 = op_mm1.value();
    const double D_mm1 = geom.mat(mm1)->D(g);
    const double dz_mm1 = geom.dz(indxs[2] - 1);
    const double d_mm1 = D_mm1 / dz_mm1;

    // Get discontinuity factors
    const double f_mp = geom.adf_zp(mm1, g);
    const double r_m = f_m / f_mp;

    const double c =
        -(2. / dz_m) * (d_m * d_mm1 / (d_m + r_m * d_mm1));  // for flux m-1
    const double alb = tile_mp1.albedo.value();
    const double R = (1. - alb) / (1. + alb);
    const double b =
        -r_m * c + (2. * d_m * R / (dz_m * (4. * d_m + R)));  // for flux m
    // const double b = -c + (2.*d_m/dy_m); // for zero flux at boundary

    M.coeffRef(m + g * geom.nmats(), m + g * geom.nmats()) += b;
    M.coeffRef(m + g * geom.nmats(), mm1 + g * geom.nmats()) += c;
  } else if (op_mp1) {
    // Boundary condition on the left
    const std::size_t mp1 = op_mp1.value();
    const double D_mp1 = geom.mat(mp1)->D(g);
    const double dz_mp1 = geom.dz(indxs[2] + 1);
    const double d_mp1 = D_mp1 / dz_mp1;

    // Get discontinuity factors
    const double f_pm = geom.adf_zn(mp1, g);
    const double r_p = f_p / f_pm;

    const double a =
        -(2. / dz_m) * (d_m * d_mp1 / (d_m + r_p * d_mp1));  // for flux m+1
    const double alb = tile_mm1.albedo.value();
    const double R = (1. - alb) / (1. + alb);
    const double b =
        -r_p * a + (2. * d_m * R / (dz_m * (4. * d_m + R)));  // for flux m
    // const double b = -a + (2.*d_m/dz_m); // for zero flux at boundary

    M.coeffRef(m + g * geom.nmats(), mp1 + g * geom.nmats()) += a;
    M.coeffRef(m + g * geom.nmats(), m + g * geom.nmats()) += b;
  } else {
    // Both sides are boundary conditions. Nothing we can do here.
    std::stringstream mssg;
    mssg << "Boundary condition on left and right of tile in z-direciton.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }
}

void load_loss_matrix(DiffusionGeometry& geom,
                      Eigen::SparseMatrix<double, Eigen::RowMajor>& M) {
  const std::size_t NMATS = geom.nmats();
  const std::size_t NGRPS = geom.ngroups();

  // Allocate M
  M.resize(NGRPS * NMATS, NGRPS * NMATS);

  if (geom.ndims() == 1) {
    M.reserve(Eigen::VectorX<std::size_t>::Constant(NGRPS * NMATS, 2 + NGRPS));
  } else if (geom.ndims() == 2) {
    M.reserve(Eigen::VectorX<std::size_t>::Constant(NGRPS * NMATS, 4 + NGRPS));
  } else {
    M.reserve(Eigen::VectorX<std::size_t>::Constant(NGRPS * NMATS, 6 + NGRPS));
  }

  for (std::size_t m = 0; m < geom.nmats(); m++) {
    const auto& mat = geom.mat(m);
    for (std::size_t g = 0; g < geom.ngroups(); g++) {
      set_x_current_diff(geom, M, g, m);

      if (geom.ndims() > 1) set_y_current_diff(geom, M, g, m);

      if (geom.ndims() > 2) set_z_current_diff(geom, M, g, m);

      // Get removal xs for m
      const double Er = mat->Er(g);

      // Add removal xs along the diagonal
      M.coeffRef(m + g * geom.nmats(), m + g * geom.nmats()) += Er;

      // Remove scattering sources
      for (std::size_t gg = 0; gg < geom.ngroups(); gg++) {
        if (gg != g) {
          M.coeffRef(m + g * geom.nmats(), m + gg * geom.nmats()) -=
              mat->Es(gg, g);
        }
      }
    }
  }
  M.makeCompressed();
}

void load_source_matrix(const DiffusionGeometry& geom,
                        Eigen::SparseMatrix<double, Eigen::RowMajor>& QM) {
  const std::size_t NMATS = geom.nmats();
  const std::size_t NGRPS = geom.ngroups();

  QM.resize(NGRPS * NMATS, NGRPS * NMATS);
  QM.reserve(Eigen::VectorX<std::size_t>::Constant(NGRPS * NMATS, NGRPS));

  for (std::size_t m = 0; m < NMATS; m++) {
    const DiffusionData& xs = *geom.mat(m);

    for (std::size_t g = 0; g < NGRPS; g++) {
      const double chi_g = xs.chi(g);

      for (std::size_t gg = 0; gg < NGRPS; gg++) {
        QM.coeffRef(m + g * NMATS, m + gg * NMATS) = chi_g * xs.vEf(gg);
      }
    }
  }

  QM.makeCompressed();
}

FDDiffusionDriver::FDDiffusionDriver(std::shared_ptr<DiffusionGeometry> geom)
    : geom_(geom), flux_(), extern_src_(), mode_(SimulationMode::Keff) {
  if (geom_ == nullptr) {
    auto mssg = "FDDiffusionDriver provided with nullptr geometry.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  flux_.resize(geom_->ngroups() * geom_->nmats());
  flux_.fill(1.);

  extern_src_.resize(geom_->ngroups() * geom_->nmats());
  extern_src_.fill(0.);
}

void FDDiffusionDriver::set_flux_tolerance(double ftol) {
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

void FDDiffusionDriver::set_keff_tolerance(double ktol) {
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

double FDDiffusionDriver::flux(std::size_t i, std::size_t g) const {
  if (geom_->ndims() != 1) {
    std::stringstream mssg;
    mssg << "Provided 1 index to a " << geom_->ndims() << "D problem.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (g >= this->ngroups()) {
    const auto mssg = "Group index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (i >= this->geometry()->nx()) {
    const auto mssg = "x index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  auto mopt = this->geometry()->geom_to_mat_indx({i});

  if (mopt.has_value() == false) {
    return 0.;
  }

  return flux_(mopt.value() + g * geom_->nmats());
}

double FDDiffusionDriver::flux(std::size_t i, std::size_t j,
                               std::size_t g) const {
  if (geom_->ndims() != 2) {
    std::stringstream mssg;
    mssg << "Provided 2 indices to a " << geom_->ndims() << "D problem.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (g >= this->ngroups()) {
    const auto mssg = "Group index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (i >= this->geometry()->nx()) {
    const auto mssg = "x index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (j >= this->geometry()->ny()) {
    const auto mssg = "y index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  auto mopt = this->geometry()->geom_to_mat_indx({i, j});

  if (mopt.has_value() == false) {
    return 0.;
  }

  return flux_(mopt.value() + g * geom_->nmats());
}

double FDDiffusionDriver::flux(std::size_t i, std::size_t j, std::size_t k,
                               std::size_t g) const {
  if (geom_->ndims() != 3) {
    std::stringstream mssg;
    mssg << "Provided 3 indices to a " << geom_->ndims() << "D problem.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (g >= this->ngroups()) {
    const auto mssg = "Group index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (i >= this->geometry()->nx()) {
    const auto mssg = "x index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (j >= this->geometry()->ny()) {
    const auto mssg = "y index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (k >= this->geometry()->nz()) {
    const auto mssg = "z index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  auto mopt = this->geometry()->geom_to_mat_indx({i, j, k});

  if (mopt.has_value() == false) {
    return 0.;
  }

  return flux_(mopt.value() + g * geom_->nmats());
}

double FDDiffusionDriver::extern_src(std::size_t i, std::size_t g) const {
  if (geom_->ndims() != 1) {
    std::stringstream mssg;
    mssg << "Provided 1 index to a " << geom_->ndims() << "D problem.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (g >= this->ngroups()) {
    const auto mssg = "Group index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (i >= this->geometry()->nx()) {
    const auto mssg = "x index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  auto mopt = this->geometry()->geom_to_mat_indx({i});

  if (mopt.has_value() == false) {
    return 0.;
  }

  return extern_src_(mopt.value() + g * geom_->nmats());
}

double FDDiffusionDriver::extern_src(std::size_t i, std::size_t j,
                                     std::size_t g) const {
  if (geom_->ndims() != 2) {
    std::stringstream mssg;
    mssg << "Provided 2 indices to a " << geom_->ndims() << "D problem.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (g >= this->ngroups()) {
    const auto mssg = "Group index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (i >= this->geometry()->nx()) {
    const auto mssg = "x index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (j >= this->geometry()->ny()) {
    const auto mssg = "y index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  auto mopt = this->geometry()->geom_to_mat_indx({i, j});

  if (mopt.has_value() == false) {
    return 0.;
  }

  return extern_src_(mopt.value() + g * geom_->nmats());
}

double FDDiffusionDriver::extern_src(std::size_t i, std::size_t j,
                                     std::size_t k, std::size_t g) const {
  if (geom_->ndims() != 3) {
    std::stringstream mssg;
    mssg << "Provided 3 indices to a " << geom_->ndims() << "D problem.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (g >= this->ngroups()) {
    const auto mssg = "Group index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (i >= this->geometry()->nx()) {
    const auto mssg = "x index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (j >= this->geometry()->ny()) {
    const auto mssg = "y index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (k >= this->geometry()->nz()) {
    const auto mssg = "z index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  auto mopt = this->geometry()->geom_to_mat_indx({i, j, k});

  if (mopt.has_value() == false) {
    return 0.;
  }

  return extern_src_(mopt.value() + g * geom_->nmats());
}

void FDDiffusionDriver::set_extern_src(std::size_t i, std::size_t g,
                                       double src) {
  if (geom_->ndims() != 1) {
    std::stringstream mssg;
    mssg << "Provided 1 index to a " << geom_->ndims() << "D problem.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (g >= this->ngroups()) {
    const auto mssg = "Group index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (i >= this->geometry()->nx()) {
    const auto mssg = "x index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  auto mopt = this->geometry()->geom_to_mat_indx({i});

  if (mopt.has_value() == false) {
    std::stringstream mssg;
    mssg << "Tile at index (" << i << ") is outside defined geometry.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  extern_src_(mopt.value() + g * geom_->nmats()) = src;
}

void FDDiffusionDriver::set_extern_src(std::size_t i, std::size_t j,
                                       std::size_t g, double src) {
  if (geom_->ndims() != 2) {
    std::stringstream mssg;
    mssg << "Provided 2 indices to a " << geom_->ndims() << "D problem.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (g >= this->ngroups()) {
    const auto mssg = "Group index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (i >= this->geometry()->nx()) {
    const auto mssg = "x index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (j >= this->geometry()->ny()) {
    const auto mssg = "y index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  auto mopt = this->geometry()->geom_to_mat_indx({i, j});

  if (mopt.has_value() == false) {
    std::stringstream mssg;
    mssg << "Tile at index (" << i << ", " << j
         << ") is outside defined geometry.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  extern_src_(mopt.value() + g * geom_->nmats()) = src;
}

void FDDiffusionDriver::set_extern_src(std::size_t i, std::size_t j,
                                       std::size_t k, std::size_t g,
                                       double src) {
  if (geom_->ndims() != 3) {
    std::stringstream mssg;
    mssg << "Provided 3 indices to a " << geom_->ndims() << "D problem.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  if (g >= this->ngroups()) {
    const auto mssg = "Group index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (i >= this->geometry()->nx()) {
    const auto mssg = "x index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (j >= this->geometry()->ny()) {
    const auto mssg = "y index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  if (k >= this->geometry()->nz()) {
    const auto mssg = "z index out of range.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  auto mopt = this->geometry()->geom_to_mat_indx({i, j, k});

  if (mopt.has_value() == false) {
    std::stringstream mssg;
    mssg << "Tile at index (" << i << ", " << j << ", " << k
         << ") is outside defined geometry.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  extern_src_(mopt.value() + g * geom_->nmats()) = src;
}

void FDDiffusionDriver::power_iteration() {
  Timer sim_timer;
  sim_timer.start();

  spdlog::info("Solving for keff.");
  spdlog::info("keff tolerance: {:.5E}", keff_tol_);
  spdlog::info("Flux tolerance: {:.5E}", flux_tol_);

  // First, we create our loss matrix
  Eigen::SparseMatrix<double, Eigen::RowMajor> M;

  // Load the loss matrix
  load_loss_matrix(*geom_, M);

  // Initialize flux and source vectors
  Eigen::VectorXd new_flux(geom_->ngroups() * geom_->nmats());
  Eigen::VectorXd Q(geom_->ngroups() * geom_->nmats());

  flux_.normalize();

  // Initialize a vector for computing keff faster
  Eigen::VectorXd VvEf(geom_->ngroups() * geom_->nmats());
  for (std::size_t m = 0; m < geom_->nmats(); m++) {
    const double Vm = geom_->volume(m);
    const auto& mat = geom_->mat(m);
    for (std::size_t g = 0; g < geom_->ngroups(); g++) {
      VvEf(m + g * geom_->nmats()) = Vm * mat->vEf(g);
    }
  }

  // Initialize a vector for computing the source vector Q faster
  Eigen::SparseMatrix<double, Eigen::RowMajor> QM;
  load_source_matrix(*geom_, QM);

  // Create a solver for the problem
  spdlog::info("Initializing iterative solver");
  Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver;
  solver.compute(M);
  solver.setTolerance(1.E-8);
  if (solver.info() != Eigen::Success) {
    std::stringstream mssg;
    mssg << "Could not initialize iterative solver";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // Begin power iteration
  double keff_diff = 100.;
  double flux_diff = 100.;
  std::size_t iteration = 0;
  Timer iteration_timer;
  while (keff_diff > keff_tol_ || flux_diff > flux_tol_) {
    iteration_timer.reset();
    iteration_timer.start();
    iteration++;

    // Compute source vector
    Q = (1. / keff_) * QM * flux_;

    // Get new flux
    new_flux = solver.solveWithGuess(Q, flux_);
    // For some reason, this doesn't seem to be working with the new versions
    // of Eigen, despite clearly succeeding. Just commenting it out for now.
    // if (solver.info() != Eigen::Success) {
    //  spdlog::error("Solution impossible.");
    //  throw ScarabeeException("Solution impossible");
    // }

    // Estiamte keff
    double prev_keff = keff_;
    keff_ = prev_keff * VvEf.dot(new_flux) / VvEf.dot(flux_);
    keff_diff = std::abs(keff_ - prev_keff) / keff_;

    // Normalize our new flux
    new_flux *= prev_keff / keff_;

    // Find the max flux error
    flux_diff = 0.;
    for (std::size_t i = 0; i < geom_->ngroups() * geom_->nmats(); i++) {
      double flux_diff_i = std::abs(new_flux(i) - flux_(i)) / new_flux(i);
      if (flux_diff_i > flux_diff) flux_diff = flux_diff_i;
    }
    flux_ = new_flux;

    // Write information
    spdlog::info("-------------------------------------");
    spdlog::info("Iteration {:>4d}          keff: {:.5f}", iteration, keff_);
    spdlog::info("     keff difference:     {:.5E}", keff_diff);
    spdlog::info("     max flux difference: {:.5E}", flux_diff);
    spdlog::info("     iteration time: {:.5E} s",
                 iteration_timer.elapsed_time());
  }

  solved_ = true;

  sim_timer.stop();
  spdlog::info("");
  spdlog::info("Simulation Time: {:.5E} s", sim_timer.elapsed_time());
}

void FDDiffusionDriver::fixed_source() {
  Timer sim_timer;
  sim_timer.start();

  spdlog::info("Solving fixed source problem.");
  spdlog::info("Flux tolerance: {:.5E}", flux_tol_);

  // First, we create our loss matrix
  Eigen::SparseMatrix<double, Eigen::RowMajor> M;

  // Load the loss matrix
  load_loss_matrix(*geom_, M);

  // Initialize a vector for computing keff faster
  Eigen::VectorXd VvEf(geom_->ngroups() * geom_->nmats());
  for (std::size_t m = 0; m < geom_->nmats(); m++) {
    const double Vm = geom_->volume(m);
    const auto& mat = geom_->mat(m);
    for (std::size_t g = 0; g < geom_->ngroups(); g++) {
      VvEf(m + g * geom_->nmats()) = Vm * mat->vEf(g);
    }
  }

  // Initialize a vector for computing the source vector Q faster
  Eigen::SparseMatrix<double, Eigen::RowMajor> QM;
  load_source_matrix(*geom_, QM);

  // Subtract source matrix from loss matrix (only for fixed-source problems)
  M -= QM;

  // Create a solver for the problem
  spdlog::info("Initializing iterative solver");
  Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver;
  solver.compute(M);
  solver.setTolerance(flux_tol_);
  if (solver.info() != Eigen::Success) {
    std::stringstream mssg;
    mssg << "Could not initialize iterative solver";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // Get new flux
  flux_ = solver.solve(extern_src_);
  // For some reason, this doesn't seem to be working with the new versions
  // of Eigen, despite clearly succeeding. Just commenting it out for now.
  // if (solver.info() != Eigen::Success) {
  //  spdlog::error("Solution impossible.");
  //  throw ScarabeeException("Solution impossible");
  // }

  solved_ = true;

  sim_timer.stop();
  spdlog::info("");
  spdlog::info("Simulation Time: {:.5E} s", sim_timer.elapsed_time());
}

void FDDiffusionDriver::solve() {
  if (sim_mode() == SimulationMode::Keff) {
    this->power_iteration();
  } else {
    this->fixed_source();
  }
}

std::tuple<xt::xarray<double>, xt::xarray<double>,
           std::optional<xt::xarray<double>>, std::optional<xt::xarray<double>>>
FDDiffusionDriver::flux() const {
  // If problem isn't solved yet, we error
  if (solved_ == false) {
    auto mssg = "Cannot rasterize flux. Problem has not been solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Initialize empty flux array with zeros
  xt::xarray<double> flux;
  if (geom_->ndims() == 1) {
    flux = xt::zeros<double>({geom_->ngroups(), geom_->nx()});
  } else if (geom_->ndims() == 2) {
    flux = xt::zeros<double>({geom_->ngroups(), geom_->nx(), geom_->ny()});
  } else {
    flux = xt::zeros<double>(
        {geom_->ngroups(), geom_->nx(), geom_->ny(), geom_->nz()});
  }

  // Fill the flux
  for (std::size_t m = 0; m < geom_->nmats(); m++) {
    auto tmp_inds = geom_->geom_indx(m);
    xt::svector<std::size_t> inds;
    inds.push_back(0);
    for (std::size_t i = 0; i < tmp_inds.size(); i++) {
      inds.push_back(tmp_inds[i]);
    }

    for (std::size_t g = 0; g < geom_->ngroups(); g++) {
      inds[0] = g;
      flux.element(inds.begin(), inds.end()) = flux_(m + g * geom_->nmats());
    }
  }

  // Create the arrays for the x, y, and z bounds
  xt::xarray<double> x_bounds;
  std::optional<xt::xarray<double>> y_bounds = std::nullopt;
  std::optional<xt::xarray<double>> z_bounds = std::nullopt;

  x_bounds = xt::zeros<double>({geom_->nx() + 1});
  for (std::size_t i = 0; i < geom_->nx(); i++) {
    x_bounds(i + 1) = x_bounds(i) + geom_->dx(i);
  }

  if (geom_->ndims() > 1) {
    y_bounds = xt::zeros<double>({geom_->ny() + 1});
    for (std::size_t j = 0; j < geom_->ny(); j++) {
      y_bounds.value()(j + 1) = y_bounds.value()(j) + geom_->dy(j);
    }
  }

  if (geom_->ndims() > 2) {
    z_bounds = xt::zeros<double>({geom_->nz() + 1});
    for (std::size_t k = 0; k < geom_->nz(); k++) {
      z_bounds.value()(k + 1) = z_bounds.value()(k) + geom_->dz(k);
    }
  }

  return {flux, x_bounds, y_bounds, z_bounds};
}

std::tuple<xt::xarray<double>, xt::xarray<double>,
           std::optional<xt::xarray<double>>, std::optional<xt::xarray<double>>>
FDDiffusionDriver::power() const {
  // If problem isn't solved yet, we error
  if (solved_ == false) {
    auto mssg = "Cannot rasterize power. Problem has not been solved.";
    spdlog::error(mssg);
    throw ScarabeeException(mssg);
  }

  // Initialize empty power array with zeros
  xt::xarray<double> power;
  if (geom_->ndims() == 1) {
    power = xt::zeros<double>({geom_->nx()});
  } else if (geom_->ndims() == 2) {
    power = xt::zeros<double>({geom_->nx(), geom_->ny()});
  } else {
    power = xt::zeros<double>({geom_->nx(), geom_->ny(), geom_->nz()});
  }

  // Fill the power
  for (std::size_t m = 0; m < geom_->nmats(); m++) {
    auto inds = geom_->geom_indx(m);
    const auto& xs = geom_->mat(m);

    for (std::size_t g = 0; g < geom_->ngroups(); g++) {
      power.element(inds.begin(), inds.end()) +=
          flux_(m + g * geom_->nmats()) * xs->Ef(g);
    }
  }

  // Create the arrays for the x, y, and z bounds
  xt::xarray<double> x_bounds;
  std::optional<xt::xarray<double>> y_bounds = std::nullopt;
  std::optional<xt::xarray<double>> z_bounds = std::nullopt;

  x_bounds = xt::zeros<double>({geom_->nx() + 1});
  for (std::size_t i = 0; i < geom_->nx(); i++) {
    x_bounds(i + 1) = x_bounds(i) + geom_->dx(i);
  }

  if (geom_->ndims() > 1) {
    y_bounds = xt::zeros<double>({geom_->ny() + 1});
    for (std::size_t j = 0; j < geom_->ny(); j++) {
      y_bounds.value()(j + 1) = y_bounds.value()(j) + geom_->dy(j);
    }
  }

  if (geom_->ndims() > 2) {
    z_bounds = xt::zeros<double>({geom_->nz() + 1});
    for (std::size_t k = 0; k < geom_->nz(); k++) {
      z_bounds.value()(k + 1) = z_bounds.value()(k) + geom_->dz(k);
    }
  }

  return {power, x_bounds, y_bounds, z_bounds};
}

void FDDiffusionDriver::save(const std::string& fname) {
  if (std::filesystem::exists(fname)) {
    std::filesystem::remove(fname);
  }

  std::ofstream file(fname, std::ios_base::binary);

  cereal::PortableBinaryOutputArchive arc(file);

  arc(*this);
}

std::unique_ptr<FDDiffusionDriver> FDDiffusionDriver::load(
    const std::string& fname) {
  if (std::filesystem::exists(fname) == false) {
    std::stringstream mssg;
    mssg << "The file \"" << fname << "\" does not exist.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  std::unique_ptr<FDDiffusionDriver> out(new FDDiffusionDriver());

  std::ifstream file(fname, std::ios_base::binary);

  cereal::PortableBinaryInputArchive arc(file);

  arc(*out);

  return out;
}

}  // namespace scarabee
