#include <utils/criticality_spectrum.hpp>
#include <utils/logging.hpp>
#include <utils/scarabee_exception.hpp>

#include <xtensor/generators/xbuilder.hpp>

#include <cereal/archives/portable_binary.hpp>

#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <sstream>

namespace scarabee {

void fill_A(Eigen::MatrixXd& A, std::shared_ptr<CrossSection> xs,
            const Eigen::MatrixXd& D, const double B2) {
  const std::size_t NG = xs->ngroups();

  A.fill(0.);

  for (std::size_t g = 0; g < NG; g++) {
    for (std::size_t gg = 0; gg < NG; gg++) {
      A(g, gg) = B2 * D(g, gg) - xs->Es(0, gg, g);
    }
    A(g, g) += xs->Et(g);
  }
}

void fill_A_fundamental_mode(Eigen::MatrixXd& A,
                             std::shared_ptr<CrossSection> xs,
                             const xt::xtensor<double, 1>& D, const double B2) {
  const std::size_t NG = xs->ngroups();
  A.fill(0.);

  for (std::size_t g = 0; g < NG; g++) {
    A(g, g) += xs->Etr(g) + B2 * D(g);
    for (std::size_t gg = 0; gg < NG; gg++) {
      A(g, gg) -= xs->Es_tr(gg, g);
    }
  }
}

void fill_Dinvs(Eigen::MatrixXd& Dinvs, std::shared_ptr<CrossSection> xs,
                const xt::xtensor<double, 1>& a) {
  const std::size_t NG = xs->ngroups();

  Dinvs.fill(0.);

  for (std::size_t g = 0; g < NG; g++) {
    for (std::size_t gg = 0; gg < NG; gg++) {
      Dinvs(g, gg) = -xs->Es(1, gg, g);
    }
    Dinvs(g, g) += a(g) * xs->Et(g);
  }

  Dinvs *= 3.;
}

void fill_alphas(xt::xtensor<double, 1>& a,
                 const std::shared_ptr<CrossSection>& xs, const double B2) {
  const std::size_t NG = xs->ngroups();
  for (std::size_t g = 0; g < NG; g++) {
    const double Et2 = xs->Et(g) * xs->Et(g);
    const double x2 = std::abs(B2 / Et2);
    const double x = std::sqrt(x2);

    // Et2 will always be > 0, so we only need to check the sign of B2
    if (x2 < 1.E-6) {
      const double y = B2 / Et2;
      const double x = (1. / 3.) - y * (1. / 5. - y / 7.);
      a(g) = (1. / x) - y;
    } else if (B2 > 0.) {
      const double atx = std::atan(x);
      a(g) = x2 * (atx / (x - atx));
    } else {
      const double ln = std::log((1. + x) / (1. - x));
      a(g) = x2 * (ln / (ln - 2. * x));
    }
  }

  a /= 3.;
}

void compute_flux_current_diff_coeffs(
    const Eigen::VectorXd& flx, Eigen::VectorXd& cur, const Eigen::MatrixXd& D,
    xt::xtensor<double, 1>& flux_, xt::xtensor<double, 1>& current_,
    xt::xtensor<double, 1>& diff_coeff_, double B2_) {
  const std::size_t NG = static_cast<std::size_t>(flx.size());

  const double sqrt_abs_B2 = std::sqrt(std::abs(B2_));
  const double B = B2_ > 0. ? sqrt_abs_B2 : -sqrt_abs_B2;

  // Get the current
  cur = B * D * flx;

  // Vector used for computing the diffusion coefficient
  const auto diff_vec = D * flx;

  // The output info is in the format (2,NG) where the first line has
  // the flux spectrum, and the second has the diffusion coefficients.
  flux_.resize({NG});
  current_.resize({NG});
  diff_coeff_.resize({NG});
  for (std::size_t g = 0; g < NG; g++) {
    flux_(g) = flx(g);
    current_(g) = cur(g);
    diff_coeff_(g) = diff_vec(g) / flx(g);
  }
}

CriticalitySpectrum::CriticalitySpectrum(py::tuple t) {
  xs_ = t[0].cast<std::shared_ptr<CrossSection>>();
  py::bytes bytes = t[1].cast<py::bytes>();
  std::istringstream bits_stream(bytes,
                                 std::ios_base::binary | std::ios_base::in);
  {
    cereal::PortableBinaryInputArchive ar(bits_stream);
    ar(flux_, diff_coeff_, k_inf_, B2_);
  }
}

py::tuple CriticalitySpectrum::to_tuple() const {
  std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
  {
    cereal::PortableBinaryOutputArchive ar(bits_stream);
    ar(flux_, diff_coeff_, k_inf_, B2_);
  }
  py::bytes bytes(bits_stream.str());

  return py::make_tuple(xs_, bytes);
}

std::shared_ptr<DiffusionCrossSection>
CriticalitySpectrum::make_diffusion_cross_section() const {
  const std::size_t NG = xs_->ngroups();
  xt::xtensor<double, 1> Ea = xt::zeros<double>({NG});
  xt::xtensor<double, 1> Ef = xt::zeros<double>({NG});
  xt::xtensor<double, 1> vEf = xt::zeros<double>({NG});
  xt::xtensor<double, 1> chi = xt::zeros<double>({NG});
  xt::xtensor<double, 2> Es = xt::zeros<double>({NG, NG});

  for (std::size_t g = 0; g < NG; g++) {
    Ea(g) = xs_->Ea(g);
    Ef(g) = xs_->Ef(g);
    vEf(g) = xs_->vEf(g);
    chi(g) = xs_->chi(g);

    for (std::size_t gg = 0; gg < NG; gg++) {
      Es(g, gg) = xs_->Es(0, g, gg);
    }
  }

  return std::make_shared<DiffusionCrossSection>(diff_coeff_, Ea, Es, Ef, vEf,
                                                 chi, xs_->name());
}

FundamentalModeCriticalitySpectrum::FundamentalModeCriticalitySpectrum(
    std::shared_ptr<CrossSection> xs) {
  if (xs->fissile() == false) {
    std::stringstream mssg;
    mssg << "Cannot compute fundamental mode spectrum of homogenized material "
            "that is not fissile.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // Save copy of cross section
  xs_ = xs;

  const std::size_t NG = xs_->ngroups();

  // Make the chi and vEf vectors. Also fill diffusion coefficients
  Eigen::VectorXd chi(NG), vEf(NG);
  diff_coeff_ = xt::zeros<double>({NG});
  for (std::size_t g = 0; g < NG; g++) {
    chi(g) = xs_->chi(g);
    vEf(g) = xs_->vEf(g);
    diff_coeff_(g) = 1. / (3. * xs_->Etr(g));
  }

  // Make vector for flux
  Eigen::VectorXd flx(NG);

  // Create the A matrix
  Eigen::MatrixXd A(NG, NG);

  // Get A for k_inf
  B2_ = 0.;
  fill_A_fundamental_mode(A, xs_, diff_coeff_, B2_);
  auto A_solver = A.colPivHouseholderQr();
  flx = A_solver.solve(chi);
  k_inf_ = vEf.dot(flx);

  // Get A for small B2, from Stamm'ler and Abbate
  B2_ = 0.001;
  if (k_inf_ < 1.) B2_ = -B2_;
  fill_A_fundamental_mode(A, xs_, diff_coeff_, B2_);
  A_solver = A.colPivHouseholderQr();
  flx = A_solver.solve(chi);
  const double k_1 = vEf.dot(flx);

  // Calculate the slope constant
  const double k_inf_M2 = B2_ / ((1. / k_1) - (1. / k_inf_));

  double k = k_1;
  while (std::abs(k - 1.) > 1.E-6) {
    B2_ += k_inf_M2 * (1. - (1. / k));
    fill_A_fundamental_mode(A, xs_, diff_coeff_, B2_);
    A_solver = A.colPivHouseholderQr();
    flx = A_solver.solve(chi);
    k = vEf.dot(flx);
  }

  // Copy flux
  flux_ = xt::zeros<double>({NG});
  for (std::size_t g = 0; g < NG; g++) {
    flux_(g) = flx(g);
  }
}

FundamentalModeCriticalitySpectrum::FundamentalModeCriticalitySpectrum(
    std::shared_ptr<CrossSection> xs, double B2) {
  // Save copy of cross section
  xs_ = xs;
  B2_ = B2;

  const std::size_t NG = xs_->ngroups();

  // Make the chi and vEf vectors. Also fill diffusion coefficients
  Eigen::VectorXd chi(NG), vEf(NG);
  diff_coeff_ = xt::zeros<double>({NG});
  for (std::size_t g = 0; g < NG; g++) {
    chi(g) = xs_->chi(g);
    vEf(g) = xs_->vEf(g);
    diff_coeff_(g) = 1. / (3. * xs_->Etr(g));
  }

  // Make vector for flux
  Eigen::VectorXd flx(NG);

  // Create the A matrix
  Eigen::MatrixXd A(NG, NG);

  // Get A for k_inf
  fill_A_fundamental_mode(A, xs_, diff_coeff_, B2_);
  auto A_solver = A.colPivHouseholderQr();
  flx = A_solver.solve(chi);
  k_inf_ = vEf.dot(flx);

  // Copy flux
  flux_ = xt::zeros<double>({NG});
  for (std::size_t g = 0; g < NG; g++) {
    flux_(g) = flx(g);
  }
}

FundamentalModeCriticalitySpectrum::FundamentalModeCriticalitySpectrum(
    py::tuple t)
    : CriticalitySpectrum(t) {}

py::tuple FundamentalModeCriticalitySpectrum::to_tuple() const {
  return CriticalitySpectrum::to_tuple();
}

CriticalitySpectrumWithCurrent::CriticalitySpectrumWithCurrent(py::tuple t)
    : CriticalitySpectrum(t[0].cast<py::tuple>()) {
  std::vector<double> flat_current = t[1].cast<std::vector<double>>();
  current_.resize({flat_current.size()});
  for (std::size_t i = 0; i < current_.size(); i++)
    current_.flat(i) = flat_current[i];
}

py::tuple CriticalitySpectrumWithCurrent::to_tuple() const {
  std::vector<double> flat_current;
  flat_current.resize(current_.size());
  for (std::size_t i = 0; i < current_.size(); i++)
    flat_current[i] = current_.flat(i);

  return py::make_tuple(CriticalitySpectrum::to_tuple(), flat_current);
}

void CriticalitySpectrumWithCurrent::P1_B1_spectrum_search(bool B1) {
  const std::size_t NG = xs_->ngroups();

  // Initialize to ones in case we are running P1
  xt::xtensor<double, 1> a = xt::ones<double>({NG});

  // First, we create and fill the Dinvs matrix for the current
  Eigen::MatrixXd Dinvs(NG, NG);

  // Declare the D matrix
  Eigen::MatrixXd D;

  // Make the chi and vEf vectors
  Eigen::VectorXd chi(NG), vEf(NG);
  for (std::size_t g = 0; g < NG; g++) {
    chi(g) = xs_->chi(g);
    vEf(g) = xs_->vEf(g);
  }

  // Make vectors for flux and current
  Eigen::VectorXd flx(NG);
  Eigen::VectorXd cur(NG);

  // Create the A matrix
  Eigen::MatrixXd A(NG, NG);

  // Get A for k_inf
  B2_ = 0.;
  if (B1) {
    fill_alphas(a, xs_, B2_);
  }
  // On first P1 iteration, we must fill Dinvs and D
  fill_Dinvs(Dinvs, xs_, a);
  D = Dinvs.inverse();
  fill_A(A, xs_, D, B2_);
  auto A_solver = A.colPivHouseholderQr();
  flx = A_solver.solve(chi);
  k_inf_ = vEf.dot(flx);

  // Get A for small B2, from Stamm'ler and Abbate
  B2_ = 0.001;
  if (k_inf_ < 1.) B2_ = -B2_;
  if (B1) {
    fill_alphas(a, xs_, B2_);
    fill_Dinvs(Dinvs, xs_, a);
    D = Dinvs.inverse();
  }
  fill_A(A, xs_, D, B2_);
  A_solver = A.colPivHouseholderQr();
  flx = A_solver.solve(chi);
  const double k_1 = vEf.dot(flx);

  // Calculate the slope constant
  const double k_inf_M2 = B2_ / ((1. / k_1) - (1. / k_inf_));

  double k = k_1;
  while (std::abs(k - 1.) > 1.E-6) {
    B2_ += k_inf_M2 * (1. - (1. / k));
    if (B1) {
      fill_alphas(a, xs_, B2_);
      fill_Dinvs(Dinvs, xs_, a);
      D = Dinvs.inverse();
    }
    fill_A(A, xs_, D, B2_);
    A_solver = A.colPivHouseholderQr();
    flx = A_solver.solve(chi);
    k = vEf.dot(flx);
  }

  // We have converged on k = 1.
  compute_flux_current_diff_coeffs(flx, cur, D, flux_, current_, diff_coeff_,
                                   B2_);
}

void CriticalitySpectrumWithCurrent::P1_B1_provided_buckling(bool B1) {
  const std::size_t NG = xs_->ngroups();

  // Initialize to ones in case of P1
  xt::xtensor<double, 1> a = xt::ones<double>({NG});

  // First, we create and fill the Dinvs matrix for the current
  Eigen::MatrixXd Dinvs(NG, NG);

  // Declare the D matrix
  Eigen::MatrixXd D;

  // Make the chi and vEf vectors
  Eigen::VectorXd chi(NG), vEf(NG);
  for (std::size_t g = 0; g < NG; g++) {
    chi(g) = xs_->chi(g);
    vEf(g) = xs_->vEf(g);
  }

  // Make vectors for flux and current
  Eigen::VectorXd flx(NG);
  Eigen::VectorXd cur(NG);

  // Create the A matrix
  Eigen::MatrixXd A(NG, NG);

  if (B1) fill_alphas(a, xs_, B2_);
  fill_Dinvs(Dinvs, xs_, a);
  D = Dinvs.inverse();
  fill_A(A, xs_, D, B2_);
  auto A_solver = A.colPivHouseholderQr();
  flx = A_solver.solve(chi);
  k_inf_ = vEf.dot(flx);

  compute_flux_current_diff_coeffs(flx, cur, D, flux_, current_, diff_coeff_,
                                   B2_);
}

P1CriticalitySpectrum::P1CriticalitySpectrum(std::shared_ptr<CrossSection> xs) {
  if (xs->fissile() == false) {
    std::stringstream mssg;
    mssg << "Cannot compute P1 spectrum of homogenized material that is not "
            "fissile.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // Save copy of cross section
  xs_ = xs;

  this->P1_B1_spectrum_search(false);
}

P1CriticalitySpectrum::P1CriticalitySpectrum(std::shared_ptr<CrossSection> xs,
                                             double B2) {
  xs_ = xs;
  B2_ = B2;

  this->P1_B1_provided_buckling(false);
}

P1CriticalitySpectrum::P1CriticalitySpectrum(py::tuple t)
    : CriticalitySpectrumWithCurrent(t) {}

py::tuple P1CriticalitySpectrum::to_tuple() const {
  return CriticalitySpectrumWithCurrent::to_tuple();
}

B1CriticalitySpectrum::B1CriticalitySpectrum(std::shared_ptr<CrossSection> xs) {
  if (xs->fissile() == false) {
    std::stringstream mssg;
    mssg << "Cannot compute B1 spectrum of homogenized material that is not "
            "fissile.";
    spdlog::error(mssg.str());
    throw ScarabeeException(mssg.str());
  }

  // Save copy of cross section
  xs_ = xs;

  this->P1_B1_spectrum_search(true);
}

B1CriticalitySpectrum::B1CriticalitySpectrum(std::shared_ptr<CrossSection> xs,
                                             double B2) {
  xs_ = xs;
  B2_ = B2;

  this->P1_B1_provided_buckling(true);
}

B1CriticalitySpectrum::B1CriticalitySpectrum(py::tuple t)
    : CriticalitySpectrumWithCurrent(t) {}

py::tuple B1CriticalitySpectrum::to_tuple() const {
  return CriticalitySpectrumWithCurrent::to_tuple();
}

}  // namespace scarabee
