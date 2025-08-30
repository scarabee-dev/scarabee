#include <utils/spherical_harmonics.hpp>
#include <utils/constants.hpp>
#include <utils/math.hpp>

namespace scarabee {

SphericalHarmonics::SphericalHarmonics(
    const std::size_t& L, const std::vector<double>& azimuthal_angle,
    const std::vector<double>& polar_angle)
    : L_(L), Nlj_((L_ + 1) * (L_ + 1)) {
  // the polar_angle contains half of the angle between [0, PI/2]
  // therefor, evaluate the remaining half between [PI/2 , PI]
  // if the indexing of angle between [0, PI/2] is i, then
  // index of next half will be (no of angle ) + i
  // Similarly, azimuthal angles are given between [0, PI] corresponds to
  // forward direction therefore, elvauate the backward direction entry
  // azimuthal angles
  std::size_t n_polar_angle = polar_angle.size();
  double theta;

  const std::size_t n_azimuthal_angle = azimuthal_angle.size();
  double phi;

  all_harmonics_.resize(
      {2 * n_azimuthal_angle, 2 * n_polar_angle, (L_ + 1) * (L_ + 1)});
  all_harmonics_.fill(0.);

  for (std::size_t azm = 0; azm < 2 * n_azimuthal_angle; azm++) {
    if (azm < n_azimuthal_angle)
      phi = azimuthal_angle[azm];
    else
      phi = azimuthal_angle[azm - n_azimuthal_angle] + PI;

    for (std::size_t p = 0; p < 2 * n_polar_angle; p++) {
      if (p < n_polar_angle)
        theta = polar_angle[p];
      else
        theta = PI - polar_angle[p - n_polar_angle];

      std::size_t it_lj = 0;
      for (unsigned int l = 0; l <= L_; l++) {
        for (int j = -static_cast<int>(l); j <= static_cast<int>(l); j++) {
          all_harmonics_(azm, p, it_lj) = spherical_hamonics(l, j, phi, theta);
          it_lj++;
        }

      }  // all scattering moments
    }  // all polar angles
  }  // all azimuthal angles
}

}  // namespace scarabee
