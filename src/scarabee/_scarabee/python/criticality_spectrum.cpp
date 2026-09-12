#include <pybind11/pybind11.h>
#include <xtensor-python/pytensor.hpp>

#include <utils/criticality_spectrum.hpp>

namespace py = pybind11;

using namespace scarabee;

void init_CriticalitySpectrum(py::module& m) {
  py::class_<CriticalitySpectrum>(m, "CriticalitySpectrum")
      .def_property_readonly("ngroups", &CriticalitySpectrum::ngroups,
                             "Number of energy groups")

      .def_property_readonly("k_inf", &CriticalitySpectrum::k_inf,
                             "Infinite multiplication factor")

      .def_property_readonly("B2", &CriticalitySpectrum::B2,
                             "Critical buckling :math:`B^2`")

      .def_property_readonly("buckling", &CriticalitySpectrum::buckling,
                             "Critical buckling :math:`B^2`")

      .def_property_readonly(
          "cross_section", &CriticalitySpectrum::cross_section,
          "CrossSection which was used in the spectrum calculation")

      .def_property_readonly(
          "flux", py::overload_cast<>(&CriticalitySpectrum::flux, py::const_),
          py::return_value_policy::reference_internal,
          "Array contianing the flux spectrum")

      .def_property_readonly(
          "diff_coeff",
          py::overload_cast<>(&CriticalitySpectrum::diff_coeff, py::const_),
          py::return_value_policy::reference_internal,
          "Array contianing the diffusion coefficients")

      .def("make_diffusion_cross_section",
           &CriticalitySpectrum::make_diffusion_cross_section,
           "Produces a DiffusionCrossSection based on the computed diffusion "
           "coefficients "
           "and the provided cross section instance used for the spectrum "
           "calculation.\n\n"
           "Returns\n"
           "-------\n"
           "DiffusionCrossSection\n"
           "    Computed diffusion coefficients.\n\n");

  py::class_<FundamentalModeCriticalitySpectrum, CriticalitySpectrum>(
      m, "FundamentalModeCriticalitySpectrum")
      .def(py::init<std::shared_ptr<CrossSection>>(),
           "Computes the criticality energy spectrum using the fundamental "
           "mode leakage approximation.\n\n"
           "Parameters\n"
           "----------\n"
           "xs : CrossSection\n"
           "     Homogenized set of cross sections for the system.\n\n",
           py::arg("xs"))

      .def(py::init<std::shared_ptr<CrossSection>, double>(),
           "Computes the flux and current energy spectrum using the "
           "fundamental mode leakage approximation for a given buckling.\n\n"
           "Parameters\n"
           "----------\n"
           "xs : CrossSection\n"
           "     Homogenized set of cross sections for the system.\n"
           "B2 : float\n"
           "    Desired value of the buckling.\n\n",
           py::arg("xs"), py::arg("B2"))

      .def(py::pickle(
          [](const FundamentalModeCriticalitySpectrum& fm) {
            return fm.to_tuple();
          },
          [](py::tuple t) { return FundamentalModeCriticalitySpectrum(t); }));

  py::class_<CriticalitySpectrumWithCurrent, CriticalitySpectrum>(
      m, "CriticalitySpectrumWithCurrent")
      .def_property_readonly(
          "current",
          py::overload_cast<>(&CriticalitySpectrumWithCurrent::current,
                              py::const_),
          py::return_value_policy::reference_internal,
          "Array contianing the current spectrum");

  py::class_<P1CriticalitySpectrum, CriticalitySpectrumWithCurrent>(
      m, "P1CriticalitySpectrum")
      .def(py::init<std::shared_ptr<CrossSection>>(),
           "Computes the criticality energy spectrum using the P1 leakage "
           "approximation.\n\n"
           "Parameters\n"
           "----------\n"
           "xs : CrossSection\n"
           "     Homogenized set of cross sections for the system.\n\n",
           py::arg("xs"))

      .def(py::init<std::shared_ptr<CrossSection>, double>(),
           "Computes the flux and current energy spectrum using the P1 leakage "
           "approximation for a given buckling.\n\n"
           "Parameters\n"
           "----------\n"
           "xs : CrossSection\n"
           "     Homogenized set of cross sections for the system.\n"
           "B2 : float\n"
           "    Desired value of the buckling.\n\n",
           py::arg("xs"), py::arg("B2"))

      .def(py::pickle(
          [](const P1CriticalitySpectrum& p1) { return p1.to_tuple(); },
          [](py::tuple t) { return P1CriticalitySpectrum(t); }));

  py::class_<B1CriticalitySpectrum, CriticalitySpectrumWithCurrent>(
      m, "B1CriticalitySpectrum")
      .def(py::init<std::shared_ptr<CrossSection>>(),
           "Computes the criticality energy spectrum using the B1 leakage "
           "approximation.\n\n"
           "Parameters\n"
           "----------\n"
           "xs : CrossSection\n"
           "     Homogenized set of cross sections for the system.\n\n",
           py::arg("xs"))

      .def(py::init<std::shared_ptr<CrossSection>, double>(),
           "Computes the flux and current energy spectrum using the B1 leakage "
           "approximation for a given buckling.\n\n"
           "Parameters\n"
           "----------\n"
           "xs : CrossSection\n"
           "     Homogenized set of cross sections for the system.\n"
           "B2 : float\n"
           "    Desired value of the buckling.\n\n",
           py::arg("xs"), py::arg("B2"))

      .def(py::pickle(
          [](const B1CriticalitySpectrum& b1) { return b1.to_tuple(); },
          [](py::tuple t) { return B1CriticalitySpectrum(t); }));
}
