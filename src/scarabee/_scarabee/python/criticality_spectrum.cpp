#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <xtensor-python/pytensor.hpp>

#include <cereal/archives/portable_binary.hpp>

#include <utils/criticality_spectrum.hpp>

#include <sstream>

namespace py = pybind11;

using namespace scarabee;

struct FundamentalModeCriticalitySpectrumPickler {
  static FundamentalModeCriticalitySpectrum from_state(py::tuple t) {
    FundamentalModeCriticalitySpectrum fm;
    fm.xs_ = t[0].cast<std::shared_ptr<CrossSection>>();
    py::bytes bytes = t[1].cast<py::bytes>();
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(fm.flux_, fm.diff_coeff_, fm.k_inf_, fm.B2_);
    }

    return fm;
  }

  static py::tuple to_state(const FundamentalModeCriticalitySpectrum& fm) {
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(fm.flux_, fm.diff_coeff_, fm.k_inf_, fm.B2_);
    }
    py::bytes bytes(bits_stream.str());

    return py::make_tuple(fm.xs_, bytes);
  }
};

struct B1CriticalitySpectrumPickler {
  static B1CriticalitySpectrum from_state(py::tuple t) {
    B1CriticalitySpectrum b1;
    b1.xs_ = t[0].cast<std::shared_ptr<CrossSection>>();
    py::bytes bytes = t[1].cast<py::bytes>();
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(b1.flux_, b1.diff_coeff_, b1.k_inf_, b1.B2_, b1.current_);
    }

    return b1;
  }

  static py::tuple to_state(const B1CriticalitySpectrum& b1) {
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(b1.flux_, b1.diff_coeff_, b1.k_inf_, b1.B2_, b1.current_);
    }
    py::bytes bytes(bits_stream.str());

    return py::make_tuple(b1.xs_, bytes);
  }
};

struct P1CriticalitySpectrumPickler {
  static P1CriticalitySpectrum from_state(py::tuple t) {
    P1CriticalitySpectrum p1;
    p1.xs_ = t[0].cast<std::shared_ptr<CrossSection>>();
    py::bytes bytes = t[1].cast<py::bytes>();
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(p1.flux_, p1.diff_coeff_, p1.k_inf_, p1.B2_, p1.current_);
    }

    return p1;
  }

  static py::tuple to_state(const P1CriticalitySpectrum& p1) {
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(p1.flux_, p1.diff_coeff_, p1.k_inf_, p1.B2_, p1.current_);
    }
    py::bytes bytes(bits_stream.str());

    return py::make_tuple(p1.xs_, bytes);
  }
};

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

      .def(py::pickle(&FundamentalModeCriticalitySpectrumPickler::to_state,
                      &FundamentalModeCriticalitySpectrumPickler::from_state));

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

      .def(py::pickle(&P1CriticalitySpectrumPickler::to_state,
                      &P1CriticalitySpectrumPickler::from_state));

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

      .def(py::pickle(&B1CriticalitySpectrumPickler::to_state,
                      &B1CriticalitySpectrumPickler::from_state));
}
