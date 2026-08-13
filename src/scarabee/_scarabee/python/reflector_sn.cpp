#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <xtensor-python/pytensor.hpp>

#include <reflector_sn.hpp>

namespace py = pybind11;

using namespace scarabee;

void init_ReflectorSN(py::module& m) {
  py::class_<ReflectorSN>(m, "ReflectorSN")
      .def(py::init<const std::vector<std::shared_ptr<CrossSection>>& /*xs*/,
                    const xt::xtensor<double, 1>& /*dx*/,
                    std::size_t /*nangles*/, bool /*anisotropic*/>(),
           "Creates a ReflectorSN object to perform a 1D Sn calculation with "
           "a reflective boundary at the -x side and a vacuum boundary on the "
           "+x side.\n\n"
           "Parameters\n"
           "----------\n"
           "xs : iterable of CrossSection\n"
           "  1D iterable for the cross section in each bin.\n"
           "dx : iterable of float\n"
           "  1D iterable of the widths of each bin along x.\n"
           "nangles : int\n"
           "  Number of angles tracked. Must be one of the following:\n"
           "  2, 4, 6, 8, 10, 12, 14, 16, 32, 64, 128.\n"
           "anisotropic : bool\n"
           "  True to use anisotropic scattering, False to use the transport "
           "correction (default value is True).\n",
           py::arg("xs"), py::arg("dx"), py::arg("nangles"),
           py::arg("anisotropic") = true)

      .def("solve", &ReflectorSN::solve,
           py::call_guard<py::gil_scoped_release>(),
           "Begins iterations to solve problem.")

      .def_property_readonly("nangles", &ReflectorSN::nangles,
                             "Number of discrete angles tracked.")

      .def_property_readonly(
          "solved", &ReflectorSN::solved,
          "True if problem has been solved, False otherwsie.")

      .def_property_readonly("keff", &ReflectorSN::keff,
                             "Value of keff estimated by solver (1 by default "
                             "if no solution has been obtained).")

      .def_property_readonly("ngroups", &ReflectorSN::ngroups,
                             "Number of energy groups.")

      .def_property_readonly("size", &ReflectorSN::size, "Number of regions.")

      .def_property_readonly("nregions", &ReflectorSN::nregions,
                             "Number of regions.")

      .def_property_readonly("nsurfaces", &ReflectorSN::nsurfaces,
                             "Number of surfaces.")

      .def_property_readonly("max_legendre_order",
                             &ReflectorSN::max_legendre_order,
                             "Maximum legendre order for scattering.")

      .def_property_readonly(
          "anisotropic", &ReflectorSN::anisotropic,
          "If True, anisotropic scattering will be simulated.")

      .def_property(
          "keff_tolerance", &ReflectorSN::keff_tolerance,
          &ReflectorSN::set_keff_tolerance,
          "Maximum relative absolute difference in keff for convergence")

      .def_property(
          "flux_tolerance", &ReflectorSN::flux_tolerance,
          &ReflectorSN::set_flux_tolerance,
          "Maximum relative absolute difference in flux for convergence")

      .def("flux", &ReflectorSN::flux,
           "Returns the scalar flux in group g in mesh region i.\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Region index.\n"
           "g : int\n"
           "    Energy group index.\n"
           "l : int\n"
           "    Legendre moment (default value is 0).\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Flux in region i and in group g for legendre moment l.\n",
           py::arg("i"), py::arg("g"), py::arg("l") = 0)

      .def("current", &ReflectorSN::current,
           "Returns the net current in group g at surface i.\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Surface index.\n"
           "g : int\n"
           "    Energy group index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Net current at surface i and in group g.\n",
           py::arg("i"), py::arg("g"))

      .def("volume", &ReflectorSN::volume,
           "Returns the volume of region i.\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Region index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Volume of region i.\n",
           py::arg("i"))

      .def("xs", &ReflectorSN::xs,
           "Returns the CrossSection in region i.\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Region index.\n\n"
           "Returns\n"
           "-------\n"
           "CrossSection\n"
           "    Material cross sections at r.\n",
           py::arg("i"))

      .def("homogenize", &ReflectorSN::homogenize,
           "Computes a homogenized set of cross sections for the set of "
           "provided region indices.\n\n"
           "Parameters\n"
           "----------\n"
           "regions : list of int\n"
           "          List of regions for homogenization.\n"
           "Returns\n"
           "-------\n"
           ":py:class:`CrossSection`\n"
           "        Homogenized cross section.\n",
           py::arg("regions"))

      .def("homogenize_flux_spectrum", &ReflectorSN::homogenize_flux_spectrum,
           "Computes a homogenized flux spectrum based on the list of "
           "provided region indices. This method will raise an exception if "
           "the problem has not yet been solved.\n\n"
           "Parameters\n"
           "----------\n"
           "regions : list of int\n"
           "          List of regions for homogenization.\n"
           "Returns\n"
           "-------\n"
           "ndarray of floats\n"
           "                 Homogenized flux spectrum.",
           py::arg("regions"));
}
