#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <xtensor-python/pytensor.hpp>

#include <diffusion/fd_diffusion_driver.hpp>

namespace py = pybind11;

using namespace scarabee;

void init_FDDiffusionDriver(py::module& m) {
  py::class_<FDDiffusionDriver>(
      m, "FDDiffusionDriver",
      "A FDDiffusionDriver solves a diffusion problem using the cell centered "
      "finite-difference method. It is capable of solving 1D, 2D, or 3D "
      "problems which are defined by providing a "
      ":py:class:`DiffusionGeometry` instance.")

      .def(py::init<std::shared_ptr<DiffusionGeometry> /*geom*/>(),
           "Initializes a finite-difference diffusion solver.\n\n"
           "Parameters\n"
           "----------\n"
           "geom : DiffusionGeometry\n"
           "       Problem deffinition to solve.")

      .def("solve", &FDDiffusionDriver::solve, "Solves the diffusion problem.")

      .def_property_readonly(
          "geometry", &FDDiffusionDriver::geometry,
          "The :py:class:`DiffusionGeometry` geometry for the problem.")

      .def_property_readonly("ngroups", &FDDiffusionDriver::ngroups,
                             "Number of energy groups.")

      .def_property_readonly(
          "solved", &FDDiffusionDriver::solved,
          "True if the problem has been solved, False otherwise.")

      .def_property_readonly(
          "keff", &FDDiffusionDriver::keff,
          "Value of keff. This is 1 by default is solved is False.")

      .def_property("keff_tolerance", &FDDiffusionDriver::keff_tolerance,
                    &FDDiffusionDriver::set_keff_tolerance,
                    "Maximum relative error in keff for problem convergence.")

      .def_property(
          "flux_tolerance", &FDDiffusionDriver::flux_tolerance,
          &FDDiffusionDriver::set_flux_tolerance,
          "Maximum relative error in the flux for problem convergence.")

      .def_property(
          "sim_mode",
          [](const FDDiffusionDriver& fdd) -> SimulationMode {
            return fdd.sim_mode();
          },
          [](FDDiffusionDriver& fdd, SimulationMode& m) { fdd.sim_mode() = m; },
          ":py:class:`SimulationMode` describing type of simulation "
          "(fixed-source or keff).")

      .def("set_extern_src",
           py::overload_cast<std::size_t, std::size_t, double>(
               &FDDiffusionDriver::set_extern_src),
           "Sets the external source in group g and region i.\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Region index along x axis.\n"
           "g : int\n"
           "    Energy group index.\n"
           "src : float\n"
           "      Value of source.\n",
           py::arg("i"), py::arg("g"), py::arg("src"))

      .def("set_extern_src",
           py::overload_cast<std::size_t, std::size_t, std::size_t, double>(
               &FDDiffusionDriver::set_extern_src),
           "Sets the external source in group g and region (i, j).\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Region index along x axis.\n"
           "j : int\n"
           "    Region index along y axis.\n"
           "g : int\n"
           "    Energy group index.\n"
           "src : float\n"
           "      Value of source.\n",
           py::arg("i"), py::arg("j"), py::arg("g"), py::arg("src"))

      .def("set_extern_src",
           py::overload_cast<std::size_t, std::size_t, std::size_t, std::size_t,
                             double>(&FDDiffusionDriver::set_extern_src),
           "Sets the external source in group g and region (i, j, k).\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Region index along x axis.\n"
           "j : int\n"
           "    Region index along y axis.\n"
           "k : int\n"
           "    Region index along z axis.\n"
           "g : int\n"
           "    Energy group index.\n"
           "src : float\n"
           "      Value of source.\n",
           py::arg("i"), py::arg("j"), py::arg("k"), py::arg("g"),
           py::arg("src"))

      .def("extern_src",
           py::overload_cast<std::size_t, std::size_t>(
               &FDDiffusionDriver::extern_src, py::const_),
           "Returns the external source in group g and region i.\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Region index along x axis.\n"
           "g : int\n"
           "    Energy group index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Value of the external source.\n",
           py::arg("i"), py::arg("g"))

      .def("extern_src",
           py::overload_cast<std::size_t, std::size_t, std::size_t>(
               &FDDiffusionDriver::extern_src, py::const_),
           "Returns the external source in group g and region (i, j).\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Region index along x axis.\n"
           "j : int\n"
           "    Region index along y axis.\n"
           "g : int\n"
           "    Energy group index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Value of the external source.\n",
           py::arg("i"), py::arg("j"), py::arg("g"))

      .def(
          "extern_src",
          py::overload_cast<std::size_t, std::size_t, std::size_t, std::size_t>(
              &FDDiffusionDriver::extern_src, py::const_),
          "Returns the external source in group g and region (i, j, k).\n\n"
          "Parameters\n"
          "----------\n"
          "i : int\n"
          "    Region index along x axis.\n"
          "j : int\n"
          "    Region index along y axis.\n"
          "k : int\n"
          "    Region index along z axis.\n"
          "g : int\n"
          "    Energy group index.\n\n"
          "Returns\n"
          "-------\n"
          "float\n"
          "     Value of the external source.\n",
          py::arg("i"), py::arg("j"), py::arg("k"), py::arg("g"))

      .def("flux",
           py::overload_cast<std::size_t, std::size_t>(&FDDiffusionDriver::flux,
                                                       py::const_),
           "Returns the scalar flux in group g and region i.\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Region index along x axis.\n"
           "g : int\n"
           "    Energy group index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Flux in region i and in group g.\n",
           py::arg("i"), py::arg("g"))

      .def("flux",
           py::overload_cast<std::size_t, std::size_t, std::size_t>(
               &FDDiffusionDriver::flux, py::const_),
           "Returns the scalar flux in group g and region (i, j).\n\n"
           "Parameters\n"
           "----------\n"
           "i : int\n"
           "    Region index along x axis.\n"
           "j : int\n"
           "    Region index along y axis.\n"
           "g : int\n"
           "    Energy group index.\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "     Flux in region (i, j) and in group g.\n",
           py::arg("i"), py::arg("j"), py::arg("g"))

      .def(
          "flux",
          py::overload_cast<std::size_t, std::size_t, std::size_t, std::size_t>(
              &FDDiffusionDriver::flux, py::const_),
          "Returns the scalar flux in group g and region (i, j, k).\n\n"
          "Parameters\n"
          "----------\n"
          "i : int\n"
          "    Region index along x axis.\n"
          "j : int\n"
          "    Region index along y axis.\n"
          "k : int\n"
          "    Region index along z axis.\n"
          "g : int\n"
          "    Energy group index.\n\n"
          "Returns\n"
          "-------\n"
          "float\n"
          "     Flux in region (i, j, k) and in group g.\n",
          py::arg("i"), py::arg("j"), py::arg("k"), py::arg("g"))

      .def("flux", py::overload_cast<>(&FDDiffusionDriver::flux, py::const_),
           "Returns the computed flux, along with the mesh bounds. The first "
           "dimension "
           "of the flux array is the energy group index. The second index is "
           "for the x "
           "coordinate. If the problem is 2 or 3 dimensional, the third and "
           "fourth "
           "indices are for the y and z coordinates respectively.\n\n"
           "Returns\n"
           "-------\n"
           "flux : ndarray\n"
           "       2D, 3D, or 4D array containing the multigroup flux.\n"
           "x_bounds : ndarray\n"
           "           1D array with the x-bounds for the flux mesh.\n"
           "y_bounds : ndarray or None\n"
           "           1D array with the y-bounds for the flux mesh, if a 2D "
           "problem.\n"
           "z_bounds : ndarray or None\n"
           "           1D array with the z-bounds for the flux mesh, if a 3D "
           "problem.\n")

      .def("power", &FDDiffusionDriver::power,
           "Returns the computed power distribution, along with the mesh "
           "bounds. The first dimension of the power array is for the x "
           "coordinate. If the problem is 2 or 3 dimensional, the second and "
           "third indices are for the y and z coordinates respectively.\n\n"
           "Returns\n"
           "-------\n"
           "power : ndarray\n"
           "        1D, 2D, or 3D array containing the power distribution.\n"
           "x_bounds : ndarray\n"
           "           1D array with the x-bounds for the power mesh.\n"
           "y_bounds : ndarray or None\n"
           "           1D array with the y-bounds for the power mesh, if a 2D "
           "problem.\n"
           "z_bounds : ndarray or None\n"
           "           1D array with the z-bounds for the power mesh, if a 3D "
           "problem.\n")

      .def("save", &FDDiffusionDriver::save,
           "Saves the FDDiffusionDriver to a binary file.\n\n"
           "Parameters\n"
           "----------\n"
           "fname : str\n"
           "  Name of the file.\n",
           py::arg("fname"))

      .def_static(
          "load", &FDDiffusionDriver::load,
          "Loads a previously save FDDiffusionDriver from a binary file.\n\n"
          "Parameters\n"
          "----------\n"
          "fname : str\n"
          "  Name of the file.\n\n"
          "Returns\n"
          "-------\n"
          "FDDiffusionDriver",
          py::arg("fname"));
}
