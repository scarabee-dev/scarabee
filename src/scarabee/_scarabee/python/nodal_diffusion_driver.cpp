#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <xtensor-python/pytensor.hpp>

#include <diffusion/nodal_diffusion_driver.hpp>
#include <diffusion/finite_difference.hpp>
#include <diffusion/nem4.hpp>

namespace py = pybind11;

using namespace scarabee;

template <NodalMethod NM>
void init_NodalDiffusionDriver(py::module& m, const char* class_name,
                               const char* description) {
  using NodalSolver = NodalDiffusionDriver<NM>;

  py::class_<NodalSolver>(m, class_name, description)

      .def(py::init<std::shared_ptr<DiffusionGeometry> /*geom*/>(),
           "Initializes a nodal diffusion solver.\n\n"
           "Parameters\n"
           "----------\n"
           "geom : DiffusionGeometry\n"
           "       Problem deffinition to solve.")

      .def("solve", &NodalSolver::solve, "Solves the diffusion problem.")

      .def_property_readonly(
          "geometry", &NodalSolver::geometry,
          "The :py:class:`DiffusionGeometry` geometry for the problem.")

      .def_property_readonly("ngroups", &NodalSolver::ngroups,
                             "Number of energy groups.")

      .def_property_readonly(
          "solved", &NodalSolver::solved,
          "True if the problem has been solved, False otherwise.")

      .def_property_readonly(
          "keff", &NodalSolver::keff,
          "Value of keff. This is 1 by default is solved is False.")

      .def_property("keff_tolerance", &NodalSolver::keff_tolerance,
                    &NodalSolver::set_keff_tolerance,
                    "Maximum relative error in keff for problem convergence.")

      .def_property(
          "flux_tolerance", &NodalSolver::flux_tolerance,
          &NodalSolver::set_flux_tolerance,
          "Maximum relative error in the flux for problem convergence.")

      .def_property(
          "nonlinear_update_frequency",
          &NodalSolver::nonlinear_update_frequency,
          &NodalSolver::set_nonlinear_update_frequency,
          "Frequency at which nonlinear diffusion coefficients are updated.")

      .def_property("leakage_corrections", &NodalSolver::leakage_corrections,
                    &NodalSolver::set_leakage_corrections,
                    "Apply leakage corrections to update node cross sections.");

  //.def("flux",
  //     py::overload_cast<double /*x*/, double /*y*/, double /*z*/,
  //                       std::size_t /*g*/>(&NEMDiffusionDriver::flux,
  //                                          py::const_),
  //     "Calculates the flux at the desired position and group. The "
  //     "lowest value for any coordinate is 0.\n\n"
  //     "Parameters\n"
  //     "----------\n"
  //     "x : float\n"
  //     "    Position along the x axis.\n"
  //     "y : float\n"
  //     "    Position along the y axis.\n"
  //     "z : float\n"
  //     "    Position along the z axis.\n"
  //     "g : ing\n"
  //     "    Energy group index.\n\n"
  //     "Returns\n"
  //     "-------\n"
  //     "float\n"
  //     "      Value of the flux.\n",
  //     py::arg("x"), py::arg("y"), py::arg("z"), py::arg("g"))

  //.def("flux",
  //     py::overload_cast<const xt::xtensor<double, 1>& /*x*/,
  //                       const xt::xtensor<double, 1>& /*y*/,
  //                       const xt::xtensor<double, 1>& /*z*/>(
  //         &NEMDiffusionDriver::flux, py::const_),
  //     "Constructs an array storing the flux at all desired (x,y,z) "
  //     "points and at all energy groups. The first index is the group, "
  //     "the second is x, the third is y, and the fourth is z.\n\n"
  //     "Parameters\n"
  //     "----------\n"
  //     "x : array of float\n"
  //     "    Positions along the x axis.\n"
  //     "y : array of float\n"
  //     "    Positions along the y axis.\n"
  //     "z : array of float\n"
  //     "    Positions along the z axis.\n\n"
  //     "Returns\n"
  //     "-------\n"
  //     "array of float\n"
  //     "      Value of the flux at all (g,x,y,z).\n",
  //     py::arg("x"), py::arg("y"), py::arg("z"))

  //.def("avg_flux", &NEMDiffusionDriver::avg_flux,
  //     "Constructs an array storing the value of the average flux in "
  //     "each node. The resulting array is indexed as (g, x, y, z).\n\n"
  //     "Returns\n"
  //     "-------\n"
  //     "array of float\n"
  //     "      Value of the average flux in each node.\n")

  //.def("power",
  //     py::overload_cast<double /*x*/, double /*y*/, double /*z*/>(
  //         &NEMDiffusionDriver::power, py::const_),
  //     "Calculates the power density at the desired position. The lowest "
  //     "value for any coordinate is 0.\n\n"
  //     "Parameters\n"
  //     "----------\n"
  //     "x : float\n"
  //     "    Position along the x axis.\n"
  //     "y : float\n"
  //     "    Position along the y axis.\n"
  //     "z : float\n"
  //     "    Position along the z axis.\n\n"
  //     "Returns\n"
  //     "-------\n"
  //     "float\n"
  //     "      Value of the power density.\n",
  //     py::arg("x"), py::arg("y"), py::arg("z"))

  //.def("power",
  //     py::overload_cast<const xt::xtensor<double, 1>& /*x*/,
  //                       const xt::xtensor<double, 1>& /*y*/,
  //                       const xt::xtensor<double, 1>& /*z*/>(
  //         &NEMDiffusionDriver::power, py::const_),
  //     "Constructs an array storing the power density at all desired "
  //     "(x,y,z) points. The first index is x, the second is y, and the "
  //     "third is z.\n\n"
  //     "Parameters\n"
  //     "----------\n"
  //     "x : array of float\n"
  //     "    Positions along the x axis.\n"
  //     "y : array of float\n"
  //     "    Positions along the y axis.\n"
  //     "z : array of float\n"
  //     "    Positions along the z axis.\n\n"
  //     "Returns\n"
  //     "-------\n"
  //     "array of float\n"
  //     "      Value of the power density at all (x,y,z).\n",
  //     py::arg("x"), py::arg("y"), py::arg("z"))

  //.def("avg_power", &NEMDiffusionDriver::avg_power,
  //     "Constructs an array storing the value of the average power "
  //     "density in each node. The resulting array is indexed as "
  //     "(x, y, z).\n\n"
  //     "Returns\n"
  //     "-------\n"
  //     "array of float\n"
  //     "      Value of the average power density in each node.\n")

  //.def("save", &NEMDiffusionDriver::save,
  //     "Saves the NEMDiffusionDriver to a binary file.\n\n"
  //     "Parameters\n"
  //     "----------\n"
  //     "fname : str\n"
  //     "  Name of the file.\n",
  //     py::arg("fname"))

  //.def_static(
  //    "load", &NEMDiffusionDriver::load,
  //    "Loads a previously save NEMDiffusionDriver from a binary file.\n\n"
  //    "Parameters\n"
  //    "----------\n"
  //    "fname : str\n"
  //    "  Name of the file.\n\n"
  //    "Returns\n"
  //    "-------\n"
  //    "NEMDiffusionDriver",
  //    py::arg("fname"));
}

void init_all_NodalDiffusionDrivers(py::module& m) {
  init_NodalDiffusionDriver<FiniteDifference>(
      m, "FDNodalDiffusionDriver",
      "A FDNodalDiffusionDriver solves a diffusion problem using the finite "
      "difference formalism, but inside the nodal solver shell. Therefore, a "
      "spatial discretization compatible with this method should be used. The "
      "geometry is defined using a :py:class:`DiffusionGeometry` instance.");

  init_NodalDiffusionDriver<NEM4>(
      m, "NEM4DiffusionDriver",
      "Solves a diffusion problem using the 4th order Nodal Expansion Method. "
      "Uses a standard quadratic transverse leakage approximation. Can be used "
      "with assembly or half assembly sized nodes. The geometry is defined "
      "using a :py:class:`DiffusionGeometry` instance.");
}
