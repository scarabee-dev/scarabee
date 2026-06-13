#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <xtensor-python/pytensor.hpp>

#include <utils/tab1.hpp>

namespace py = pybind11;

using namespace scarabee;

void init_Tab1(py::module& m) {
  py::class_<Tab1, std::shared_ptr<Tab1>>(m, "Tab1")
      .def(py::init<const std::vector<double>& /*x*/,
                    const std::vector<double>& /*y*/>(),
           "Creates a Tab1 with Lin-Lin interpolation.\n\n"
           "Parameters\n"
           "----------\n"
           "x : list of float\n"
           "    x values\n"
           "y : list of float\n"
           "    y values\n\n",
           py::arg("x"), py::arg("y"))

      .def(py::init<const xt::xtensor<double, 1>& /*x*/,
                    const xt::xtensor<double, 1>& /*y*/>(),
           "Creates a Tab1 with Lin-Lin interpolation.\n\n"
           "Parameters\n"
           "----------\n"
           "x : np.ndarray of float\n"
           "    x values\n"
           "y : np.ndarray of float\n"
           "    y values\n\n",
           py::arg("x"), py::arg("y"))

      .def(py::init<const std::vector<double>& /*x*/,
                    const std::vector<double>& /*y*/,
                    const std::vector<std::size_t>& /*breakpoints*/,
                    const std::vector<int>& /*interpolations*/>(),
           "Creates a Tab1 with specified interpolation regions.\n\n"
           "Parameters\n"
           "----------\n"
           "x : list of float\n"
           "    x values\n"
           "y : list of float\n"
           "    y values\n",
           "breakpoints : list of int\n"
           "    The upper index + 1 (Fortran-like index) of each interpolation "
           "region.\n"
           "interpolations : list of int\n"
           "    Interpolation rule for each region. Each entry must be in "
           "[1,5].\n\n",
           py::arg("x"), py::arg("y"), py::arg("breakpoints"),
           py::arg("interpolations"))

      .def(py::init<const xt::xtensor<double, 1>& /*x*/,
                    const xt::xtensor<double, 1>& /*y*/,
                    const std::vector<std::size_t>& /*breakpoints*/,
                    const std::vector<int>& /*interpolations*/>(),
           "Creates a Tab1 with specified interpolation regions.\n\n"
           "Parameters\n"
           "----------\n"
           "x : np.ndarray of float\n"
           "    x values\n"
           "y : np.ndarray of float\n"
           "    y values\n",
           "breakpoints : list of int\n"
           "    The upper index + 1 (Fortran-like index) of each interpolation "
           "region.\n"
           "interpolations : list of int\n"
           "    Interpolation rule for each region. Each entry must be in "
           "[1,5].\n\n",
           py::arg("x"), py::arg("y"), py::arg("breakpoints"),
           py::arg("interpolations"))

      .def_property_readonly("x", &Tab1::x, "Array of x values.")
      .def_property_readonly("y", &Tab1::y, "Array of y values.")
      .def_property_readonly("breakpoints", &Tab1::breakpoints,
                             "List of interpolation region break points with "
                             "fortran-like indexing.")
      .def_property_readonly("interpolations", &Tab1::interpolations,
                             "List of interpolation rules for each region.")

      .def("__call__",
           py::overload_cast<const double /*x*/>(&Tab1::operator(), py::const_),
           "Evaluates the function at a given value.\n\n"
           "Parameters\n"
           "----------\n"
           "x : float\n\n"
           "Returns\n"
           "-------\n"
           "float\n\n",
           py::arg("x"))

      .def("__call__",
           py::overload_cast<const xt::pytensor<double, 1>& /*x*/>(
               &Tab1::operator(), py::const_),
           "Evaluates the function at all given values.\n\n"
           "Parameters\n"
           "----------\n"
           "x : np.ndarray of float\n\n"
           "Returns\n"
           "-------\n"
           "np.ndarray of float\n\n",
           py::arg("x"))

      .def("integrate",
           py::overload_cast<const double /*a*/, const double /*b*/>(
               &Tab1::integrate, py::const_),
           "Integrates the function over a specified domain.\n\n"
           "Parameters\n"
           "----------\n"
           "a : float\n"
           "    Lower bound of integration\n"
           "b : float\n"
           "    Upper bound of integration\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "    Integral from a to b.",
           py::arg("a"), py::arg("b"));
}
