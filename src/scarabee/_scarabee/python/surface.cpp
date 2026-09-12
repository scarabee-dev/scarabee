#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <moc/surface.hpp>

#include <memory>

namespace py = pybind11;

using namespace scarabee;

/* This class cannot be instantiated on the Python side, and is only bound to
 * facilitate the complete (de)pickling of Scarabee classes while preserving all
 * internal references between different MOC / geometry objects.
 */

void init_Surface(py::module& m) {
  py::class_<Surface, std::shared_ptr<Surface>>(m, "Surface")
      .def("side", &Surface::side)
      .def("distance", &Surface::distance)
      .def("integrate_x", &Surface::integrate_x)
      .def("integrate_y", &Surface::integrate_y)
      .def(
          py::pickle([](std::shared_ptr<Surface> s) { return s->to_tuple(); },
                     [](py::tuple t) { return std::make_shared<Surface>(t); }));
}
