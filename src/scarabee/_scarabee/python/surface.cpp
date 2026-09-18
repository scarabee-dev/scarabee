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

struct SurfacePickler {
  static std::shared_ptr<Surface> from_state(py::tuple t) {
    std::shared_ptr<Surface> s(new Surface);
    s->type_ = static_cast<Surface::Type>(t[0].cast<char>());
    s->params_ = t[0].cast<std::array<double, 5>>();
    return s;
  }

  static py::tuple to_state(const std::shared_ptr<Surface>& s) {
    return py::make_tuple(static_cast<char>(s->type_), s->params_);
  }
};

void init_Surface(py::module& m) {
  py::class_<Surface, std::shared_ptr<Surface>>(m, "Surface")
      .def("side", &Surface::side)
      .def("distance", &Surface::distance)
      .def("integrate_x", &Surface::integrate_x)
      .def("integrate_y", &Surface::integrate_y)
      .def(py::pickle(&SurfacePickler::to_state, &SurfacePickler::from_state));
}
