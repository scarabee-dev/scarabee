#include <pybind11/pybind11.h>

#include <moc/direction.hpp>

#include <sstream>

namespace py = pybind11;

using namespace scarabee;

void init_Direction(py::module& m) {
  py::class_<Direction, Vector>(
      m, "Direction",
      "A Direction object inherits from a Vector, but is may only have a "
      "length of unity. It can also participate in vector arithmetic, but "
      "the result is always a Vector object. Like Vectors, Directions are "
      "also immutable.")

      .def(py::init<double, double>(),
           "Creates a Direction from x and y components. The components will "
           "be scaled such that the lengths of the is unity.\n\n"
           "Parameters\n"
           "----------\n"
           "x : float\n"
           "    x component of vector.\n"
           "y : float\n"
           "    y component of vector.\n\n",
           py::arg("x"), py::arg("y"))

      .def(py::init<double>(),
           "Creates a direction from an angle formed with the x-axis. The "
           "angle is measured by starting at the x-axis and moving counter "
           "clockwise.\n\n"
           "Parameters\n"
           "----------\n"
           "phi : float\n"
           "      Angle formed by the vector and the x-axis, in radians.\n\n",
           py::arg("phi"))

      .def(
          "__add__",
          [](const Direction& d1, const Direction& d2) { return d1 + d2; },
          "Returns the sum of two Directions as a Vector.")

      .def(
          "__add__", [](const Direction& d, const Vector& v) { return d + v; },
          "Returns the sum of a Direction and a Vector, as a Vector.")

      .def(
          "__radd__", [](const Direction& d, const Vector& v) { return v + d; },
          "Returns the sum of a Direction and a Vector, as a Vector.")

      .def(
          "__sub__",
          [](const Direction& d1, const Direction& d2) { return d1 - d2; },
          "Returns the difference of two Directions, as a Vector.")

      .def(
          "__sub__", [](const Direction& d, const Vector& v) { return d - v; },
          "Returns the difference of a Direction and Vector, as a Vector.")

      .def(
          "__rsub__", [](const Direction& d, const Vector& v) { return v - d; },
          "Returns the difference of a Vector and a Direction, as a Vector.")

      .def(
          "__mul__",
          [](const Direction& d1, const Direction& d2) { return d1 * d2; },
          "Returns the dot product of two Directions.")

      .def(
          "__mul__", [](const Direction& d, const Vector& v) { return d * v; },
          "Returns the dot product of a Direction and a Vector.")

      .def(
          "__rmul__", [](const Direction& d, const Vector& v) { return v * d; },
          "Returns the dot product of a Direction and a Vector.")

      .def(
          "__mul__", [](const Direction& u, double d) { return u * d; },
          "Scales a Direction by a constant, returning a Vector.")

      .def(
          "__rmul__", [](const Direction& u, double d) { return d * u; },
          "Scales a Direction by a constant, returning a Vector.")

      .def(
          "__truediv__", [](const Direction& u, double d) { return u / d; },
          "Scales a Direction by the inverse of a constant, returning a "
          "Vector.")

      .def(
          "__repr__",
          [](const Vector& d) {
            std::stringstream mssg;
            mssg << "<<" << d.x() << "," << d.y() << ">>";
            return mssg.str();
          },
          "String representation of a Direction.")

      .def(py::pickle(
          [](const Direction& d) { return py::make_tuple(d.x(), d.y()); },
          [](py::tuple t) {
            return Direction(t[0].cast<double>(), t[1].cast<double>());
          }));
}
