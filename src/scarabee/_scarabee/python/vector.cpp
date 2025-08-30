#include <pybind11/pybind11.h>

#include <moc/vector.hpp>

#include <sstream>

namespace py = pybind11;

using namespace scarabee;

void init_Vector(py::module& m) {
  py::class_<Vector>(
      m, "Vector",
      "A Vector object represents a two dimentional spatial vector of "
      "arbitrary magnitude. It supports standard vector arithmetic such "
      "as addtion, subtraction, and scaling. Vector objects are immutable, "
      "and can only be changed through assigning a new Vector.")

      .def(py::init<double, double>(),
           "Creates a Vector for an x and a y component.\n\n"
           "Parameters\n"
           "----------\n"
           "x : float\n"
           "    x component of vector.\n"
           "y : float\n"
           "    y component of vector.\n",
           py::arg("x"), py::arg("y"))

      .def("dot", &Vector::dot,
           "Computes dot product with another vector.\n\n"
           ".. math:: d = \\text{self}.x \\cdot v.x + \\text{self}.y \\cdot "
           "v.y\n\n"
           "Parameters\n"
           "----------\n"
           "v : Vector\n"
           "    Vector with which to take the dot product.\n\n"
           "Returns\n"
           "-------\n"
           "d : float\n"
           "    The result of  the dot product\n",
           py::arg("v"))

      .def_property_readonly("x", &Vector::x, "x component of vector.")

      .def_property_readonly("y", &Vector::y, "y component of vector.")

      .def_property_readonly("norm", &Vector::norm, "Length of the vector.")

      .def("__neg__", &Vector::operator-, "Returns the negation of the vector.")

      .def(
          "__eq__", &Vector::operator==,
          "Returns True if two vectors are equal (within a certain tolerance).")

      .def("__ne__", &Vector::operator!=,
           "Returns True if two vectors are not equal (within a "
           "certain tolerance).")

      .def(
          "__add__", [](const Vector& v1, const Vector& v2) { return v1 + v2; },
          "Computes the sum of two vectors.")

      .def(
          "__sub__", [](const Vector& v1, const Vector& v2) { return v1 - v2; },
          "Computes the difference of two vectors.")

      .def(
          "__mul__", [](const Vector& v1, const Vector& v2) { return v1 * v2; },
          "Computes the dot product of two vectors (an alias for v1.dot(v2)).")

      .def(
          "__mul__", [](const Vector& v, double d) { return v * d; },
          "Scales a vector by a constant.")

      .def(
          "__rmul__", [](const Vector& v, double d) { return d * v; },
          "Scales a vector by a constant.")

      .def(
          "__truediv__", [](const Vector& v, double d) { return v / d; },
          "Scales a vector by the inverse of a constant.")

      .def(
          "__repr__",
          [](const Vector& v) {
            std::stringstream mssg;
            mssg << "<" << v.x() << "," << v.y() << ">";
            return mssg.str();
          },
          "String representation of a Vector.");
}
