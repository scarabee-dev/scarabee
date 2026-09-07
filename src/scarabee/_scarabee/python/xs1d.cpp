#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pytensor.hpp>

#include <data/xs1d.hpp>

#include <cereal/types/memory.hpp>
#include <cereal/archives/portable_binary.hpp>

#include <sstream>
#include <iomanip>

namespace py = pybind11;

using namespace scarabee;

void init_XS1D(py::module& m) {
  py::class_<XS1D, std::shared_ptr<XS1D>>(
      m, "XS1D",
      "A XS1D objects holds a 1-dimensional cross section. It is generalized "
      "to hold threshold reactions, so that if you request the cross section "
      "for group g where g >= the number of groups, zero is returned.")

      .def(py::init<const xt::xtensor<double, 1>& /*xs*/>(),
           "Creates an XS1D from a 1D numpy array.\n\n"
           "Parameters\n"
           "----------\n"
           "xs : ndarray\n"
           "     Cross section values for each group.\n",
           py::arg("xs"))

      .def_property_readonly("ngroups", &XS1D::ngroups,
                             "Number of energy groups.")

      .def("__call__", &XS1D::operator(),
           "Returns the value of the cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("set_value", &XS1D::set_value,
           "Sets the value of the cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.\n"
           "v : float\n"
           "    Value of cross section.",
           py::arg("g"), py::arg("v"))

      .def("resize", &XS1D::resize,
           "Resizes the XS1D to a different number of stored energy groups. "
           "The original cross section values are preserved to the extent "
           "possible. If NG is greater than the number of groups, zeros are "
           "added for the additional groups. If NG is smaller than the number "
           "of groups, the first NG values are preserved.\n\n"
           "Parameters\n"
           "----------\n"
           "NG : int\n"
           "     New number of energy groups.")

      .def("__add__", &XS1D::operator+)
      .def("__iadd__", &XS1D::operator+=)
      .def("__sub__", &XS1D::operator-)
      .def("__isub__", &XS1D::operator-=)
      .def("__imul__", py::overload_cast<const double>(&XS1D::operator*=))
      .def("__imul__", py::overload_cast<const XS1D&>(&XS1D::operator*=))
      .def("__mul__",
           py::overload_cast<const XS1D&>(&XS1D::operator*, py::const_))
      .def("__mul__",
           py::overload_cast<const double>(&XS1D::operator*, py::const_))
      .def("__rmul__", [](const XS1D& xs, double N) { return xs * N; })
      .def("__itruediv__", &XS1D::operator/=)
      .def("__truediv__", &XS1D::operator/)

      .def("__str__",
           [](const XS1D& xs) {
             std::stringstream out;
             out << std::scientific;
             out << '[';
             for (std::size_t i = 0; i < xs.ngroups(); i++) {
               out << xs(i);
               if (i < xs.ngroups() - 1) out << ", ";
             }
             out << ']';
             return out.str();
           })

      .def("__repr__",
           [](const XS1D& xs) {
             std::stringstream out;
             out << std::scientific;
             out << '[';
             for (std::size_t i = 0; i < xs.ngroups(); i++) {
               out << xs(i);
               if (i < xs.ngroups() - 1) out << ", ";
             }
             out << ']';
             return out.str();
           })

      .def(py::pickle(
          [](const std::shared_ptr<XS1D>& p) {
            std::ostringstream bits_stream(std::ios_base::binary |
                                           std::ios_base::out);
            {
              cereal::PortableBinaryOutputArchive ar(bits_stream);
              ar(p);
            }
            return py::bytes(bits_stream.str());
          },
          [](py::bytes bites) {
            std::istringstream bits_stream(
                bites, std::ios_base::binary | std::ios_base::in);
            std::shared_ptr<XS1D> p;
            {
              cereal::PortableBinaryInputArchive ar(bits_stream);
              ar(p);
            }
            return p;
          }));
}
