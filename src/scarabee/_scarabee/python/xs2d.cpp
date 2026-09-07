#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pytensor.hpp>

#include <data/xs2d.hpp>

#include <cereal/types/memory.hpp>
#include <cereal/archives/portable_binary.hpp>

#include <sstream>
#include <iomanip>

namespace py = pybind11;

using namespace scarabee;

void init_XS2D(py::module& m) {
  py::class_<XS2D, std::shared_ptr<XS2D>>(
      m, "XS2D",
      "A XS2D object holds a 2-dimensionl cross section (scattering matrix) "
      "in a special type of spare compressed format. It also is generalized "
      "to hold all legendre moments for a scattering matrix, effectively "
      "making it 3-dimensional. The data compression is only performed "
      "according to the P0 moment.")

      .def(py::init<const xt::xtensor<double, 2>& /*xs*/,
                    const xt::xtensor<std::uint32_t, 2>& /*packing*/>(),
           "Creates an XS2D from the compressed data array and packing "
           "information array. The packing array has two dimensions. The "
           "first must have a length of the number of energy groups. The "
           "second must have a length of 3. For each incident group, the "
           "starting index in the data array, lowest outgoing group index, "
           "and highest outgoing group index are stored.\n\n"
           "Parameters\n"
           "----------\n"
           "xs : ndarray\n"
           "     2-dimensional numpy array with compressed scatter matrix "
           "data. First index is for the legendre moment.\n"
           "packing : ndarray\n"
           "     2-dimensional numpy array of integers with compression "
           "description.\n\n",
           py::arg("xs"), py::arg("packing"))

      .def(py::init<const xt::xtensor<double, 2>& /*Es*/>(),
           "Creates an XS2D from a P0 scattering matrix.\n\n"
           "Parameters\n"
           "----------\n"
           "Es : ndarray\n"
           "     2-dimensional square array with P0 scattering matrix.\n\n",
           py::arg("Es"))

      .def(py::init<const xt::xtensor<double, 3>& /*Es*/>(),
           "Creates an XS2D from a scattering matrix with multiple legendre "
           "moments.\n\n"
           "Parameters\n"
           "----------\n"
           "Es : ndarray\n"
           "     3-dimensional array, where the first index is the legendre "
           "moment, and the last two dimensions are equal.\n\n",
           py::arg("Es"))

      .def_property_readonly("ngroups", &XS2D::ngroups,
                             "Number of energy groups.")

      .def_property_readonly("anisotropic", &XS2D::anisotropic,
                             "True if there is a P1 scattering matrix.")

      .def_property_readonly(
          "max_legendre_order", &XS2D::max_legendre_order,
          "Order of the largest stored legendre moment scattering matrix.")

      .def("set_value", &XS2D::set_value,
           "Sets the value of the cross section for legendre moment l and "
           "energy transition gin -> gout.\n\n"
           "Parameters\n"
           "----------\n"
           "l : int\n"
           "    Legendre moment.\n"
           "gin : int\n"
           "    Incident energy group.\n"
           "gout : int\n"
           "    Outgoing energy group.\n"
           "v : float\n"
           "    Value of cross section.",
           py::arg("l"), py::arg("gin"), py::arg("gout"), py::arg("v"))

      .def("__call__",
           py::overload_cast<const std::size_t, const std::size_t>(
               &XS2D::operator(), py::const_),
           "Evaluates the total scattering cross section for legendre "
           "moment l and incident group g.\n\n"
           "Parameters\n"
           "----------\n"
           "l : int\n"
           "    Legendre moment.\n"
           "g : int\n"
           "    Incident energy group.\n",
           py::arg("l"), py::arg("g"))

      .def("__call__",
           py::overload_cast<const std::size_t, const std::size_t,
                             const std::size_t>(&XS2D::operator(), py::const_),
           "Evaluates the scattering cross section for legendre moment l "
           "and energy transfer gin -> gout.\n\n"
           "Parameters\n"
           "----------\n"
           "l : int\n"
           "    Legendre moment.\n"
           "gin : int\n"
           "    Incident energy group.\n"
           "gout : int\n"
           "    Outgoing energy group.\n",
           py::arg("l"), py::arg("gin"), py::arg("gout"))

      .def("__add__", &XS2D::operator+)
      .def("__iadd__", &XS2D::operator+=)
      .def("__sub__", &XS2D::operator-)
      .def("__isub__", &XS2D::operator-=)
      .def("__imul__", &XS2D::operator*=)
      .def("__mul__", &XS2D::operator*)
      .def("__rmul__", [](const XS2D& xs, double N) { return xs * N; })
      .def("__itruediv__", &XS2D::operator/=)
      .def("__truediv__", &XS2D::operator/)

      .def("__repr__",
           [](const XS2D& xs) {
             std::stringstream out;
             out << std::scientific;
             out << '[';
             for (std::size_t l = 0; l < xs.max_legendre_order() + 1; l++) {
               out << '[';
               for (std::size_t g = 0; g < xs.ngroups(); g++) {
                 out << '[';
                 for (std::size_t gg = 0; gg < xs.ngroups(); gg++) {
                   out << xs(l, g, gg);
                   if (gg < xs.ngroups() - 1) out << ", ";
                 }
                 out << ']';
                 if (g < xs.ngroups() - 1) out << ",\n  ";
               }
               out << ']';
               if (l != xs.max_legendre_order()) out << ",\n ";
             }
             out << ']';

             return out.str();
           })

      .def("__str__",
           [](const XS2D& xs) {
             std::stringstream out;
             out << std::scientific;
             out << '[';
             for (std::size_t l = 0; l < xs.max_legendre_order() + 1; l++) {
               out << '[';
               for (std::size_t g = 0; g < xs.ngroups(); g++) {
                 out << '[';
                 for (std::size_t gg = 0; gg < xs.ngroups(); gg++) {
                   out << xs(l, g, gg);
                   if (gg < xs.ngroups() - 1) out << ", ";
                 }
                 out << ']';
                 if (g < xs.ngroups() - 1) out << ",\n  ";
               }
               out << ']';
               if (l != xs.max_legendre_order()) out << ",\n ";
             }
             out << ']';

             return out.str();
           })

      .def(py::pickle(
          [](const std::shared_ptr<XS2D>& p) {
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
            std::shared_ptr<XS2D> p;
            {
              cereal::PortableBinaryInputArchive ar(bits_stream);
              ar(p);
            }
            return p;
          }));
}
