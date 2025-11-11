#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <moc/quadrature/legendre.hpp>
#include <moc/quadrature/yamamoto_tabuchi.hpp>
#include <moc/quadrature/polar_quadrature.hpp>

#include <vector>

namespace py = pybind11;

using namespace scarabee;

template <typename PQ>
void init_quad(py::module& m, const std::string& name,
               const std::string& description) {
  py::class_<PQ>(m, name.c_str())
      .def(py::init<>(), description.c_str())
      .def(
          "sin",
          [](const PQ& q) {
            return std::vector<double>(q.sin().begin(), q.sin().end());
          },
          "Array of sine values.")
      .def(
          "invs_sin",
          [](const PQ& q) {
            return std::vector<double>(q.invs_sin().begin(),
                                       q.invs_sin().end());
          },
          "Array of inverse sine values.")
      .def(
          "wgt",
          [](const PQ& q) {
            return std::vector<double>(q.wgt().begin(), q.wgt().end());
          },
          "Array of weight values.");
}

void init_PolarQuadrature(py::module& m) {
  // Legendre quadratures
  init_quad<Legendre<2>>(m, "Legendre2",
                         "Two point Legendre polar quadrature.");
  init_quad<Legendre<4>>(m, "Legendre4",
                         "Four point Legendre polar quadrature.");
  init_quad<Legendre<6>>(m, "Legendre6",
                         "Six point Legendre polar quadrature.");
  init_quad<Legendre<8>>(m, "Legendre8",
                         "Eight point Legendre polar quadrature.");
  init_quad<Legendre<10>>(m, "Legendre10",
                          "Ten point Legendre polar quadrature.");
  init_quad<Legendre<12>>(m, "Legendre12",
                          "Twelve point Legendre polar quadrature.");
  init_quad<Legendre<16>>(m, "Legendre16",
                          "Sixteen point Legendre polar quadrature.");
  init_quad<Legendre<32>>(m, "Legendre32",
                          "Thirty-two point Legendre polar quadrature.");
  init_quad<Legendre<64>>(m, "Legendre64",
                          "Sixty-four point Legendre polar quadrature.");

  // Yamamoto-Tabuchi quadratures
  init_quad<YamamotoTabuchi<2>>(m, "YamamotoTabuchi2",
                                "Two point Yamamoto-Tabuchi polar quadrature.");
  init_quad<YamamotoTabuchi<4>>(
      m, "YamamotoTabuchi4", "Four point Yamamoto-Tabuchi polar quadrature.");
  init_quad<YamamotoTabuchi<6>>(m, "YamamotoTabuchi6",
                                "Six point Yamamoto-Tabuchi polar quadrature.");

  // PolarQuadrature
  py::class_<PolarQuadrature>(m, "PolarQuadrature")
      .def(py::init<PolarQuadratureType>(), "General polar quadrature.")
      .def(
          "sin",
          [](const PolarQuadrature& q) {
            return std::vector<double>(q.sin().begin(), q.sin().end());
          },
          "Array of sine values.")
      .def(
          "invs_sin",
          [](const PolarQuadrature& q) {
            return std::vector<double>(q.invs_sin().begin(),
                                       q.invs_sin().end());
          },
          "Array of inverse sine values.")
      .def(
          "wgt",
          [](const PolarQuadrature& q) {
            return std::vector<double>(q.wgt().begin(), q.wgt().end());
          },
          "Array of weight values.");
}
