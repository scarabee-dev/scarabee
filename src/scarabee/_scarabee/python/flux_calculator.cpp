#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <xtensor-python/pytensor.hpp>

#include <data/flux_calculator.hpp>

#include <cereal/types/memory.hpp>
#include <cereal/archives/portable_binary.hpp>

#include <sstream>

namespace py = pybind11;

using namespace scarabee;

void init_FluxCalculator(py::module& m) {
  py::class_<FluxCalculator, std::shared_ptr<FluxCalculator>>(
      m, "FluxCalculator",
      "A FluxCalculator solves the neutron slowing down equation for a single "
      "resonant isotope plus any number of background isotopes which have "
      "constant "
      "cross sections and are pure scatterers.")

      .def(py::init<const xt::xtensor<double, 1>& /*energy_boundaries*/,
                    const xt::xtensor<double, 1>& /*sig_t_r*/,
                    const xt::xtensor<double, 1>& /*sig_s_r*/,
                    double /*awr_r*/>(),
           "Creates a FluxCalculator instance for a resonant isotope.\n\n"
           "Parameters\n"
           "----------\n"
           "energy_boundaries : ndarray\n"
           "    Boundaries of the energy groups from low to high energy.\n"
           "sig_t : ndarray\n"
           "    Microscopic total cross section of resonant isotope in each "
           "group.\n"
           "sig_s : ndarray\n"
           "    Microscopic scattering cross section of resonant isotope in "
           "each group.\n"
           "awr : float\n"
           "    Atomic weight ratio of thes resonant isotope.\n",
           py::arg("energy_boundaries"), py::arg("sig_t"), py::arg("sig_s"),
           py::arg("awr"))

      .def("add_background_nuclide", &FluxCalculator::add_background_nuclide,
           "Adds a pure scattering background nuclide with constant cross "
           "section "
           "to the slowing down problem.\n\n"
           "Parameters\n"
           "----------\n"
           "sig_s : float\n"
           "    Constant background scattering cross section for nuclide.\n"
           "awr : float\n"
           "    Atomic weight ratio of the nuclide.\n")

      .def("solve", &FluxCalculator::solve,
           "Solves the slowing down problem.\n")

      .def_property_readonly("energy_boundaries",
                             &FluxCalculator::energy_boundaries,
                             "Boundaries of the energy groups from low to high "
                             "energy in an ndarray.")

      .def_property_readonly(
          "avg_energy", &FluxCalculator::avg_energy,
          "Average energy in each group from low to high energy in an ndarray.")

      .def_property_readonly(
          "sig_t", &FluxCalculator::sig_t,
          "Total cross section of the resonant nuclide in each group from low "
          "to high energy in an ndarray.")

      .def_property_readonly(
          "sig_s", &FluxCalculator::sig_s,
          "Scattering cross section of the resonant nuclide in each group from "
          "low to high energy in an ndarray.")

      .def_property_readonly("chi", &FluxCalculator::chi,
                             "Fixed fission source in each group from low to "
                             "high energy in an ndarray.")

      .def_property_readonly(
          "flux", &FluxCalculator::flux,
          "Computed flux in each group from low to high energy in an ndarray.")

      .def_property_readonly("awr", &FluxCalculator::awr,
                             "Atomic weight ratio of the resonant nuclide.")

      .def_property_readonly("alpha", &FluxCalculator::awr,
                             "Slowing down parameter of the resonant nuclide "
                             "((awr - 1)/(awr + 1))^2.")

      .def(py::pickle(
          [](const std::shared_ptr<FluxCalculator>& p) {
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
            std::shared_ptr<FluxCalculator> p;
            {
              cereal::PortableBinaryInputArchive ar(bits_stream);
              ar(p);
            }
            return p;
          }));
}
