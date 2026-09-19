#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <xtensor-python/pytensor.hpp>

#include <cereal/archives/portable_binary.hpp>

#include <data/micro_cross_sections.hpp>

#include <sstream>

namespace py = pybind11;

using namespace scarabee;

struct MicroDepletionXSPickler {
  static MicroDepletionXS from_state(py::tuple t) {
    MicroDepletionXS out;
    py::bytes bytes = t[0].cast<py::bytes>();
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(out);
    }
    return out;
  }

  static py::tuple to_state(const MicroDepletionXS xs) {
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(xs);
    }
    py::bytes bytes(bits_stream.str());
    return py::make_tuple(bytes);
  }
};

struct MicroNuclideXSPickler {
  static MicroNuclideXS from_state(py::tuple t) {
    MicroNuclideXS out;
    py::bytes bytes = t[0].cast<py::bytes>();
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(out);
    }
    return out;
  }

  static py::tuple to_state(const MicroNuclideXS xs) {
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(xs);
    }
    py::bytes bytes(bits_stream.str());
    return py::make_tuple(bytes);
  }
};

struct ResonantOneGroupXSPickler {
  static ResonantOneGroupXS from_state(py::tuple t) {
    ResonantOneGroupXS out;
    py::bytes bytes = t[0].cast<py::bytes>();
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(out);
    }
    return out;
  }

  static py::tuple to_state(const ResonantOneGroupXS& xs) {
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(xs);
    }
    py::bytes bytes(bits_stream.str());
    return py::make_tuple(bytes);
  }
};

struct DepletionReactionRatesPickler {
  static DepletionReactionRates from_state(py::tuple t) {
    DepletionReactionRates out;
    py::bytes bytes = t[0].cast<py::bytes>();
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(out);
    }
    return out;
  }

  static py::tuple to_state(const DepletionReactionRates& rr) {
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(rr);
    }
    py::bytes bytes(bits_stream.str());
    return py::make_tuple(bytes);
  }
};

void init_MicroCrossSectionStructs(py::module& m) {
  py::class_<MicroDepletionXS>(
      m, "MicroDepletionXS",
      "Holds the microscopic cross sections needed for depletion calculations.")
      .def_readonly("n_fission", &MicroDepletionXS::n_fission,
                    "Fission cross section.")
      .def_readonly("n_gamma", &MicroDepletionXS::n_gamma,
                    "Capture cross section.")
      .def_readonly("n_2n", &MicroDepletionXS::n_2n, "(n,2n) cross section.")
      .def_readonly("n_3n", &MicroDepletionXS::n_3n, "(n,3n) cross section.")
      .def_readonly("n_alpha", &MicroDepletionXS::n_alpha,
                    "(n,alpha) cross section.")
      .def_readonly("n_p", &MicroDepletionXS::n_p, "(n,p) cross section.")
      .def(py::pickle(&MicroDepletionXSPickler::to_state,
                      &MicroDepletionXSPickler::from_state));

  py::class_<MicroNuclideXS>(
      m, "MicroNuclideXS",
      "Holds the microscopic cross sections for a nuclide.")
      .def_readonly("Et", &MicroNuclideXS::Et, "Total cross section.")
      .def_readonly("Dtr", &MicroNuclideXS::Dtr, "Transport correction.")
      .def_readonly("Es", &MicroNuclideXS::Es, "Scattering cross section.")
      .def_readonly("Ea", &MicroNuclideXS::Ea, "Absorption cross section.")
      .def_readonly("Ef", &MicroNuclideXS::Ef, "Fission cross section.")
      .def_readonly("nu", &MicroNuclideXS::nu, "Fission yield.")
      .def_readonly("chi", &MicroNuclideXS::chi, "Fission spectrum.")
      .def(py::pickle(&MicroNuclideXSPickler::to_state,
                      &MicroNuclideXSPickler::from_state));

  py::class_<ResonantOneGroupXS>(m, "ResonantOneGroupXS",
                                 "Single group resonant cross sections.")
      .def_readonly("Dtr", &ResonantOneGroupXS::Dtr, "Transport correction.")
      .def_readonly("Ea", &ResonantOneGroupXS::Ea, "Absorption cross section.")
      .def_readonly("Ef", &ResonantOneGroupXS::Ef, "Fission cross section.")
      .def_readonly("Es", &ResonantOneGroupXS::Es, "Scattering cross section.")
      .def_readonly("gout_min", &ResonantOneGroupXS::gout_min,
                    "First outgoing energy group.")
      .def_readonly("n_gamma", &ResonantOneGroupXS::n_gamma,
                    "Capture cross section.")
      .def(py::pickle(&ResonantOneGroupXSPickler::to_state,
                      &ResonantOneGroupXSPickler::from_state));

  py::class_<DepletionReactionRates>(
      m, "DepletionReactionRates",
      "Holds the reaction rate data for a single nuclide, used in depletion "
      "calculations.")

      .def(py::init<>(), "Initializes an empty set of reaction rates.\n\n")

      .def_readwrite("nuclide", &DepletionReactionRates::nuclide,
                     "Name of the nuclide.")
      .def_readwrite(
          "number_density", &DepletionReactionRates::number_density,
          "Number of the specified nuclide in units of atoms per barn-cm.")
      .def_readwrite("n_gamma", &DepletionReactionRates::n_gamma,
                     "Energy integrated (n,gamma) microscopic reaction rate in "
                     "units of 1/s.")
      .def_readwrite(
          "n_2n", &DepletionReactionRates::n_2n,
          "Energy integrated (n,2n) microscopic reaction rate in units of 1/s.")
      .def_readwrite(
          "n_3n", &DepletionReactionRates::n_3n,
          "Energy integrated (n,3n) microscopic reaction rate in units of 1/s.")
      .def_readwrite(
          "n_p", &DepletionReactionRates::n_p,
          "Energy integrated (n,p) microscopic reaction rate in units of 1/s.")
      .def_readwrite("n_alpha", &DepletionReactionRates::n_alpha,
                     "Energy integrated (n,alpha) microscopic reaction rate in "
                     "units of 1/s.")
      .def_readwrite("n_fission", &DepletionReactionRates::n_fission,
                     "Energy integrated (n,fission) microscopic reaction rate "
                     "in units of 1/s.")
      .def_readwrite("average_fission_energy",
                     &DepletionReactionRates::average_fission_energy,
                     "Average energy, in eV, of a neutron inducing fission.")
      .def(py::pickle(&DepletionReactionRatesPickler::to_state,
                      &DepletionReactionRatesPickler::from_state));
}
