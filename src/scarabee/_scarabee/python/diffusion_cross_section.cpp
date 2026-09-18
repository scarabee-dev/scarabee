#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pytensor.hpp>

#include <data/diffusion_cross_section.hpp>

#include <cereal/types/memory.hpp>
#include <cereal/archives/portable_binary.hpp>

#include <sstream>

namespace py = pybind11;

using namespace scarabee;

struct DiffusionCrossSectionPickler {
  static std::shared_ptr<DiffusionCrossSection> from_state(py::tuple t) {
    py::bytes bytes = t[0].cast<py::tuple>();
    std::istringstream bits_stream(bytes,
                                   std::ios_base::binary | std::ios_base::in);
    std::shared_ptr<DiffusionCrossSection> p;
    {
      cereal::PortableBinaryInputArchive ar(bits_stream);
      ar(p);
    }
    return p;
  }

  static py::tuple to_state(const std::shared_ptr<DiffusionCrossSection>& xs) {
    std::ostringstream bits_stream(std::ios_base::binary | std::ios_base::out);
    {
      cereal::PortableBinaryOutputArchive ar(bits_stream);
      ar(xs);
    }
    return py::make_tuple(py::bytes(bits_stream.str()));
  }
};

void init_DiffusionCrossSection(py::module& m) {
  py::class_<DiffusionCrossSection, std::shared_ptr<DiffusionCrossSection>>(
      m, "DiffusionCrossSection",
      "A DiffusionCrossSection object contains all necessary cross section "
      "information to perform a diffusion calculation.")

      .def(py::init<const xt::xtensor<double, 1>& /*D*/,
                    const xt::xtensor<double, 1>& /*Ea*/,
                    const xt::xtensor<double, 2>& /*Es*/,
                    const xt::xtensor<double, 1>& /*Ef*/,
                    const xt::xtensor<double, 1>& /*vEf*/,
                    const xt::xtensor<double, 1>& /*chi*/,
                    const std::string& /*name*/>(),
           "Creates a DiffusionCrossSection from transport corrected data.\n\n"
           "Parameters\n"
           "----------\n"
           "D : ndarray\n"
           "      Diffusion coefficients.\n"
           "Ea : ndarray\n"
           "     Absorption cross section.\n"
           "Es : ndarray\n"
           "     Transport corrected scattering cross section matrix.\n"
           "Ef : ndarray\n"
           "     Fission cross section.\n"
           "vEf : ndarray\n"
           "      Fission yield time cross section.\n"
           "chi : ndarray\n"
           "      Fission spectrum.\n"
           "name : str (optional)\n"
           "       Name of material.\n\n",
           py::arg("D"), py::arg("Ea"), py::arg("Es"), py::arg("Ef"),
           py::arg("vEf"), py::arg("chi"), py::arg("name") = "")

      .def(py::init<const xt::xtensor<double, 1>& /*D*/,
                    const xt::xtensor<double, 1>& /*Ea*/,
                    const xt::xtensor<double, 2>& /*Es*/,
                    const std::string& /*name*/>(),
           "Creates a DiffusionCrossSection from transport corrected data. "
           "Fission cross sections and spectrum initialized to zero.\n\n"
           "Parameters\n"
           "----------\n"
           "D : ndarray\n"
           "    Diffusion coefficients.\n"
           "Ea : ndarray\n"
           "     Absorption cross section.\n"
           "Es : ndarray\n"
           "     Transport corrected scattering cross section matrix.\n"
           "name : str (optional)\n"
           "       Name of material.\n\n",
           py::arg("D"), py::arg("Ea"), py::arg("Es"), py::arg("name") = "")

      .def_property_readonly("ngroups", &DiffusionCrossSection::ngroups,
                             "Number of energy groups.")

      .def_property("name", &DiffusionCrossSection::name,
                    &DiffusionCrossSection::set_name, "Name of material.")

      .def_property_readonly("fissile", &DiffusionCrossSection::fissile,
                             "True if material is fissile.")

      .def("D", &DiffusionCrossSection::D,
           "Diffusion coefficient in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("Ea", &DiffusionCrossSection::Ea,
           "Absorption cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("Ef", &DiffusionCrossSection::Ef,
           "Fission cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("vEf", &DiffusionCrossSection::vEf,
           "Fission yield * cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("nu", &DiffusionCrossSection::nu,
           "Fission yield in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("Er", &DiffusionCrossSection::Er,
           "Removal cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("chi", &DiffusionCrossSection::chi,
           "Fission spectrum in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("Es",
           py::overload_cast<std::size_t>(&DiffusionCrossSection::Es,
                                          py::const_),
           "Transport corrected scattering cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.\n\n",
           py::arg("g"))

      .def("Es",
           py::overload_cast<std::size_t, std::size_t>(
               &DiffusionCrossSection::Es, py::const_),
           "Transport corrected scattering cross section from group gin to "
           "gout\n\n"
           "Parameters\n"
           "----------\n"
           "gin : int\n"
           "      Incoming energy group.\n"
           "gout : int\n"
           "       Outgoing energy group.\n\n",
           py::arg("gin"), py::arg("gout"))

      .def("condense", &DiffusionCrossSection::condense,
           "Condenses the cross sections to a new energy group structure. The "
           "condensation group structure is provided as a list of pairs "
           "(2D tuples), indicating the lower and upper group indices "
           "(inclusive) of "
           "a macro energy group. \n\n"
           "Parameters\n"
           "----------\n"
           "groups : list of 2D tuples of ints.\n"
           "         The scheme for condensing energy groups.\n"
           "flux : ndarray of floats.\n"
           "       The weighting flux spectrum of the original group "
           "structure.\n\n"
           "Returns\n"
           "-------\n"
           "DiffusionsCrossSection\n"
           "                      Condensed set of diffusion cross sections.\n",
           py::arg("groups"), py::arg("flux"))

      .def(py::pickle(&DiffusionCrossSectionPickler::to_state,
                      &DiffusionCrossSectionPickler::from_state));
}
