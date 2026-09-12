#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pytensor.hpp>

#include <data/cross_section.hpp>

#include <cereal/types/memory.hpp>
#include <cereal/archives/portable_binary.hpp>

#include <sstream>

namespace py = pybind11;

using namespace scarabee;

void init_CrossSection(py::module& m) {
  py::class_<CrossSection, std::shared_ptr<CrossSection>>(
      m, "CrossSection",
      "A CrossSection object contains all necessary cross section information "
      "to perform a transport calculation. A reduced set of arithmetic "
      "operations can be performed on CrossSections to facilitate the "
      "construction of macroscopic material cross sections. This includes "
      "scalar multiplication, and the addition of CrossSection objects.")

      .def(py::init<const xt::xtensor<double, 1>& /*Etr*/,
                    const xt::xtensor<double, 1>& /*Ea*/,
                    const xt::xtensor<double, 2>& /*Es_tr*/,
                    const xt::xtensor<double, 1>& /*Ef*/,
                    const xt::xtensor<double, 1>& /*vEf*/,
                    const xt::xtensor<double, 1>& /*chi*/,
                    const std::string& /*name*/>(),
           "Creates a CrossSection from transport corrected data.\n\n"
           "Parameters\n"
           "----------\n"
           "Etr : ndarray\n"
           "      Transport corrected total cross section.\n"
           "Ea : ndarray\n"
           "     Absorption cross section.\n"
           "Es_tr : ndarray\n"
           "        Transport corrected scattering cross section matrix.\n"
           "Ef : ndarray\n"
           "     Fission cross section.\n"
           "vEf : ndarray\n"
           "      Fission yield time cross section.\n"
           "chi : ndarray\n"
           "      Fission spectrum.\n"
           "name : str (optional)\n"
           "       Name of material.\n\n",
           py::arg("Etr"), py::arg("Ea"), py::arg("Es_tr"), py::arg("Ef"),
           py::arg("vEf"), py::arg("chi"), py::arg("name") = "")

      .def(py::init<const XS1D& /*Etr*/, const XS1D& /*Ea*/,
                    const XS2D& /*Es_tr*/, const XS1D& /*Ef*/,
                    const XS1D& /*vEf*/, const XS1D& /*chi*/,
                    const std::string& /*name*/>(),
           "Creates a CrossSection from transport corrected data.\n\n"
           "Parameters\n"
           "----------\n"
           "Etr : XS1D\n"
           "      Transport corrected total cross section.\n"
           "Ea : XS1D\n"
           "     Absorption cross section.\n"
           "Es_tr : XS2D\n"
           "        Transport corrected scattering cross section matrix.\n"
           "Ef : XS1D\n"
           "     Fission cross section.\n"
           "vEf : XS1D\n"
           "      Fission yield time cross section.\n"
           "chi : XS1D\n"
           "      Fission spectrum.\n"
           "name : str (optional)\n"
           "       Name of material.\n\n",
           py::arg("Etr"), py::arg("Ea"), py::arg("Es_tr"), py::arg("Ef"),
           py::arg("vEf"), py::arg("chi"), py::arg("name") = "")

      .def(py::init<const xt::xtensor<double, 1>& /*Et*/,
                    const xt::xtensor<double, 1>& /*Dtr*/,
                    const xt::xtensor<double, 1>& /*Ea*/,
                    const xt::xtensor<double, 3>& /*Es*/,
                    const xt::xtensor<double, 1>& /*Ef*/,
                    const xt::xtensor<double, 1>& /*vEf*/,
                    const xt::xtensor<double, 1>& /*chi*/,
                    const std::string& /*name*/>(),
           "Creates a CrossSection object from uncorrected cross section "
           "data.\n\n"
           "Parameters\n"
           "----------\n"
           "Et : ndarray\n"
           "     Total cross section\n"
           "Dtr : ndarray\n"
           "     Transport correction\n"
           "Ea : ndarray\n"
           "     Absorption cross section\n"
           "Es : ndarray\n"
           "     Scattering matrices for all legendre moments\n"
           "Ef : ndarray\n"
           "     Fission cross section\n"
           "vEf : ndarray\n"
           "      Fission yield time cross section\n"
           "chi : ndarray\n"
           "      Fission spectrum\n"
           "name : str (optional)\n"
           "       Name of material.\n\n",
           py::arg("Et"), py::arg("Dtr"), py::arg("Ea"), py::arg("Es"),
           py::arg("Ef"), py::arg("vEf"), py::arg("chi"), py::arg("name") = "")

      .def(py::init<const XS1D& /*Et*/, const XS1D& /*Dtr*/, const XS1D& /*Ea*/,
                    const XS2D& /*Es*/, const XS1D& /*Ef*/, const XS1D& /*vEf*/,
                    const XS1D& /*chi*/, const std::string& /*name*/>(),
           "Creates a CrossSection object from uncorrected cross section "
           "data.\n\n"
           "Parameters\n"
           "----------\n"
           "Et : XS1D\n"
           "     Total cross section\n"
           "Dtr : XS1D\n"
           "     Transport correction\n"
           "Ea : XS1D\n"
           "     Absorption cross section\n"
           "Es : XS2D\n"
           "     Scattering matrices for all legendre moments\n"
           "Ef : XS1D\n"
           "     Fission cross section\n"
           "vEf : XS1D\n"
           "      Fission yield time cross section\n"
           "chi : XS1D\n"
           "      Fission spectrum\n"
           "name : str (optional)\n"
           "       Name of material.\n\n",
           py::arg("Et"), py::arg("Dtr"), py::arg("Ea"), py::arg("Es"),
           py::arg("Ef"), py::arg("vEf"), py::arg("chi"), py::arg("name") = "")

      .def(py::init<const xt::xtensor<double, 1>& /*Etr*/,
                    const xt::xtensor<double, 1>& /*Ea*/,
                    const xt::xtensor<double, 2>& /*Es_tr*/,
                    const std::string& /*name*/>(),
           "Creates a CrossSection from transport corrected data. "
           "Fission cross sections and spectrum initialized to zero.\n\n"
           "Parameters\n"
           "----------\n"
           "Etr : ndarray\n"
           "      Transport corrected total cross section.\n"
           "Ea : ndarray\n"
           "     Absorption cross section.\n"
           "Es_tr : ndarray\n"
           "        Transport corrected scattering cross section matrix.\n"
           "name : str (optional)\n"
           "       Name of material.\n\n",
           py::arg("Etr"), py::arg("Ea"), py::arg("Es_tr"),
           py::arg("name") = "")

      .def(py::init<const XS1D& /*Etr*/, const XS1D& /*Ea*/,
                    const XS2D& /*Es_tr*/, const std::string& /*name*/>(),
           "Creates a CrossSection from transport corrected data. "
           "Fission cross sections and spectrum initialized to zero.\n\n"
           "Parameters\n"
           "----------\n"
           "Etr : XS1D\n"
           "      Transport corrected total cross section.\n"
           "Ea : XS1D\n"
           "     Absorption cross section.\n"
           "Es_tr : XS2D\n"
           "        Transport corrected scattering cross section matrix.\n"
           "name : str (optional)\n"
           "       Name of material.\n\n",
           py::arg("Etr"), py::arg("Ea"), py::arg("Es_tr"),
           py::arg("name") = "")

      .def(py::init<const xt::xtensor<double, 1>& /*Et*/,
                    const xt::xtensor<double, 1>& /*Dtr*/,
                    const xt::xtensor<double, 1>& /*Ea*/,
                    const xt::xtensor<double, 3>& /*Es*/,
                    const std::string& /*name*/>(),
           "Creates a CrossSection object from uncorrected cross section data. "
           "Fission cross sections and spectrum initialized to zero.\n\n"
           "Parameters\n"
           "----------\n"
           "Et : ndarray\n"
           "     Total cross section.\n"
           "Dtr : ndarray\n"
           "     Transport correction\n"
           "Ea : ndarray\n"
           "     Absorption cross section.\n"
           "Es : ndarray\n"
           "     Scattering matrices for all legendre moments\n"
           "name : str (optional)\n"
           "       Name of material.\n\n",
           py::arg("Et"), py::arg("Dtr"), py::arg("Ea"), py::arg("Es"),
           py::arg("name") = "")

      .def(py::init<const XS1D& /*Et*/, const XS1D& /*Dtr*/, const XS1D& /*Ea*/,
                    const XS2D& /*Es*/, const std::string& /*name*/>(),
           "Creates a CrossSection object from uncorrected cross section data. "
           "Fission cross sections and spectrum initialized to zero.\n\n"
           "Parameters\n"
           "----------\n"
           "Et : XS1D\n"
           "     Total cross section.\n"
           "Dtr : XS1D\n"
           "     Transport correction\n"
           "Ea : XS1D\n"
           "     Absorption cross section.\n"
           "Es : XS2D\n"
           "     Scattering matrices for all legendre moments\n"
           "name : str (optional)\n"
           "       Name of material.\n\n",
           py::arg("Et"), py::arg("Dtr"), py::arg("Ea"), py::arg("Es"),
           py::arg("name") = "")

      .def_property_readonly("ngroups", &CrossSection::ngroups,
                             "Number of energy groups.")

      .def_property("name", &CrossSection::name, &CrossSection::set_name,
                    "Name of material.")

      .def_property_readonly("fissile", &CrossSection::fissile,
                             "True if material is fissile.")

      .def_property_readonly("anisotropic", &CrossSection::anisotropic,
                             "True if the material has a P1 scattering matrix.")

      .def_property_readonly("max_legendre_order",
                             &CrossSection::max_legendre_order,
                             "Maximum legendre order for scattering.")

      .def("set", &CrossSection::set,
           "Reinitializes to be copy of another cross section.\n\n"
           "Parameters\n"
           "----------\n"
           "other : CrossSection\n"
           "    Cross section used to reinitialize the current data.\n")

      .def("Etr",
           py::overload_cast<std::size_t>(&CrossSection::Etr, py::const_),
           "Transport corrected total cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("Dtr", &CrossSection::Dtr,
           "Transport correction in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("Et", &CrossSection::Et,
           "Total cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("Ea", &CrossSection::Ea,
           "Absorption cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("Ef", &CrossSection::Ef,
           "Fission cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("vEf", &CrossSection::vEf,
           "Fission yield * cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("nu", &CrossSection::nu,
           "Fission yield in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("Er", &CrossSection::Er,
           "Removal cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("chi", &CrossSection::chi,
           "Fission spectrum in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("Es_tr",
           py::overload_cast<std::size_t>(&CrossSection::Es_tr, py::const_),
           "Transport corrected scattering cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.\n\n",
           py::arg("g"))

      .def("Es_tr",
           py::overload_cast<std::size_t, std::size_t>(&CrossSection::Es_tr,
                                                       py::const_),
           "Transport corrected scattering cross section from group gin to "
           "gout\n\n"
           "Parameters\n"
           "----------\n"
           "gin : int\n"
           "      Incoming energy group.\n"
           "gout : int\n"
           "       Outgoing energy group.\n\n",
           py::arg("gin"), py::arg("gout"))

      .def("Es",
           py::overload_cast<std::size_t, std::size_t>(&CrossSection::Es,
                                                       py::const_),
           "Pl scattering cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "l : int\n"
           "    Scattering moment.\n"
           "g : int\n"
           "    Energy group.\n\n",
           py::arg("l"), py::arg("g"))

      .def("Es",
           py::overload_cast<std::size_t, std::size_t, std::size_t>(
               &CrossSection::Es, py::const_),
           "Pl scattering cross section from group gin to gout.\n\n"
           "Parameters\n"
           "----------\n"
           "l   : int\n"
           "      Scattering moment.\n"
           "gin : int\n"
           "      Incoming energy group.\n"
           "gout : int\n"
           "       Outgoing energy group.\n\n",
           py::arg("l"), py::arg("gin"), py::arg("gout"))

      .def("condense", &CrossSection::condense,
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
           "CrossSection\n"
           "            Condensed set of cross sections.\n",
           py::arg("groups"), py::arg("flux"))

      .def("diffusion_xs", &CrossSection::diffusion_xs,
           "Creates a :py:class:`DiffusionCrossSection` from the cross "
           "section.\n\n"
           "Returns\n"
           "-------\n"
           "DiffusionCrossSection\n"
           "    Diffusion cross sections.\n")

      .def("__mul__", &CrossSection::operator*)
      .def("__rmul__", [](const CrossSection& xs, double N) { return xs * N; })
      .def("__add__", &CrossSection::operator+)
      .def("__imul__", &CrossSection::operator*=)
      .def("__iadd__", &CrossSection::operator+=)

      .def(py::pickle(
          [](const std::shared_ptr<CrossSection>& p) {
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
            std::shared_ptr<CrossSection> p;
            {
              cereal::PortableBinaryInputArchive ar(bits_stream);
              ar(p);
            }
            return p;
          }));
}
