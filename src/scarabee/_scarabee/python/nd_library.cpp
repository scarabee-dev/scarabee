#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <data/nd_library.hpp>
#include <utils/constants.hpp>

#include <memory>

namespace py = pybind11;

using namespace scarabee;

void init_NuclideHandle(py::module& m) {
  py::class_<NuclideHandle>(
      m, "NuclideHandle",
      "A NuclideHandle contains information about a specific nuclide, obtained "
      "from the nuclear data library (see :py:class:`NDLibrary` ).")

      .def_readonly("name", &NuclideHandle::name, "Indentifier of the nuclide.")

      .def_readonly("label", &NuclideHandle::label,
                    "Optional label provided at library creation.")

      .def_readonly(
          "temperatures", &NuclideHandle::temperatures,
          "List of temperatures at which cross sections are tabulated.")

      .def_readonly("dilutions", &NuclideHandle::dilutions,
                    "List of dilutions at which cross sections are tabulated.")

      .def_readonly("awr", &NuclideHandle::awr,
                    "Atomic weight ratio of the nuclide.")

      .def_readonly("potential_xs", &NuclideHandle::potential_xs,
                    "Potential scattering cross section of the nuclide.")

      .def_readonly(
          "ir_lambda", &NuclideHandle::ir_lambda,
          "List of intermediate resonance parameters for the nuclide.")

      .def_readonly("ZA", &NuclideHandle::ZA,
                    "The ZA number of the nuclide, constructed as Z*1000 + A.")

      .def_readonly("fissile", &NuclideHandle::fissile,
                    "True if the nuclide is fissile, False otherwise.")

      .def_readonly(
          "fission_energy", &NuclideHandle::fission_energy,
          "Average energy release per fission of the nuclide, in MeV.")

      .def_readonly("resonant", &NuclideHandle::resonant,
                    "True if the nuclide is resonant, False otherwise.");
}

void init_NDLibrary(py::module& m) {
  py::class_<NDLibrary, std::shared_ptr<NDLibrary>>(m, "NDLibrary")
      .def(py::init<>(),
           "Creates a new NDLibrary object from the HDF5 file pointed to by "
           "the environemnt variable " NDL_ENV_VAR ".\n\n")

      .def(py::init<const std::string&>(),
           "Creates a new NDLibrary object from an HDF5 file.\n\n"
           "Parameters\n"
           "----------\n"
           "fname : str\n"
           "        Name of the hdf5 file with the library.\n\n",
           py::arg("fname"))

      .def("get_nuclide",
           py::overload_cast<const std::string&>(&NDLibrary::get_nuclide,
                                                 py::const_),
           "Returns the :py:class:`NuclideHandle` of the the desired "
           "nuclide.\n\n"
           "Parameters\n"
           "----------\n"
           "name : str\n"
           "       Name of the desired nuclide.",
           py::arg("name"))

      .def("infinite_dilution_xs", &NDLibrary::infinite_dilution_xs,
           "Calculates the infinite dilution cross sections for the nuclide at "
           "the desired temperatures.\n\n"
           "Parameters\n"
           "----------\n"
           "name : str\n"
           "       Name of the desired nuclide.\n"
           "temp : float\n"
           "       Desired temperature in kelvin.\n"
           "max_l : int\n"
           "        Maximum legendre moment (default is 1).\n\n"
           "Returns\n"
           "-------\n"
           "MicroNuclideXS, MicroDepletionXS\n"
           "  Interpolated infinite dilution cross sections at desired "
           "temperature.")

      .def("dilution_xs", &NDLibrary::dilution_xs,
           "Interpolates the cross section of the prescribed nuclide at the "
           "prescribed energy group to the desired temperature and dilution. "
           "If the nuclide is not resonant or the desired group g is not "
           "resonant, an exception is raised.\n\n"
           "Parameters\n"
           "----------\n"
           "name : str\n"
           "       Name of the desired nuclide.\n"
           "g    : int\n"
           "       Energy group index.\n"
           "temp : float\n"
           "       Desired temperature in kelvin.\n"
           "dil : str\n"
           "      Desired dilution in barns.\n"
           "max_l : int\n"
           "        Maximum legendre moment (default is 1).\n\n"
           "Returns\n"
           "-------\n"
           "ResonantOneGroupXS\n"
           "  Interpolated cross sections in group g.",
           py::arg("name"), py::arg("g"), py::arg("temp"), py::arg("dil"),
           py::arg("max_l") = 1)

      .def(
          "two_term_xs", &NDLibrary::two_term_xs,
          "Uses the two-term rational approximation for self shielding of "
          "cross sections, where the fuel escape probability is approximated "
          "as \n\n"
          ".. math:: P^{F\\to M}(E) = \\beta_1\\frac{\\alpha_1 "
          "\\Sigma_e}{\\Sigma_t(E) + \\alpha_1\\Sigma_e} + "
          "\\beta_2\\frac{\\alpha_2 \\Sigma_e}{\\Sigma_t(E) + "
          "\\alpha_2\\Sigma_e}.\n\n"
          "If shielding isotope :math:`r` with potential cross section "
          ":math:`\\sigma_p^r` and number density :math:`N_r`, then the "
          "background cross sections are computed as\n\n"
          ".. math:: \\sigma_1 = \\frac{\\lambda\\Sigma_p - "
          "\\lambda_rN_r\\sigma_p^r + \\alpha_1\\Sigma_e}{N_r} \\\\\\\\"
          "          \\sigma_2 = \\frac{\\lambda\\Sigma_p - "
          "\\lambda_rN_r\\sigma_p^r + \\alpha_2\\Sigma_e}{N_r}\n\n"
          "where :math:`\\Sigma_p` is the macroscopic potential scattering "
          "cross section of the material, :math:`\\lambda_r` is the "
          "intermediate resonance parameter for isotope :math:`r`, and "
          ":math:`\\Sigma_e` is the escape cross section. If the nuclide is "
          "not resonant or the desired group g is not resonant, an exception "
          "is raised.\n\n"
          "Parameters\n"
          "----------\n"
          "name : str\n"
          "       Name of the nuclide to be treated.\n"
          "g    : int\n"
          "       Energy group index.\n"
          "temp : float\n"
          "       Temperature of the material (in kelvin).\n"
          "b1 : float\n"
          "     :math:`\\beta_1`.\n"
          "b2 : float\n"
          "     :math:`\\beta_2`.\n"
          "xs1 : float\n"
          "      Background cross section for first term (:math:`\\sigma_1`).\n"
          "xs2 : float\n"
          "      Background cross section for second term "
          "(:math:`\\sigma_2`).\n"
          "max_l : int\n"
          "        Maximum legendre moment (default is 1).\n\n"
          "Returns\n"
          "-------\n"
          "ResonantOneGroupXS\n"
          "  Interpolated cross sections in group g.\n",
          py::arg("name"), py::arg("g"), py::arg("temp"), py::arg("b1"),
          py::arg("b2"), py::arg("xs1"), py::arg("xs2"), py::arg("max_l") = 1)

      .def("ring_two_term_xs", &NDLibrary::ring_two_term_xs,
           "Uses the two-term rational approximation and the Stoker-Weiss "
           "method to produce the self-shielded cross sections for a single "
           "nuclide in a ring of fuel. If the nuclide is not resonant or the "
           "desired group g is not resonant, an exception is raised.\n\n"
           "Parameters\n"
           "----------\n"
           "name : str\n"
           "       Name of the nuclide to be treated.\n"
           "g    : int\n"
           "       Energy group index.\n"
           "temp : float\n"
           "       Temperature of the material (in kelvin).\n"
           "a1 : float\n"
           "     :math:`\\alpha_1`.\n"
           "a2 : float\n"
           "     :math:`\\alpha_2`.\n"
           "b1 : float\n"
           "     :math:`\\beta_1`.\n"
           "b2 : float\n"
           "     :math:`\\beta_2`.\n"
           "mat_pot_xs : float\n"
           "     Macroscopic potential cross section of material.\n"
           "N : float\n"
           "    Number density of the nuclie being shielded.\n"
           "Rfuel : float\n"
           "     Radius of the fuel pellet.\n"
           "Rin : float\n"
           "     Inner radius of the fuel ring.\n"
           "Rout : float\n"
           "     Outer radius of the fuel ring.\n"
           "max_l : int\n"
           "        Maximum legendre moment (default is 1).\n\n"
           "Returns\n"
           "-------\n"
           "ResonantOneGroupXS\n"
           "  Interpolated cross sections in group g.\n",
           py::arg("name"), py::arg("g"), py::arg("temp"), py::arg("a1"),
           py::arg("a2"), py::arg("b1"), py::arg("b2"), py::arg("mat_pot_xs"),
           py::arg("N"), py::arg("Rfuel"), py::arg("Rin"), py::arg("Rout"),
           py::arg("max_l") = 1)

      .def("unload", &NDLibrary::unload,
           "Deallocates all NuclideHandles which contained raw nuclear data.")

      .def_property_readonly("library", &NDLibrary::library,
                             "Name of the nuclear data library (if provided).")

      .def_property_readonly("ngroups", &NDLibrary::ngroups,
                             "Number of energy groups in the library.")

      .def_property_readonly(
          "first_resonant_group", &NDLibrary::first_resonant_group,
          "Index of the first resonant group in the library.")

      .def_property_readonly("last_resonant_group",
                             &NDLibrary::last_resonant_group,
                             "Index of the last resonant group in the library.")

      .def_property_readonly("group_bounds", &NDLibrary::group_bounds,
                             "The boundaries of the energy groups for the "
                             "group structure (in decreasing order).")

      .def_property_readonly("group_structure", &NDLibrary::group_structure,
                             "The name of the group structure (if provided).")

      .def_property_readonly(
          "condensation_scheme", &NDLibrary::condensation_scheme,
          "The condensation scheme to obtain the default few-group structure "
          "from the library group structure (if provided).")

      .def_property_readonly("cmfd_condensation_scheme",
                             &NDLibrary::cmfd_condensation_scheme,
                             "The CMFD condensation scheme for acceleration in "
                             "the MOC calculation (if provided).")

      .def_property_readonly(
          "depletion_chain", &NDLibrary::depletion_chain,
          "The DepletionChain intended for use with the library.")

      .def(py::pickle(
          [](const std::shared_ptr<NDLibrary>& ndl) {
            return py::make_tuple(ndl->fname());
          },
          [](py::tuple t) {
            return std::make_shared<NDLibrary>(t[0].cast<std::string>());
          }));
}
