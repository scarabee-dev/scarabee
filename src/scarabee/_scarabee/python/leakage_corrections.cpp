#include <pybind11/pybind11.h>

#include <diffusion/leakage_corrections.hpp>

namespace py = pybind11;

using namespace scarabee;

void init_LeakageCorrections(py::module& m) {
  py::class_<LeakageCorrections>(m, "LeakageCorrections")
      .def(py::init<std::size_t>(),
           "Creates a LeakageCorrections object which contains coefficients to "
           "correct few-group diffusion cross sections for the actual "
           "experienced leakage. All coefficients are initialized to zero.\n\n"
           "Parameters\n"
           "----------\n"
           "ngroups : int\n"
           "    Number of energy groups.\n\n",
           py::arg("ngroups"))

      .def_property_readonly("ngroups", &LeakageCorrections::ngroups,
                             "Number of energy groups.")

      .def("D", &LeakageCorrections::D,
           "Obtains the correction coefficient for the diffusion coefficient "
           "in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group index\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "  Leakage correction coefficient for group g.\n\n",
           py::arg("g"))
      .def("set_D", &LeakageCorrections::set_D,
           "Sets the correction coefficient for the diffusion coefficient in "
           "group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group index\n"
           "val : float\n"
           "    New value of the correction coefficient\n\n",
           py::arg("g"), py::arg("val"))

      .def("Ea", &LeakageCorrections::Ea,
           "Obtains the correction coefficient for the absorption cross "
           "section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group index\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "  Leakage correction coefficient for group g.\n\n",
           py::arg("g"))
      .def("set_Ea", &LeakageCorrections::set_Ea,
           "Sets the correction coefficient for the absorption cross section "
           "in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group index\n"
           "val : float\n"
           "    New value of the correction coefficient\n\n",
           py::arg("g"), py::arg("val"))

      .def("Ef", &LeakageCorrections::Ef,
           "Obtains the correction coefficient for the fission cross section "
           "in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group index\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "  Leakage correction coefficient for group g.\n\n",
           py::arg("g"))
      .def("set_Ef", &LeakageCorrections::set_Ef,
           "Sets the correction coefficient for the fission cross section in "
           "group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group index\n"
           "val : float\n"
           "    New value of the correction coefficient\n\n",
           py::arg("g"), py::arg("val"))

      .def("vEf", &LeakageCorrections::vEf,
           "Obtains the correction coefficient for the nu-fission cross "
           "section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group index\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "  Leakage correction coefficient for group g.\n\n",
           py::arg("g"))
      .def("set_vEf", &LeakageCorrections::set_vEf,
           "Sets the correction coefficient for the nu-fission cross section "
           "in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group index\n"
           "val : float\n"
           "    New value of the correction coefficient\n\n",
           py::arg("g"), py::arg("val"))

      .def("Es", &LeakageCorrections::Es,
           "Obtains the correction coefficient for the scattering cross "
           "section from group g_in to group g_out. Only down-scattering cross "
           "sections are available (g_out > g_in).\n\n"
           "Parameters\n"
           "----------\n"
           "g_in : int\n"
           "    Incident energy group index\n"
           "g_out : int\n"
           "    Outgoing energy group index\n\n"
           "Returns\n"
           "-------\n"
           "float\n"
           "  Leakage correction coefficient for group g.\n\n",
           py::arg("g_in"), py::arg("g_out"))
      .def("set_Es", &LeakageCorrections::set_Es,
           "Sets the correction coefficient for the scattering cross section "
           "from group g_in to group g_out. Only down-scattering cross "
           "sections are available (g_out > g_in).\n\n"
           "Parameters\n"
           "----------\n"
           "g_in : int\n"
           "    Incident energy group index\n"
           "g_out : int\n"
           "    Outgoing energy group index\n"
           "val : float\n"
           "    New value of the correction coefficient\n\n",
           py::arg("g_in"), py::arg("g_out"), py::arg("val"))

      .def(py::pickle([](const LeakageCorrections& l) { return l.to_tuple(); },
                      [](py::tuple t) { return LeakageCorrections(t); }));
}
