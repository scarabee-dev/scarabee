#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pytensor.hpp>

#include <cereal/types/memory.hpp>
#include <cereal/archives/portable_binary.hpp>

#include <diffusion/diffusion_data.hpp>

namespace py = pybind11;

using namespace scarabee;

void init_DiffusionData(py::module& m) {
  py::enum_<DiffusionData::ADF>(m, "ADF")
      .value("XN", DiffusionData::ADF::XN, "Assembly x < 0 side.")
      .value("XP", DiffusionData::ADF::XP, "Assembly x > 0 side.")
      .value("YN", DiffusionData::ADF::YN, "Assembly y < 0 side.")
      .value("YP", DiffusionData::ADF::YP, "Assembly y > 0 side.")
      .value("ZN", DiffusionData::ADF::ZN, "Assembly z < 0 side.")
      .value("ZP", DiffusionData::ADF::ZP, "Assembly z > 0 side.");

  py::enum_<DiffusionData::CDF>(m, "CDF")
      .value("I", DiffusionData::CDF::I, "Corner of assembly in quadrant I.")
      .value("II", DiffusionData::CDF::II, "Corner of assembly in quadrant II.")
      .value("III", DiffusionData::CDF::III,
             "Corner of assembly in quadrant III.")
      .value("IV", DiffusionData::CDF::IV,
             "Corner of assembly in quadrant IV.");

  py::class_<DiffusionData, std::shared_ptr<DiffusionData>>(
      m, "DiffusionData",
      "A DiffusionData object contains all necessary cross section "
      "information to perform a diffusion calculation, along with "
      "assembly discontinuity factors and corner discontinuity factors.")

      .def(py::init<std::shared_ptr<DiffusionCrossSection> /*xs*/>(),
           "Creates a DiffusionData object.\n\n"
           "Parameters\n"
           "----------\n"
           "xs : DiffusionCrossSection\n"
           "     Diffusion cross sections for calculations.\n\n",
           py::arg("xs"))

      .def(py::init<std::shared_ptr<DiffusionCrossSection> /*xs*/,
                    const xt::xtensor<double, 2>& /*adf*/,
                    const xt::xtensor<double, 2>& /*cdf*/>(),
           "Creates a DiffusionData object.\n\n"
           "Parameters\n"
           "----------\n"
           "xs : DiffusionCrossSection\n"
           "     Diffusion cross sections for calculations.\n"
           "adf : ndarray\n"
           "    A 2D array of ADFs. The first axis is energy group, and the "
           "second axis is the side (Y+, X+, Y-, X-).\n"
           "cdf : ndarray\n"
           "    A 2D array of CDFs. The first axis is energy group, and the "
           "second axis is the corner (I, II, III, IV).\n\n",
           py::arg("xs"), py::arg("adf"), py::arg("cdf"))

      .def_property_readonly("ngroups", &DiffusionData::ngroups,
                             "Number of energy groups.")

      .def_property_readonly("xs_name", &DiffusionData::xs_name,
                             "Name of the DiffusionCrossSection.")

      .def_property_readonly("xs", &DiffusionData::xs,
                             "The contained DiffusionCrossSection object.")

      .def_property_readonly("fissile", &DiffusionData::fissile,
                             "True if material is fissile.")

      .def_property("name", &DiffusionData::name, &DiffusionData::set_name,
                    "Name of material.")

      .def_property("leakage_corrections", &DiffusionData::leakage_corrections,
                    &DiffusionData::set_leakage_corrections,
                    "Optional LeakageCorrection data to update cross sections "
                    "based on in-situ leakage in nodal solver.")

      .def_property("reflector", &DiffusionData::reflector,
                    &DiffusionData::set_reflector,
                    "Flag to indicate the data is for a reflector, and the "
                    "ADFs should be multiplied by the adjacent fuel ADFs.")

      .def_property("adf", &DiffusionData::adf, &DiffusionData::set_adf,
                    "Assembly discontinuity factors.")

      .def_property("cdf", &DiffusionData::cdf, &DiffusionData::set_cdf,
                    "Corner discontinuity factors.")

      .def("D", &DiffusionData::D,
           "Diffusion coefficient in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("Ea", &DiffusionData::Ea,
           "Absorption cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("Ef", &DiffusionData::Ef,
           "Fission cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("vEf", &DiffusionData::vEf,
           "Fission yield * cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("nu", &DiffusionData::nu,
           "Fission yield in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("Er", &DiffusionData::Er,
           "Removal cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("chi", &DiffusionData::chi,
           "Fission spectrum in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("Es", py::overload_cast<std::size_t>(&DiffusionData::Es, py::const_),
           "Transport corrected scattering cross section in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.\n\n",
           py::arg("g"))

      .def("Es",
           py::overload_cast<std::size_t, std::size_t>(&DiffusionData::Es,
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

      .def("adf_xp", &DiffusionData::adf_xp,
           "The ADS for the x+ surface in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("adf_xn", &DiffusionData::adf_xn,
           "The ADS for the x- surface in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("adf_yp", &DiffusionData::adf_yp,
           "The ADS for the y+ surface in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("adf_yn", &DiffusionData::adf_yn,
           "The ADS for the y- surface in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("adf_zp", &DiffusionData::adf_zp,
           "The ADS for the z+ surface in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("adf_zn", &DiffusionData::adf_zn,
           "The ADS for the z- surface in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("cdf_I", &DiffusionData::cdf_I,
           "The CDF for the quadrant I corner in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("cdf_II", &DiffusionData::cdf_II,
           "The CDF for the quadrant II corner in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("cdf_III", &DiffusionData::cdf_III,
           "The CDF for the quadrant III corner in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("cdf_IV", &DiffusionData::cdf_IV,
           "The CDF for the quadrant IV corner in group g.\n\n"
           "Parameters\n"
           "----------\n"
           "g : int\n"
           "    Energy group.",
           py::arg("g"))

      .def("rotate_clockwise", &DiffusionData::rotate_clockwise,
           "Rotates the ADFs corresponding to a 90 degree "
           "rotation of the assembly in the clockwise direction.",
           py::return_value_policy::reference_internal)

      .def("rotate_counterclockwise", &DiffusionData::rotate_counterclockwise,
           "Rotates the ADFs corresponding to a 90 degree "
           "rotation of the assembly in the counter clockwise direction.",
           py::return_value_policy::reference_internal)

      .def("reflect_across_x_axis", &DiffusionData::reflect_across_x_axis,
           "Performs a reflection of the ADFs across the x "
           "axis. A reflection across the x axis means that the +y and -y ADFs "
           "are swapped.",
           py::return_value_policy::reference_internal)

      .def("reflect_across_y_axis", &DiffusionData::reflect_across_y_axis,
           "Performs a reflection of the ADFs across the y "
           "axis. A reflection across the y axis means that the +x and -x ADFs "
           "are swapped.",
           py::return_value_policy::reference_internal)

      .def("save", &DiffusionData::save,
           "Saves the diffuion data to a binary file.\n\n"
           "Parameters\n"
           "----------\n"
           "fname : str\n"
           "        Name of file in which to save data.",
           py::arg("fname"))

      .def_static("load", &DiffusionData::load,
                  "Loads diffusion data from a binary file.\n\n"
                  "Parameters\n"
                  "----------\n"
                  "fname : str\n"
                  "        Name of file from which to load data.\n\n"
                  "Returns\n"
                  "-------\n"
                  "DiffusionData\n"
                  "    Diffusion cross sections and ADF from the file.\n",
                  py::arg("fname"))

      .def("__deepcopy__",
           [](const DiffusionData& dd, py::dict) {
             DiffusionData out(
                 std::make_shared<DiffusionCrossSection>(*dd.xs()));
             out.set_adf(dd.adf());
             out.set_cdf(dd.cdf());
             out.set_leakage_corrections(dd.leakage_corrections());
             return out;
           })

      .def(py::pickle([](const DiffusionData& d) { return d.to_tuple(); },
                      [](py::tuple t) { return DiffusionData(t); }));
}
